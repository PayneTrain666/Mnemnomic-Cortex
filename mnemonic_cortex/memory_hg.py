import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from .utils import fast_pairwise_l2

def _complex_from_phase(phase: torch.Tensor) -> torch.Tensor:
    """phase: (..., D) real -> complex unit vector e^{i phase}."""
    return torch.polar(torch.ones_like(phase), phase)

class EnhancedHyperGeometricMemory(nn.Module):
    """HyperGeometric-ish memory with:
      • Fractal addressing over real keys
      • Quantum-inspired holographic storage per slot (complex HRR via FFT)
      • Entangler phases for key/value mixing (unitary diagonal)
      • Usage tracking + active-slots + consolidation
    Returns (B,S,input_dim) on read.
    """
    def __init__(self, input_dim: int, manifold_dim: int = 24, mem_slots: int = 1024,
                 quantum_qubits: int = 8, topk: int = 32, fractal_scales: int = 4,
                 holo_dim: int = 256, holo_lr: float = 0.1, holo_decay: float = 0.98,
                 ann_centroids: int = 256, ann_top_centroids: int = 8,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.D = manifold_dim
        self.M = mem_slots
        self.Q = quantum_qubits
        self.K_base = min(topk, mem_slots)
        self.SCALES = fractal_scales
        self.holo_dim = int(2 ** math.ceil(math.log2(max(64, holo_dim))))
        self.holo_lr = float(holo_lr)
        self.holo_decay = float(holo_decay)

        # ---------- Micro-batch update buffers ----------
        self._upd_idx_buffer = []   # list[Tensor]
        self._upd_val_buffer = []   # list[Tensor]
        self.flush_every = 4        # fuse updates every N writes

        # Addressing keys/values (real) for fractal search
        self.keys   = nn.Parameter(torch.randn(self.M, self.D))
        self.values = nn.Parameter(torch.randn(self.M, self.D * 3))  # legacy/fallback

        # ---------- Coarse quantiser for ANN search ----------
        self.use_ann = ann_centroids > 0 and ann_centroids < self.M
        self.C = ann_centroids
        self.C_top = ann_top_centroids
        if self.use_ann:
            self.register_buffer("centroids", torch.randn(self.C, self.D))
            # assignment of each key to centroid id (M,)
            self.register_buffer("centroid_ids", torch.zeros(self.M, dtype=torch.long))
            self._recompute_centroids()

        # Quantum-ish extras
        self.quantum_phase = nn.Parameter(torch.randn(self.M, self.Q))
        self.quantum_gain  = nn.Parameter(torch.ones(self.M))

        # Ricci flow operator across the manifold dimension
        self.ricci_flow = nn.Parameter(torch.eye(self.D))

        # Projections
        self.input_projection  = nn.Sequential(nn.Linear(input_dim, self.D * 3), nn.LayerNorm(self.D * 3), nn.GELU())
        self.output_projection = nn.Sequential(nn.Linear(self.D * 3, input_dim), nn.LayerNorm(input_dim), nn.GELU())

        # Learnable weights per fractal scale
        self.fractal_weights = nn.Parameter(torch.ones(self.SCALES))

        # Temperature used to scale distances (can be modulated externally)
        self.register_buffer("temperature", torch.tensor(1.0))

        # Usage tracking + active slots
        self.register_buffer("usage_counts", torch.zeros(self.M, dtype=torch.float32))
        self.active_slots = self.M  # may be reduced in energy-efficient mode
        
        # Collision control
        self.soft_capacity_per_slot = float(self.M * 10)  # max hits per slot before penalty
        self.collision_penalty_weight = 0.1  # weight for usage penalty in distance

        # ---------- Holographic (complex) associative memory ----------
        # Complex-valued holograms in frequency domain (superposition of bound pairs)
        self.holograms_fft = nn.Parameter(torch.zeros(self.M, self.holo_dim, dtype=torch.complex64))

        # Key/Value phase encoders -> complex unit sequences
        self.key_phase  = nn.Linear(input_dim, self.holo_dim)
        self.val_phase  = nn.Linear(input_dim, self.holo_dim)

        # Optional entangler phases (unitary diagonals in freq domain)
        self.entangle_key = nn.Parameter(torch.zeros(self.holo_dim))
        self.entangle_val = nn.Parameter(torch.zeros(self.holo_dim))

        # Readout from retrieved holographic vector to manifold triplet
        self.readout = nn.Linear(2 * self.holo_dim, self.D * 3)

        # Lightbulb internal tracking (running avg of top-1 distance)
        self.register_buffer('lb_top1_avg', torch.tensor(1.0))
        self.lb_momentum = 0.99
        self.lb_drop_ratio = 0.7  # fire if current < 0.7 * avg

    @torch.no_grad()
    def _recompute_centroids(self, iters: int = 10):
        """Lightweight k-means on self.keys to (re)initialise centroids."""
        if not self.use_ann:
            return
        k = self.keys.data.clone()
        # init centroids from random keys
        idx = torch.randperm(self.M)[:self.C]
        c = k[idx].clone()
        for _ in range(iters):
            dist = fast_pairwise_l2(k, c)          # (M,C)
            assign = dist.argmin(dim=-1)           # (M,)
            # compute new centroids
            for ci in range(self.C):
                mask = assign == ci
                if mask.any():
                    c[ci] = k[mask].mean(dim=0)
        self.centroids.copy_(c)
        self.centroid_ids.copy_(assign)

    # -------------------- Controls --------------------
    def set_temperature(self, t: torch.Tensor):
        if t.numel() == 1:
            self.temperature = t.detach()
        else:
            self.temperature = t.mean().detach()

    def set_active_fraction(self, frac: float):
        frac = float(max(0.1, min(1.0, frac)))
        self.active_slots = max(8, int(self.M * frac))

    def enable_energy_efficient_mode(self, enable: bool = True):
        self.set_active_fraction(0.5 if enable else 1.0)

    # ------------ DDP sync ---------------
    @torch.no_grad()
    def _sync_buffers_ddp(self):
        """Synchronize external memory parameters/buffers across DDP ranks."""
        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            return
        for t in [self.holograms_fft.data, self.keys.data, self.values.data, self.usage_counts]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t /= dist.get_world_size()

    @torch.no_grad()
    def consolidate_unused(self, threshold: float = 0.1, ema: float = 0.9):
        """Average rarely used slots back toward the global mean; reset their counts."""
        if self.usage_counts.max() <= 0:
            return
        usage_ratio = self.usage_counts / (self.usage_counts.max() + 1e-6)
        mask = usage_ratio < threshold
        if mask.any():
            k_mean = self.keys[~mask].mean(dim=0, keepdim=True) if (~mask).any() else self.keys.mean(dim=0, keepdim=True)
            v_mean = self.values[~mask].mean(dim=0, keepdim=True) if (~mask).any() else self.values.mean(dim=0, keepdim=True)
            self.keys.data[mask] = ema * self.keys.data[mask] + (1-ema) * k_mean
            self.values.data[mask] = ema * self.values.data[mask] + (1-ema) * v_mean
            # gentle decay on holograms
            self.holograms_fft.data[mask] *= self.holo_decay
            self.usage_counts[mask] = 0.0

    @torch.no_grad()
    def defragment_slots(self, n_clusters: int = None, iters: int = 10):
        """Redistribute memories via k-means on value embeddings to reduce collisions."""
        if n_clusters is None:
            n_clusters = self.M
        
        # Use values as embedding space for clustering
        vals = self.values.data.clone()  # (M, D*3)
        
        # Simple k-means
        idx = torch.randperm(self.M)[:n_clusters]
        centroids = vals[idx].clone()
        
        for _ in range(iters):
            dist = fast_pairwise_l2(vals, centroids)  # (M, n_clusters)
            assign = dist.argmin(dim=-1)  # (M,)
            # Update centroids
            for ci in range(n_clusters):
                mask = assign == ci
                if mask.any():
                    centroids[ci] = vals[mask].mean(dim=0)
        
        # Reassign keys to balance load (spread assignments evenly)
        for ci in range(n_clusters):
            mask = assign == ci
            if mask.any():
                # Reset usage counts for this cluster
                self.usage_counts[mask] *= 0.5

    def get_metrics(self):
        """Return dict of diagnostic metrics."""
        M = self.active_slots
        holo_rms = self.holograms_fft[:M].abs().mean(dim=-1)  # (M,)
        return {
            'hg_temp': self.temperature.mean().item() if self.temperature.numel() > 1 else self.temperature.item(),
            'hg_topk_base': self.K_base,
            'hg_active_slots': M,
            'hg_holo_rms_mean': holo_rms.mean().item(),
            'hg_holo_rms_std': holo_rms.std().item(),
            'hg_holo_rms_max': holo_rms.max().item(),
            'hg_usage_mean': self.usage_counts[:M].mean().item(),
            'hg_usage_max': self.usage_counts[:M].max().item(),
            'hg_lb_top1_avg': self.lb_top1_avg.item(),
        }

    # -------------------- Core ops --------------------
    def encode_to_manifold(self, x):
        # x: (B,S,input_dim) -> (B,S,D,3)
        B, S, _ = x.shape
        z = self.input_projection(x).view(B, S, self.D, 3)     # (B,S,D,3)
        # Apply Ricci flow across D: z[b,s,:,q] = sum_e z[b,s,e,q] * R[e,d]
        z = torch.einsum('bseq,ed->bsdq', z, self.ricci_flow)  # (B,S,D,3)
        return z

    @torch.no_grad()
    def _fractal_cdist(self, q: torch.Tensor, k: torch.Tensor):
        if self.use_ann:
            return self._fractal_cdist_ann(q)
        # original brute force
        # q: (B,S,D)  k: (M,D) -> (B,S,M)
        weights = torch.softmax(self.fractal_weights, dim=0)
        dsum = 0.0
        for s, w in enumerate(weights):
            sf = 2 ** s
            d = fast_pairwise_l2(q / sf, k / sf)  # (B,S,M or M_active)
            dsum = dsum + w * d
        return dsum

    @torch.no_grad()
    def _fractal_cdist_ann(self, q: torch.Tensor):
        """Two-stage ANN search: centroids shortlist -> key subset."""
        B,S,D = q.shape
        # stage 1: distance to centroids
        wts = torch.softmax(self.fractal_weights, dim=0)
        d_cent = 0.0
        for s,w in enumerate(wts):
            sf = 2 ** s
            d_cent = d_cent + w * fast_pairwise_l2(q / sf, self.centroids / sf)  # (B,S,C)
        # select top-C_top centroids per query token
        _, topc = torch.topk(d_cent, k=min(self.C_top, self.C), dim=-1, largest=False)  # (B,S,Ctop)
        # build mask for keys belonging to selected centroids
        device = q.device
        mask = torch.zeros(self.M, device=device, dtype=torch.bool)
        # gather unique centroid ids in batch for efficiency
        uniq_c = torch.unique(topc)
        mask_cent = torch.isin(self.centroid_ids.to(device), uniq_c)
        candid_keys = self.keys[mask_cent]
        # fallback to brute if too many candidates removed
        if candid_keys.size(0) < 32:
            candid_keys = self.keys
        # compute full fractal distance on restricted keys
        dsum = 0.0
        for s, w in enumerate(wts):
            sf = 2 ** s
            d = fast_pairwise_l2(q / sf, candid_keys / sf)  # (B,S,M')
            dsum = dsum + w * d
        # need to map distances back to full M size for downstream logic
        # create large tensor filled with inf
        dist_full = torch.full((B,S,self.M), float('inf'), device=device, dtype=dsum.dtype)
        idx_full = torch.nonzero(mask_cent, as_tuple=False).view(-1)
        dist_full[:,:,idx_full] = dsum
        return dist_full

    @torch.no_grad()
    def _record_usage(self, indices):
        # indices: (B,S,K)
        flat = indices.reshape(-1)
        self.usage_counts.index_add_(0, flat, torch.ones_like(flat, dtype=self.usage_counts.dtype))

    # --------- Holographic helpers ---------
    def _key_complex(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,S,input_dim) -> complex time-domain (B,S,holo_dim)
        phase = 2 * math.pi * torch.sigmoid(self.key_phase(x))
        return _complex_from_phase(phase)

    def _val_complex(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,S,input_dim) -> complex time-domain (B,S,holo_dim)
        phase = 2 * math.pi * torch.sigmoid(self.val_phase(x))
        return _complex_from_phase(phase)

    def _entangle_fft(self, fft_vec: torch.Tensor, which: str) -> torch.Tensor:
        # fft_vec: (..., holo_dim) complex
        if which == 'key':
            e = _complex_from_phase(self.entangle_key)
        else:
            e = _complex_from_phase(self.entangle_val)
        e = e.to(fft_vec.device).to(fft_vec.dtype)
        return fft_vec * e

    # --------- Write/Read in frequency domain ---------
    @torch.no_grad()
    def _holo_write(self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, lr_mult: float = 1.0):
        """Bind key/value and add to selected holograms via EMA.
        x: (B,S,input_dim)
        indices: (B,S,K), weights: (B,S,K)
        """
        B,S,_ = x.shape
        M = self.active_slots
        kc = self._key_complex(x)             # (B,S,H)
        vc = self._val_complex(x)             # (B,S,H)
        Kf = torch.fft.fft(kc, dim=-1)        # (B,S,H)
        Vf = torch.fft.fft(vc, dim=-1)        # (B,S,H)
        Kf = self._entangle_fft(Kf, 'key')
        Vf = self._entangle_fft(Vf, 'val')
        Kv = (Kf * Vf).unsqueeze(2)           # (B,S,1,H)
        contrib = weights.unsqueeze(-1) * Kv  # (B,S,K,H)
        flat_idx = indices.reshape(-1)        # (B*S*K,)
        flat_contrib = contrib.reshape(-1, self.holo_dim)  # (B*S*K,H)
        # ---- Micro-batch accumulation ----
        self._upd_idx_buffer.append(flat_idx)
        self._upd_val_buffer.append(flat_contrib)

        if len(self._upd_idx_buffer) >= self.flush_every:
            self._flush_pending_updates(lr_mult)

    @torch.no_grad()
    def _flush_pending_updates(self, lr_mult: float = 1.0):
        if not self._upd_idx_buffer:
            return
        flat_idx = torch.cat(self._upd_idx_buffer, dim=0)
        flat_contrib = torch.cat(self._upd_val_buffer, dim=0)
        self._upd_idx_buffer.clear()
        self._upd_val_buffer.clear()

        M = self.active_slots
        device = self.holograms_fft.device
        accum = torch.zeros((M, self.holo_dim), dtype=self.holograms_fft.dtype, device=device)
        accum.index_add_(0, flat_idx, flat_contrib)
        counts = torch.zeros(M, device=device, dtype=accum.real.dtype).index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=accum.real.dtype))
        counts = counts.clamp_min_(1.0).unsqueeze(-1).to(accum.dtype)
        avg = accum / counts

        lr = self.holo_lr * lr_mult
        self.holograms_fft.data[:M] = self.holo_decay * self.holograms_fft.data[:M] + lr * avg

        # After applying, run clamp/sync steps
        rms = self.holograms_fft.abs().mean(dim=-1, keepdim=True)
        max_rms = 5.0
        scale = (max_rms / rms).clamp(max=1.0)
        self.holograms_fft.data[:M] *= scale
        self._sync_buffers_ddp()

    def _holo_read(self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Unbind with conj(key) from selected holograms; weight and sum.
        Returns a (B,S,2*holo_dim) real feature (real||imag of time-domain retrieval).
        """
        B,S,_ = x.shape
        M = self.active_slots
        kc = self._key_complex(x)                             # (B,S,H)
        Kf = torch.fft.fft(kc, dim=-1)                        # (B,S,H)
        Kf = self._entangle_fft(Kf, 'key')
        Hsel = self.holograms_fft[:M][indices]                # (B,S,K,H) complex
        Vhatf = torch.conj(Kf).unsqueeze(2) * Hsel            # (B,S,K,H)
        Vhatf = (weights.unsqueeze(-1) * Vhatf).sum(dim=2)    # (B,S,H)
        vhat = torch.fft.ifft(Vhatf, dim=-1)                  # (B,S,H) complex
        feat = torch.cat([vhat.real, vhat.imag], dim=-1)      # (B,S,2H)
        return feat

    def forward(self, x: torch.Tensor, operation: str = 'read', fire_mask=None, recall_boost: float = 0.3):
        B, S, _ = x.shape
        z = self.encode_to_manifold(x)           # (B,S,D,3)
        q = z.mean(dim=3)                        # (B,S,D)
        M = self.active_slots
        k_active = self.keys[:M]
        dist = self._fractal_cdist(q, k_active)  # (B,S,M)
        
        # Apply collision penalty: penalize over-used slots
        usage_penalty = (self.usage_counts[:M] / self.soft_capacity_per_slot).clamp(0, 1.0)
        dist = dist + self.collision_penalty_weight * usage_penalty.view(1, 1, M)

        # temperature: allow per-batch explosive recall scaling
        if fire_mask is not None and isinstance(fire_mask, torch.Tensor) and fire_mask.any():
            t_eff = (self.temperature / (1.0 + recall_boost * fire_mask.float())).view(-1,1,1)
        elif isinstance(fire_mask, bool) and fire_mask:
            t_eff = self.temperature / (1.0 + recall_boost)
        else:
            t_eff = self.temperature
        dist = dist / t_eff.clamp_min(1e-6)

        # Determine effective K (boost on lightbulb)
        K = min(self.K_base, M)
        dtmp, _ = torch.topk(dist, max(1,K), dim=-1, largest=False)
        top1 = dtmp[...,0].mean(dim=1).mean(dim=0) if dtmp.dim()==3 else dtmp.mean()
        # update running avg (scalar)
        self.lb_top1_avg = self.lb_momentum * self.lb_top1_avg + (1-self.lb_momentum) * top1.detach()
        internal_fire = bool(top1 < self.lb_drop_ratio * self.lb_top1_avg)
        any_fire = internal_fire or (
            isinstance(fire_mask, torch.Tensor) and fire_mask.any()
        ) or (
            isinstance(fire_mask, bool) and fire_mask
        )
        if any_fire:
            K = min(M, max(K, int(self.K_base * 1.5)))

        dtop, itop = torch.topk(dist, K, dim=-1, largest=False)  # (B,S,K)
        w = torch.softmax(-dtop, dim=-1)                         # (B,S,K)

        self._record_usage(itop)

        if operation == 'write':
            lr_mult = 1.0 + (0.5 if any_fire else 0.0)
            self._holo_write(x, itop, w, lr_mult=lr_mult)
            return x

        holo_feat = self._holo_read(x, itop, w)                 # (B,S,2H)
        triplet = self.readout(holo_feat)                       # (B,S,3D)
        out = self.output_projection(triplet)                   # (B,S,input_dim)

        phases = torch.tanh(self.quantum_phase[:M][itop]).mean(dim=-1)  # (B,S,K)
        gain = 1.0 + 0.02 * (w * phases).sum(dim=-1, keepdim=True)      # (B,S,1)
        if any_fire:
            gain = gain * (1.0 + recall_boost)
        return out * gain
