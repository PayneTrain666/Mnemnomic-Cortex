import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict

# --------- Quaternion utils (spin channel) ---------
def qnormalize(q: torch.Tensor, eps=1e-9):
    # q shape [..., 4] with (w, x, y, z)
    return q / (q.norm(dim=-1, keepdim=True) + eps)

def qangle(q1: torch.Tensor, q2: torch.Tensor, eps=1e-7):
    """
    Smallest rotation angle between two unit quaternions.
    q1, q2: [..., 4], assumed near-unit; handles the q ~ -q equivalence via abs(dot).
    Returns angle in radians, shape [...,]
    """
    q1 = qnormalize(q1)
    q2 = qnormalize(q2)
    dot = torch.sum(q1 * q2, dim=-1).abs().clamp_(0.0, 1.0 - eps)
    return 2.0 * torch.acos(dot)

# --------- Metric warps (monotone, stable) ---------
def euc_warp(dist: torch.Tensor) -> torch.Tensor:
    return dist

def hyp_warp(dist: torch.Tensor, slot_c: torch.Tensor, alpha=0.6) -> torch.Tensor:
    # Hyperbolic-like steepening; slot curvature shapes slope.
    # asinh is smooth/monotone; tanh(curv) bounds influence.
    return torch.asinh((1.0 + alpha * torch.tanh(slot_c)) * dist)

def sph_warp(dist: torch.Tensor, gamma=0.9) -> torch.Tensor:
    # Spherical-like compression for large distances; keep monotone and positive
    return (2.0 * torch.sin(0.5 * gamma * dist)).abs()

# --------- Colored 1/f^alpha noise for exploration ---------
@torch.no_grad()
def colored_noise_1f(size: int, device, dtype, alpha=1.0) -> torch.Tensor:
    size = int(max(1, size))
    n_freq = size // 2 + 1
    freqs = torch.arange(1, n_freq + 1, device=device, dtype=torch.float32)
    mag = (1.0 / freqs.pow(float(alpha) * 0.5)).to(dtype=torch.float32)
    phase = torch.exp(1j * 2.0 * math.pi * torch.rand(n_freq, device=device))
    spectrum = (phase * mag.to(phase.dtype)).to(torch.complex64)
    spectrum[0] = 0.0 + 0.0j
    noise = torch.fft.irfft(spectrum, n=size).real
    noise = noise - noise.mean()
    noise = noise / noise.std(unbiased=False).clamp_min(1e-6)
    return noise.to(dtype)

# ============================================================
#                  Topology Manager V3
# ============================================================
class TopologyManagerV3(nn.Module):
    """
    Geometry/meta-controller for HG / CGMN / Curved subsystems.

    Features:
      • EMA fitness + hysteresis per subsystem
      • Slot-level curvature with bounded, loss-aware mutations (incl. 1/f fractal)
      • Mixture-of-metrics (Euclidean / Hyperbolic / Spherical / Spin) via a small gate
      • Spin coupling (optional): add a spin-distance term from quaternions
      • Single call warp_and_blend(...) to apply before softmax

    Contracts:
      • memory_curvature: [mem_slots] OR [mem_slots, D] (we mean-reduce internally)
      • dist, indices: [B, K] from your sampler/top-k
      • q_query: [B, 4], q_memory: [mem_slots, 4]   (optional; if None → no spin term)
    """
    def __init__(
        self,
        subsystems=("hg", "cgmn", "curved"),
        mutation_rate=0.01,
        ema_decay=0.9,
        hysteresis=0.03,
        curv_clamp=1.5,
        fractal_alpha=1.0,
        default_beta=0.12,      # curvature warp gain
        spin_scale=0.7,         # scales spin-angle contribution
        gate_nudge=0.5,
    ):
        super().__init__()
        self.subsystems = list(subsystems)

        # Per-subsystem state
        self._ema_fit: Dict[str, Optional[float]] = {s: None for s in self.subsystems}
        self._topo: Dict[str, str] = {s: "hyperbolic" for s in self.subsystems}

        # thresholds
        self._t_hyp, self._t_sph, self._t_euc = 0.80, 0.60, 0.40
        self.hysteresis = hysteresis
        self.ema_decay = ema_decay

        # mutation / bounds
        self.mutation_rate = mutation_rate
        self.curv_clamp = curv_clamp
        self.fractal_alpha = fractal_alpha

        # warp gains
        self.default_beta = default_beta
        self.spin_scale = spin_scale
        self.gate_nudge = gate_nudge

        # Lightweight learned gate: inputs=[fitness, entropy, mean|curv|, spin_coh] → weights over 4 metrics
        self.gates = nn.ModuleDict({
            s: nn.Sequential(
                nn.Linear(4, 32),
                nn.SiLU(),
                nn.Linear(32, 4),   # [euc, hyp, sph, spin]
                nn.Softmax(dim=-1)
            ) for s in self.subsystems
        })

        # Learn per-topology curvature warp strength if you like (optional; start at default_beta)
        self.beta_table = nn.ParameterDict({
            # order: euc, hyp, sph, spin   (spin uses separate scale)
            s: nn.Parameter(torch.tensor([self.default_beta, self.default_beta, self.default_beta, self.spin_scale], dtype=torch.float32))
            for s in self.subsystems
        })
        self.mode_bias = {
            "stabilise": torch.tensor([+0.3, +0.3, +0.1, +0.1, 0.0, 0.0]),
            "structured": torch.tensor([0.0, +0.4, +0.2, 0.0, +0.2, +0.2]),
            "explore": torch.tensor([+0.2, 0.0, +0.2, +0.3, +0.1, +0.2]),
            "spinquant": torch.tensor([0.0, 0.0, +0.1, 0.0, +0.6, +0.3]),
        }

    # ---------- EMA & hysteresis ----------
    def _hysteresis_thresholds(self, subsystem: str):
        h = self.hysteresis
        cur = self._topo[subsystem]
        if cur == 'hyperbolic': return self._t_hyp - h, self._t_sph, self._t_euc
        if cur == 'spherical':  return self._t_hyp, self._t_sph - h, self._t_euc
        if cur == 'euclidean':  return self._t_hyp, self._t_sph, self._t_euc - h
        return self._t_hyp, self._t_sph, self._t_euc  # fractal

    def _resolve_subsystem(self, subsystem: str) -> str:
        if subsystem in self._topo:
            return subsystem
        if not self.subsystems:
            raise ValueError("TopologyManagerV3 has no registered subsystems")
        return self.subsystems[0]

    def evolve_topology(self, fitness: float, subsystem: str) -> str:
        subsystem = self._resolve_subsystem(subsystem)
        f_old = self._ema_fit[subsystem]
        f_new = float(fitness) if f_old is None else self.ema_decay * f_old + (1 - self.ema_decay) * float(fitness)
        self._ema_fit[subsystem] = f_new

        t_h, t_s, t_e = self._hysteresis_thresholds(subsystem)
        if f_new >= t_h: topo = 'hyperbolic'
        elif f_new >= t_s: topo = 'spherical'
        elif f_new >= t_e: topo = 'euclidean'
        else: topo = 'fractal'
        self._topo[subsystem] = topo
        return topo

    def _mode_from_fitness(self, fitness: float) -> str:
        if fitness >= 0.80:
            return "structured"
        if fitness >= 0.60:
            return "stabilise"
        if fitness >= 0.40:
            return "explore"
        return "spinquant"

    @torch.no_grad()
    def steer_merger(self, merger, subsystem: str):
        """Nudge GeometryMergerV2 channel logits based on subsystem fitness."""
        if not hasattr(merger, "set_gate_bias"):
            return
        subsystem = self._resolve_subsystem(subsystem)
        fit = self._ema_fit.get(subsystem, None)
        fit = 0.5 if fit is None else float(fit)
        mode = self._mode_from_fitness(fit)
        bias = self.mode_bias[mode].to(merger.gate_bias.device, merger.gate_bias.dtype)
        merger.set_gate_bias(bias.clamp(-1, 1) * self.gate_nudge)

    # ---------- curvature helpers ----------
    @staticmethod
    def _ensure_slot_scalar(curvature: torch.Tensor) -> torch.Tensor:
        if curvature.dim() == 1: return curvature
        if curvature.dim() == 2: return curvature.mean(dim=1)
        raise ValueError(f"curvature must be 1D or 2D, got {list(curvature.shape)}")

    @torch.no_grad()
    def mutate_curvature(self, curvature: torch.Tensor, loss_value: float, subsystem: str) -> torch.Tensor:
        subsystem = self._resolve_subsystem(subsystem)
        orig_shape = curvature.shape
        device, dtype = curvature.device, curvature.dtype
        slot_c = self._ensure_slot_scalar(curvature).clone()

        # loss-aware amplitude
        fitness = 1.0 / (1.0 + float(loss_value))
        amp = self.mutation_rate * (1.0 - fitness)
        amp = float(max(1e-6, min(self.mutation_rate, amp)))

        topo = self._topo[subsystem]
        if topo == 'hyperbolic':
            noise = torch.randn_like(slot_c) * amp
            slot_c = slot_c + noise - 0.25 * amp
        elif topo == 'spherical':
            noise = torch.randn_like(slot_c) * amp
            slot_c = slot_c + noise + 0.25 * amp
        elif topo == 'euclidean':
            noise = torch.randn_like(slot_c) * (0.5 * amp)
            slot_c = slot_c * (1.0 - 0.1 * amp) + noise
        else:
            colored = colored_noise_1f(slot_c.numel(), device, dtype, alpha=self.fractal_alpha)
            slot_c = slot_c + colored * amp

        # bound smoothly
        slot_c = self.curv_clamp * torch.tanh(slot_c / self.curv_clamp)

        # restore shape if caller used [mem_slots, D]
        if len(orig_shape) == 2:
            slot_c = slot_c.unsqueeze(1).expand(orig_shape)
        return slot_c

    @torch.no_grad()
    def mutate_tensor_like(self, tensor: torch.Tensor, subsystem: str) -> torch.Tensor:
        """Safe generic mutation for tensors like SPD factors."""
        fit = self._ema_fit.get(subsystem, None)
        fit = 0.5 if fit is None else float(fit)
        mode = self._mode_from_fitness(fit)
        step = self.mutation_rate
        if mode == "structured":
            out = tensor + 0.5 * step * torch.randn_like(tensor)
        elif mode == "stabilise":
            out = 0.98 * tensor
        elif mode == "explore":
            out = tensor + step * torch.randn_like(tensor)
        else:
            out = 0.97 * tensor + 0.25 * step * torch.randn_like(tensor)
        return out

    # ---------- mixture-of-metrics warp ----------
    def warp_and_blend(
        self,
        dist: torch.Tensor,              # [B, K] raw euclidean-ish distances
        indices: torch.Tensor,           # [B, K] slot indices
        memory_curvature: torch.Tensor,  # [mem_slots] or [mem_slots, D]
        subsystem: str,
        q_query: Optional[torch.Tensor] = None,     # [B, 4] unit-ish quaternions
        q_memory: Optional[torch.Tensor] = None,    # [mem_slots, 4]
        beta_override: Optional[torch.Tensor] = None # optional per-metric scales
    ) -> torch.Tensor:
        """
        Returns warped distances [B, K] combining Euclidean/Hyperbolic/Spherical/Spin
        according to a learned gate conditioned on simple live stats.
        """
        B, K = dist.shape
        device, dtype = dist.device, dist.dtype
        subsystem = self._resolve_subsystem(subsystem)

        # Gather per-slot curvature scalar -> [B, K]
        if memory_curvature.dim() == 2:
            slot_scalar = memory_curvature.mean(dim=1)
        else:
            slot_scalar = memory_curvature
        slot_c = slot_scalar.index_select(0, indices.view(-1)).view(B, K)

        # Live stats for the gate (detached scalars)
        with torch.no_grad():
            p = F.softmax(-dist, dim=-1)
            entropy = (-p * (p.clamp_min(1e-9)).log()).sum(dim=-1).mean()  # scalar
            mean_abs_curv = slot_c.abs().mean()
            spin_coh = torch.tensor(0.0, device=device, dtype=dtype)
            if (q_query is not None) and (q_memory is not None):
                q_query = qnormalize(q_query)
                q_mem_sel = qnormalize(q_memory.index_select(0, indices.view(-1))).view(B, K, 4)
                ang = qangle(q_query.unsqueeze(1).expand(B, K, 4), q_mem_sel)  # [B,K]
                # coherence ~ inverse angle
                spin_coh = (1.0 - (ang / math.pi)).mean()

            fitness = torch.tensor(
                self._ema_fit.get(subsystem, 0.5) if self._ema_fit.get(subsystem, None) is not None else 0.5,
                device=device, dtype=dtype
            )

        gate_in = torch.stack([
            fitness,                            # how well we’re doing overall
            entropy.to(dtype),                  # how peaked is current attention
            mean_abs_curv.to(dtype),            # how strong curvature is
            spin_coh.to(dtype)                  # spin coherence signal
        ], dim=-1).unsqueeze(0)                 # [1, 4]

        weights = self.gates[subsystem](gate_in).squeeze(0)     # [4], sums to 1
        # metric scales (learnable, per-subsystem)
        beta_vec = self.beta_table[subsystem] if beta_override is None else beta_override  # [4]

        # Individual warped terms (all [B, K])
        de = euc_warp(dist)
        dh = hyp_warp(dist, slot_c, alpha=beta_vec[1].item())
        ds = sph_warp(dist, gamma=beta_vec[2].item())

        dspin = 0.0
        if (q_query is not None) and (q_memory is not None):
            q_query = qnormalize(q_query)
            q_mem_sel = qnormalize(q_memory.index_select(0, indices.view(-1))).view(B, K, 4)
            ang = qangle(q_query.unsqueeze(1).expand(B, K, 4), q_mem_sel)  # [B,K]
            dspin = beta_vec[3].item() * (ang / math.pi)                   # normalize to [0,1]

        # Blend (weights are scalars here; broadcasting across [B,K] is fine)
        d_mix = (weights[0] * de +
                 weights[1] * dh +
                 weights[2] * ds +
                 weights[3] * (dspin if isinstance(dspin, torch.Tensor) else 0.0))

        # Final curvature-coupled micro-warp (like V2) for extra slot-local shaping
        beta_local = self.default_beta
        d_final = d_mix * (1.0 + beta_local * torch.tanh(slot_c) * d_mix)
        return d_final