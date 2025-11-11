import torch
import torch.nn as nn
from .sensory_buffer import EnhancedSensoryBuffer
from .memory_curved import EnhancedCurvedMemory
from .triple_hybrid import EnhancedTripleHybridMemory
from .lightbulb import LightbulbDetector, ExplosiveRecallScaler

class EnhancedMnemonicCortex(nn.Module):
    """Top-level controller that routes inputs through buffer → WM → LTM with
    lightbulb-triggered 'explosive recall' (temperature modulation).
    Adds:
      • enable_energy_mode()
      • forgetting-style consolidation via consolidate_memories(threshold)
    """
    def __init__(self, input_dim: int, output_dim: int,
                 sensory_buffer_size: int = 5,
                 wm_slots: int = 7, wm_slot_dim: int = 256,
                 ltm_hg_slots: int = 2048, ltm_cgmn_slots: int = 1024, ltm_curved_slots: int = 512,
                 fusion: str = 'weighted'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.sensory_buffer = EnhancedSensoryBuffer(sensory_buffer_size, input_dim)
        self.working_memory = EnhancedCurvedMemory(input_dim, hidden_dim=wm_slot_dim, mem_slots=wm_slots)
        self.long_term_memory = EnhancedTripleHybridMemory(input_dim, output_dim,
                                                           hg_slots=ltm_hg_slots, cgmn_slots=ltm_cgmn_slots, curved_slots=ltm_curved_slots,
                                                           fusion=fusion)

        # Context projection (kept simple: same dim by default)
        self.ctx_proj = nn.Linear(input_dim, input_dim)

        # Encoding and retrieval heads
        self.hippocampal_encoder = nn.Sequential(nn.Linear(input_dim*2, 512), nn.ReLU(), nn.Linear(512, 256))
        self.r_proj = nn.Linear(input_dim, 256)
        self.cue_to_input = nn.Linear(256, input_dim)
        self.retrieval = nn.Sequential(nn.Linear(256 + input_dim, 512), nn.ReLU(), nn.Linear(512, input_dim))

        # Lightbulb + temperature scaler
        self.lightbulb = LightbulbDetector(input_dim, thresh=2.0)
        self.temp_scaler = ExplosiveRecallScaler(base_temp=1.0, min_temp=0.5, boost=0.3)

        # Importance predictor for consolidation condition
        self.importance_predictor = nn.Sequential(nn.Linear(input_dim,64), nn.ReLU(), nn.Linear(64,1), nn.Sigmoid())

        # Forgetting threshold for consolidation
        self.forgetting_threshold = 0.3
        self.energy_mode = False

        # Contrastive recall head (InfoNCE)
        self.contrastive_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        self.recall_temp = 0.07  # temperature for InfoNCE

        # Learned write gate with STE
        self.write_gate = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    # ---------------- Helpers ----------------
    def enable_energy_mode(self, enable: bool = True):
        self.energy_mode = enable
        self.long_term_memory.enable_energy_efficient_mode(enable)
        self.working_memory.enable_energy_efficient_mode(enable)

    def _ste_write_gate(self, x):
        """Compute write gate with straight-through estimator.
        x: (B,S,d) -> gate_prob: (B,1), gate_hard: (B,1)
        """
        pooled = x.mean(dim=1)  # (B,d)
        gate_prob = self.write_gate(pooled)  # (B,1) continuous in [0,1]
        
        if self.training:
            # Sample binary decision
            gate_hard = (torch.rand_like(gate_prob) < gate_prob).float()
            # STE: forward uses hard, backward uses soft
            gate_ste = gate_hard - gate_prob.detach() + gate_prob
        else:
            # At inference, threshold at 0.5
            gate_ste = (gate_prob > 0.5).float()
        
        return gate_ste, gate_prob

    def _tile_context(self, context, S):
        # context: (B,c) -> project to (B,S,input_dim)
        cproj = self.ctx_proj(context)                    # (B,input_dim)
        return cproj.unsqueeze(1).expand(-1, S, -1)

    def process_sensory_input(self, sensory_input):
        self.sensory_buffer.update(sensory_input)
        return self.sensory_buffer.attention_filter(sensory_input)

    def encode_memory(self, info, context, mtype):
        B,S,d = info.shape
        ctx = self._tile_context(context, S)
        pooled = torch.cat([info, ctx], dim=-1).mean(dim=1)      # (B, 2d)
        idx = self.hippocampal_encoder(pooled)                   # (B,256)

        cue = idx.unsqueeze(1).expand(-1, S, -1)                 # (B,S,256)
        # Project cue to input_dim if needed
        if cue.size(-1) != self.input_dim:
            cue = self.cue_to_input(cue)

        if mtype == 'episodic':
            self.long_term_memory.hg(cue, operation='write')
        elif mtype == 'semantic':
            self.long_term_memory.cgmn(cue, operation='write')
        else:
            self.long_term_memory.curved(cue, operation='write')
        return idx

    def retrieve_memory(self, cue, context, strategy='associative', fire_mask=None, recall_boost: float = 0.3):
        """Retrieve memories with optional explosive recall (fire_mask)."""
        B,S,d = cue.shape
        ctx = self._tile_context(context, S)
        c = (cue + ctx) * 0.5

        if strategy == 'direct':
            r = self.long_term_memory(c, operation='read', fire_mask=fire_mask, recall_boost=recall_boost)
        elif strategy == 'associative':
            # Use curved memory path (kept simple; doesn't use fire)
            r = self.long_term_memory.curved(c, operation='read')
        else:
            # Reconstructive via HG
            r = self.long_term_memory.hg(c, operation='read', fire_mask=fire_mask, recall_boost=recall_boost)

        cue_vec = self.r_proj(r.mean(dim=1))
        return self.retrieval(torch.cat([cue_vec, context], dim=-1))

    @torch.no_grad()
    def consolidate_memories(self, threshold: float = None):
        """Forget rarely used slots (usage-based) and gently decay working-memory importance."""
        th = self.forgetting_threshold if threshold is None else float(threshold)
        self.long_term_memory.consolidate_unused(th)
        self.working_memory.memory_importance.mul_(0.999)

    def get_metrics(self):
        """Collect diagnostic metrics from all components."""
        metrics = {}
        # Lightbulb metrics
        metrics.update(self.lightbulb.get_metrics())
        # Memory module metrics
        metrics.update(self.long_term_memory.hg.get_metrics())
        metrics.update(self.long_term_memory.cgmn.get_metrics())
        metrics.update(self.long_term_memory.curved.get_metrics())
        metrics.update(self.working_memory.get_metrics())
        # Cortex-level
        metrics['energy_mode'] = self.energy_mode
        metrics['forgetting_threshold'] = self.forgetting_threshold
        return metrics

    def save_checkpoint(self, path: str, version: str = '1.0'):
        """Save versioned checkpoint with explicit schema."""
        import torch
        checkpoint = {
            'version': version,
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'forgetting_threshold': self.forgetting_threshold,
                'energy_mode': self.energy_mode,
            },
            'lightbulb_state': {
                'trigger_rate_ema': self.lightbulb.trigger_rate_ema.item(),
                'threshold': self.lightbulb.thresh,
                'total_fires': self.lightbulb.total_fires.item(),
                'total_samples': self.lightbulb.total_samples.item(),
            },
            'memory_state': {
                'hg_usage': self.long_term_memory.hg.usage_counts.clone(),
                'cgmn_usage': self.long_term_memory.cgmn.usage_counts.clone(),
                'curved_usage': self.long_term_memory.curved.usage_counts.clone(),
            }
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, strict: bool = True):
        """Load checkpoint with version validation."""
        import torch
        checkpoint = torch.load(path, map_location='cpu')
        
        # Version check
        version = checkpoint.get('version', 'unknown')
        if version != '1.0' and strict:
            raise ValueError(f"Checkpoint version {version} does not match expected '1.0'")
        elif version != '1.0':
            print(f"Warning: Loading checkpoint version {version}, expected '1.0'. Compatibility not guaranteed.")
        
        # Load state dict
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # Restore config
        if 'config' in checkpoint:
            self.forgetting_threshold = checkpoint['config'].get('forgetting_threshold', self.forgetting_threshold)
            self.energy_mode = checkpoint['config'].get('energy_mode', self.energy_mode)
        
        # Restore lightbulb state
        if 'lightbulb_state' in checkpoint:
            lb = checkpoint['lightbulb_state']
            self.lightbulb.trigger_rate_ema.fill_(lb.get('trigger_rate_ema', 0.0))
            self.lightbulb.thresh = lb.get('threshold', 2.0)
            self.lightbulb.total_fires.fill_(lb.get('total_fires', 0))
            self.lightbulb.total_samples.fill_(lb.get('total_samples', 0))
        
        # Restore memory usage
        if 'memory_state' in checkpoint:
            mem = checkpoint['memory_state']
            if 'hg_usage' in mem:
                self.long_term_memory.hg.usage_counts.copy_(mem['hg_usage'])
            if 'cgmn_usage' in mem:
                self.long_term_memory.cgmn.usage_counts.copy_(mem['cgmn_usage'])
            if 'curved_usage' in mem:
                self.long_term_memory.curved.usage_counts.copy_(mem['curved_usage'])

    def compute_recall_loss(self, cue, context):
        """InfoNCE contrastive loss: cue vs. retrieved memory.
        cue: (B,S,d), context: (B,d)
        Returns scalar loss.
        """
        B, S, d = cue.shape
        # Retrieve from LTM
        retrieved = self.retrieve_memory(cue, context, strategy='direct')  # (B,d)
        
        # Project both to contrastive space
        cue_pooled = cue.mean(dim=1)  # (B,d)
        z_cue = torch.nn.functional.normalize(self.contrastive_proj(cue_pooled), dim=-1)  # (B,128)
        z_ret = torch.nn.functional.normalize(self.contrastive_proj(retrieved), dim=-1)   # (B,128)
        
        # InfoNCE: positive = same batch index, negatives = others
        logits = torch.matmul(z_cue, z_ret.T) / self.recall_temp  # (B,B)
        labels = torch.arange(B, device=logits.device)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss

    # ---------------- Forward ----------------
    def forward(self, sensory_input, context, operation='process', return_aux_losses=False):
        fire = self.lightbulb(sensory_input)                # (B,)
        temp = self.temp_scaler(fire)                       # (B,) scalars
        self.working_memory.set_temperature(temp)
        self.long_term_memory.set_temperature(temp)

        if operation == 'process':
            filtered = self.process_sensory_input(sensory_input)          # (B,S,d)

            # --- Working-memory write phase ----------------------------------
            # Store filtered sensory input into WM slots with importance gating.
            imp = self.importance_predictor(filtered.mean(dim=1))         # (B,1)
            self.working_memory(filtered, operation='write', importance=imp)

            # --- Working-memory read phase -----------------------------------
            wm_out = self.working_memory(filtered, operation='read')      # (B,S,d)

            # --- Consolidation into long-term memory ------------------------
            if self.training:  # consolidate only during training
                # Learned write gate with STE
                gate, gate_prob = self._ste_write_gate(filtered)  # (B,1)
                scaled = wm_out * imp.unsqueeze(-1)  # (B,S,d)
                
                # Only encode if gate=1 (batched conditional write)
                if gate.sum() > 0:  # at least one sample wants to write
                    # Mask the batch
                    write_mask = gate.squeeze(-1) > 0.5  # (B,)
                    if write_mask.any():
                        self.encode_memory(scaled[write_mask], context[write_mask], mtype='episodic')
            
            # --- Compute auxiliary losses if requested ----------------------
            if return_aux_losses and self.training:
                recall_loss = self.compute_recall_loss(filtered, context)
                gate, gate_prob = self._ste_write_gate(filtered)
                return wm_out, {'recall_loss': recall_loss, 'write_gate_prob': gate_prob.mean()}
            
            return wm_out

        elif operation == 'retrieve':
            # Route fire to LTM for sharper readout on the cue as well
            fire_ret = self.lightbulb(sensory_input)  # (B,)
            return self.retrieve_memory(sensory_input, context, strategy='direct', fire_mask=fire_ret, recall_boost=0.3)

        else:
            self.consolidate_memories()
            return None
