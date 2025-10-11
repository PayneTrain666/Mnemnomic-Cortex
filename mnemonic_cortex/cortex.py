import torch
import torch.nn as nn
from .sensory_buffer import EnhancedSensoryBuffer
from .memory_curved import EnhancedCurvedMemory
from .triple_hybrid import EnhancedTripleHybridMemory
from .lightbulb import LightbulbDetector, ExplosiveRecallScaler

class EnhancedMnemonicCortex(nn.Module):
    """Top-level controller that routes inputs through buffer → WM → LTM with
    lightbulb-triggered 'explosive recall' (temperature modulation).
    """
    def __init__(self, input_dim: int, output_dim: int,
                 sensory_buffer_size: int = 5,
                 wm_slots: int = 7, wm_slot_dim: int = 256,
                 ltm_hg_slots: int = 2048, ltm_cgmn_slots: int = 1024, ltm_curved_slots: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.sensory_buffer = EnhancedSensoryBuffer(sensory_buffer_size, input_dim)
        self.working_memory = EnhancedCurvedMemory(input_dim, hidden_dim=wm_slot_dim, mem_slots=wm_slots)
        self.long_term_memory = EnhancedTripleHybridMemory(input_dim, output_dim,
                                                           hg_slots=ltm_hg_slots, cgmn_slots=ltm_cgmn_slots, curved_slots=ltm_curved_slots)

        # Context projection (kept simple: same dim by default)
        self.ctx_proj = nn.Linear(input_dim, input_dim)

        # Encoding and retrieval heads
        self.hippocampal_encoder = nn.Sequential(nn.Linear(input_dim*2, 512), nn.ReLU(), nn.Linear(512, 256))
        # Retrieval expects concatenation of a 256-d memory cue and the original context (input_dim)
        self.r_proj = nn.Linear(input_dim, 256)  # project memory readout to 256-d cue
        self.retrieval = nn.Sequential(nn.Linear(256 + input_dim, 512), nn.ReLU(), nn.Linear(512, input_dim))

        # Lightbulb + temperature scaler
        self.lightbulb = LightbulbDetector(input_dim, thresh=2.0)
        self.temp_scaler = ExplosiveRecallScaler(base_temp=1.0, min_temp=0.5, boost=0.3)

        # Importance predictor for consolidation condition
        self.importance_predictor = nn.Sequential(nn.Linear(input_dim,64), nn.ReLU(), nn.Linear(64,1), nn.Sigmoid())

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
            cue = torch.nn.functional.linear(cue, torch.empty(self.input_dim, 256, device=cue.device).normal_(std=0.02))

        if mtype == 'episodic':
            self.long_term_memory.hg(cue, operation='write')
        elif mtype == 'semantic':
            self.long_term_memory.cgmn(cue, operation='write')
        else:
            self.long_term_memory.curved(cue, operation='write')
        return idx

    def retrieve_memory(self, cue, context, strategy='associative'):
        B,S,d = cue.shape
        ctx = self._tile_context(context, S)
        # Combine cue with context
        c = (cue + ctx) * 0.5

        if strategy == 'direct':
            r = self.long_term_memory(c, operation='read')
        elif strategy == 'associative':
            r = self.long_term_memory.curved(c, operation='read')
        else:
            r = self.long_term_memory.hg(c, operation='read')
        # Project memory readout to 256-dimensional embedding to match retrieval head expectations
        cue_vec = self.r_proj(r.mean(dim=1))                 # (B,256)
        return self.retrieval(torch.cat([cue_vec, context], dim=-1))

    def consolidate_memories(self):
        # simple decay on curved memory importance; hook others similarly
        with torch.no_grad():
            self.long_term_memory.curved.memory_importance.mul_(0.999)

    def forward(self, sensory_input, context, operation='process'):
        # Lightbulb-driven temperature control
        fire = self.lightbulb(sensory_input)                # (B,)
        temp = self.temp_scaler(fire)                       # (B,) scalars
        self.working_memory.set_temperature(temp)
        self.long_term_memory.set_temperature(temp)

        if operation == 'process':
            filtered = self.process_sensory_input(sensory_input)          # (B,S,d)
            wm_out = self.working_memory(filtered, operation='read')      # (B,S,d)
            imp = self.importance_predictor(wm_out.mean(dim=1)).mean().item()
            if imp > 0.7:
                self.encode_memory(wm_out, context, mtype='episodic')
            return wm_out
        elif operation == 'retrieve':
            return self.retrieve_memory(sensory_input, context)
        else:
            self.consolidate_memories()
            return None
