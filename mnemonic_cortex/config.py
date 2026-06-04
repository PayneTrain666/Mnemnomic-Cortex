from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class CortexConfig:
    # Core dims
    input_dim: int = 160
    output_dim: int = 160

    # HG memory
    hg_manifold_dim: int = 24
    hg_mem_slots: int = 1024
    hg_qubits: int = 8
    hg_topk: int = 32
    hg_fractal_scales: int = 4
    hg_holo_dim: int = 256

    # CGMN memory
    cgmn_manifold_dim: int = 16
    cgmn_mem_slots: int = 512
    cgmn_slot_dim: int = 256
    cgmn_topk: int = 32

    # Curved memory
    curved_hidden_dim: int = 256
    curved_mem_slots: int = 128
    curved_topk: int = 16

    # Working memory
    wm_slots: int = 7
    wm_slot_dim: int = 256

    # Sensory buffer
    sensory_buffer_size: int = 5

    # Misc
    seed: int = 42

    # Consolidated lexicon (optional)
    cms_vocab_size: int = 0
    cms_senses: int = 3

    def to_cortex_kwargs(self) -> Dict[str, Any]:
        return {
            "input_dim": int(self.input_dim),
            "output_dim": int(self.output_dim),
            "sensory_buffer_size": int(self.sensory_buffer_size),
            "wm_slots": int(self.wm_slots),
            "wm_slot_dim": int(self.wm_slot_dim),
            "ltm_hg_slots": int(self.hg_mem_slots),
            "ltm_cgmn_slots": int(self.cgmn_mem_slots),
            "ltm_curved_slots": int(self.curved_mem_slots),
            "cms_vocab_size": int(self.cms_vocab_size),
            "cms_senses": int(self.cms_senses),
        }
