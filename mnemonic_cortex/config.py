from dataclasses import dataclass

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
    fusion: str = "weighted"
    hgm_enabled: bool = False

    # Consolidated lexicon (optional)
    cms_vocab_size: int = 0
    cms_senses: int = 3

    def to_cortex_kwargs(self) -> dict:
        return {
            "input_dim": int(self.input_dim),
            "output_dim": int(self.output_dim),
            "wm_slots": int(self.wm_slots),
            "wm_slot_dim": int(self.wm_slot_dim),
            "fusion": str(self.fusion),
            "hgm_enabled": bool(self.hgm_enabled),
            "cms_vocab_size": int(self.cms_vocab_size),
            "cms_senses": int(self.cms_senses),
        }
