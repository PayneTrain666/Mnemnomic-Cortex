from dataclasses import dataclass

@dataclass
class CortexConfig:
    # Core dims
    input_dim: int = 128
    output_dim: int = 128

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
