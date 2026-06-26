from dataclasses import dataclass

from .capacity_profile import CapacityProfile


@dataclass
class CortexConfig:
    # Canonical sizing profile
    capacity_profile: str = "standard"

    # Core dims
    input_dim: int = 160
    output_dim: int = 160

    # HG memory
    hg_manifold_dim: int = 24
    hg_mem_slots: int = 1028
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
    wm_slots: int = 8
    wm_slot_dim: int = 256
    wm_transformer_layers: int = 2

    # Sensory/context budget
    sensory_buffer_size: int = 8
    max_external_context_tokens: int = 64
    global_hidden_max_layers: int = 128
    max_parameter_tokens: int = 48

    # Optional consolidated parameter storage loop stack
    enable_parameter_storage_loop_stack: bool = False
    parameter_loop_slots_per_layer: int = 64
    parameter_loop_free_hidden_layers: int = 4
    enable_parameter_loop_ltm_context: bool = True
    enable_parameter_loop_training_writes: bool = False
    parameter_loop_training_write_scale: float = 1.0

    # Working-memory fabric
    working_memory_fabric: str = "legacy"
    qdt_hardware_profile: str = "single_gpu_8_12gb"
    qdt_num_slots: int = 0
    qdt_transformer_layers: int = 0
    qdt_qspin_guarded_shadow: bool = True

    # Transformer depth policy
    depth_profile: str = "standard"

    # Misc
    seed: int = 42
    fusion: str = "weighted"
    hgm_enabled: bool = False

    # Consolidated lexicon (optional)
    cms_vocab_size: int = 0
    cms_senses: int = 3

    def __post_init__(self) -> None:
        profile = CapacityProfile.from_name(self.capacity_profile)
        if int(self.input_dim) <= 0:
            self.input_dim = int(profile.input_dim)
        if int(self.output_dim) <= 0:
            self.output_dim = int(profile.output_dim)
        self.wm_slots = int(max(1, self.wm_slots if self.wm_slots > 0 else profile.wm_slots))
        self.wm_slot_dim = int(max(8, self.wm_slot_dim if self.wm_slot_dim > 0 else profile.wm_slot_dim))
        self.wm_transformer_layers = int(max(0, self.wm_transformer_layers))
        self.parameter_loop_slots_per_layer = int(max(1, self.parameter_loop_slots_per_layer))
        self.parameter_loop_free_hidden_layers = int(max(0, self.parameter_loop_free_hidden_layers))
        self.parameter_loop_training_write_scale = float(max(0.0, self.parameter_loop_training_write_scale))
        self.working_memory_fabric = str(self.working_memory_fabric).strip().lower()
        if self.working_memory_fabric not in {"legacy", "qdt"}:
            raise ValueError("working_memory_fabric must be 'legacy' or 'qdt'")
        self.qdt_num_slots = int(max(0, self.qdt_num_slots))
        self.qdt_transformer_layers = int(max(0, self.qdt_transformer_layers))

    def to_cortex_kwargs(self) -> dict:
        return {
            "input_dim": int(self.input_dim),
            "output_dim": int(self.output_dim),
            "sensory_buffer_size": int(self.sensory_buffer_size),
            "wm_slots": int(self.wm_slots),
            "wm_slot_dim": int(self.wm_slot_dim),
            "wm_transformer_layers": int(self.wm_transformer_layers),
            "ltm_hg_dim": int(self.hg_manifold_dim),
            "ltm_hg_slots": int(self.hg_mem_slots),
            "ltm_hg_qubits": int(self.hg_qubits),
            "ltm_cgmn_dim": int(self.cgmn_manifold_dim),
            "ltm_cgmn_slots": int(self.cgmn_mem_slots),
            "ltm_cgmn_slot_dim": int(self.cgmn_slot_dim),
            "ltm_curved_hidden": int(self.curved_hidden_dim),
            "ltm_curved_slots": int(self.curved_mem_slots),
            "ltm_n_transformer_layers": 3 if self.depth_profile == "standard" else (2 if self.depth_profile == "compact" else 5),
            "ltm_fusion_transformer_layers": 2 if self.depth_profile == "compact" else 4,
            "ltm_cross_model_attention_layers": 1 if self.depth_profile == "compact" else 4,
            "ltm_prefusion_specialization_layers": 1 if self.depth_profile == "compact" else 2,
            "global_hidden_max_layers": int(self.global_hidden_max_layers),
            "max_parameter_tokens": int(self.max_parameter_tokens),
            "max_external_context_tokens": int(self.max_external_context_tokens),
            "enable_parameter_storage_loop_stack": bool(self.enable_parameter_storage_loop_stack),
            "parameter_loop_slots_per_layer": int(self.parameter_loop_slots_per_layer),
            "parameter_loop_free_hidden_layers": int(self.parameter_loop_free_hidden_layers),
            "enable_parameter_loop_ltm_context": bool(self.enable_parameter_loop_ltm_context),
            "enable_parameter_loop_training_writes": bool(self.enable_parameter_loop_training_writes),
            "parameter_loop_training_write_scale": float(self.parameter_loop_training_write_scale),
            "working_memory_fabric": str(self.working_memory_fabric),
            "qdt_hardware_profile": str(self.qdt_hardware_profile),
            "qdt_num_slots": int(self.qdt_num_slots),
            "qdt_transformer_layers": int(self.qdt_transformer_layers),
            "qdt_qspin_guarded_shadow": bool(self.qdt_qspin_guarded_shadow),
            "fusion": str(self.fusion),
            "hgm_enabled": bool(self.hgm_enabled),
            "cms_vocab_size": int(self.cms_vocab_size),
            "cms_senses": int(self.cms_senses),
            "ltm_depth_profile": str(self.depth_profile),
        }
