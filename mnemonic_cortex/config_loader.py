"""
Plain-language summary
----------------------
What this file is for: Reads YAML / unified config files into live settings objects.
How it fits in the system: Bridges human-editable configs to cortex construction.
Status: ACTIVE
Important notes for non-coders: Prefer config files + this loader over hard-coding sizes elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import torch

from .ahg import AHGConfig
from .capacity_profile import CapacityProfile
from .consolidation_broker import StoreConfig

if TYPE_CHECKING:
    from .cortex import EnhancedMnemonicCortex


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _as_str_list(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return (str(value),)


@dataclass
class DistillConfig:
    enabled: bool = False
    embedding_weight: float = 1.0
    mse_weight: float = 0.0
    neighbor_kl_weight: float = 0.0
    cms_teacher_weight: float = 0.0
    teacher_domain: str = "core"
    student_domains: Tuple[str, ...] = ()
    neighbor_k: int = 16
    sim_temp: float = 0.07


@dataclass
class CortexBuildConfig:
    capacity_profile: str = "standard"
    apply_capacity_profile: bool = False
    input_dim: int = 160
    output_dim: int = 160
    sensory_buffer_size: int = 8
    wm_slots: int = 8
    wm_slot_dim: int = 256
    wm_transformer_layers: int = 2
    fusion: str = "weighted"
    hgm_enabled: bool = False
    ltm_hg_slots: int = 1028
    ltm_cgmn_slots: int = 512
    ltm_curved_slots: int = 128
    ltm_spatial_slots: int = 256
    ltm_n_transformer_layers: int = 3
    ltm_depth_profile: str = "standard"
    max_external_context_tokens: int = 64
    global_hidden_max_layers: int = 128
    max_parameter_tokens: int = 48
    enable_parameter_storage_loop_stack: bool = False
    parameter_loop_slots_per_layer: int = 64
    parameter_loop_free_hidden_layers: int = 4
    enable_parameter_loop_ltm_context: bool = True
    enable_parameter_loop_training_writes: bool = False
    parameter_loop_training_write_scale: float = 1.0
    enable_parameter_loop_auto_consolidation: bool = False
    parameter_loop_consolidation_interval: int = 100
    parameter_loop_consolidation_max_bundles: int = 4
    parameter_loop_consolidation_min_params: int = 1
    parameter_loop_consolidation_min_total_numel: int = 1024
    parameter_loop_consolidation_include: Tuple[str, ...] = ()
    parameter_loop_consolidation_exclude: Tuple[str, ...] = (
        "parameter_storage_loop_stack",
        "trainable_parameter_cps",
    )
    enable_trainable_parameter_cps: bool = False
    trainable_parameter_cps_include: Tuple[str, ...] = ()
    trainable_parameter_cps_exclude: Tuple[str, ...] = ("qspin", "trainable_parameter_cps")
    trainable_parameter_cps_enable_compression: bool = False
    trainable_parameter_cps_max_rank: int = 8
    trainable_parameter_cps_min_cohort_size: int = 2
    trainable_parameter_cps_reconstruction_tolerance: float = 1e-4
    trainable_parameter_cps_output_tolerance: float = 1e-5
    working_memory_fabric: str = "legacy"
    qdt_hardware_profile: str = "single_gpu_8_12gb"
    qdt_num_slots: int = 0
    qdt_transformer_layers: int = 0
    qdt_qspin_guarded_shadow: bool = True
    qdt_qspin_live_activation: Optional[bool] = False
    qdt_qspin_live_kill_switch_enabled: bool = True
    qdt_qspin_live_max_payload_tokens: int = 8

    def __post_init__(self) -> None:
        profile = CapacityProfile.from_name(self.capacity_profile)
        self.capacity_profile = profile.name
        if not bool(self.apply_capacity_profile):
            return
        self.input_dim = int(profile.input_dim)
        self.output_dim = int(profile.output_dim)
        self.sensory_buffer_size = int(profile.sensory_buffer_size)
        self.wm_slots = int(profile.wm_slots)
        self.wm_slot_dim = int(profile.wm_slot_dim)
        self.ltm_hg_slots = int(profile.hg_slots)
        self.ltm_cgmn_slots = int(profile.cgmn_slots)
        self.ltm_curved_slots = int(profile.curved_slots)
        self.ltm_spatial_slots = int(profile.spatial_slots)
        self.max_external_context_tokens = int(profile.max_external_context_tokens)
        self.global_hidden_max_layers = int(profile.global_hidden_max_layers)
        self.max_parameter_tokens = int(profile.max_parameter_tokens)

    def to_cortex_kwargs(self) -> Dict[str, Any]:
        return {
            "input_dim": int(self.input_dim),
            "output_dim": int(self.output_dim),
            "sensory_buffer_size": int(self.sensory_buffer_size),
            "wm_slots": int(self.wm_slots),
            "wm_slot_dim": int(self.wm_slot_dim),
            "wm_transformer_layers": int(self.wm_transformer_layers),
            "fusion": str(self.fusion),
            "hgm_enabled": bool(self.hgm_enabled),
            "ltm_hg_slots": int(self.ltm_hg_slots),
            "ltm_cgmn_slots": int(self.ltm_cgmn_slots),
            "ltm_curved_slots": int(self.ltm_curved_slots),
            "ltm_spatial_slots": int(self.ltm_spatial_slots),
            "ltm_n_transformer_layers": int(self.ltm_n_transformer_layers),
            "ltm_depth_profile": str(self.ltm_depth_profile),
            "max_external_context_tokens": int(self.max_external_context_tokens),
            "global_hidden_max_layers": int(self.global_hidden_max_layers),
            "max_parameter_tokens": int(self.max_parameter_tokens),
            "enable_parameter_storage_loop_stack": bool(self.enable_parameter_storage_loop_stack),
            "parameter_loop_slots_per_layer": int(self.parameter_loop_slots_per_layer),
            "parameter_loop_free_hidden_layers": int(self.parameter_loop_free_hidden_layers),
            "enable_parameter_loop_ltm_context": bool(self.enable_parameter_loop_ltm_context),
            "enable_parameter_loop_training_writes": bool(self.enable_parameter_loop_training_writes),
            "parameter_loop_training_write_scale": float(self.parameter_loop_training_write_scale),
            "enable_parameter_loop_auto_consolidation": bool(self.enable_parameter_loop_auto_consolidation),
            "parameter_loop_consolidation_interval": int(self.parameter_loop_consolidation_interval),
            "parameter_loop_consolidation_max_bundles": int(self.parameter_loop_consolidation_max_bundles),
            "parameter_loop_consolidation_min_params": int(self.parameter_loop_consolidation_min_params),
            "parameter_loop_consolidation_min_total_numel": int(self.parameter_loop_consolidation_min_total_numel),
            "parameter_loop_consolidation_include": tuple(self.parameter_loop_consolidation_include),
            "parameter_loop_consolidation_exclude": tuple(self.parameter_loop_consolidation_exclude),
            "enable_trainable_parameter_cps": bool(self.enable_trainable_parameter_cps),
            "trainable_parameter_cps_include": tuple(self.trainable_parameter_cps_include),
            "trainable_parameter_cps_exclude": tuple(self.trainable_parameter_cps_exclude),
            "trainable_parameter_cps_enable_compression": bool(
                self.trainable_parameter_cps_enable_compression
            ),
            "trainable_parameter_cps_max_rank": int(self.trainable_parameter_cps_max_rank),
            "trainable_parameter_cps_min_cohort_size": int(
                self.trainable_parameter_cps_min_cohort_size
            ),
            "trainable_parameter_cps_reconstruction_tolerance": float(
                self.trainable_parameter_cps_reconstruction_tolerance
            ),
            "trainable_parameter_cps_output_tolerance": float(
                self.trainable_parameter_cps_output_tolerance
            ),
            "working_memory_fabric": str(self.working_memory_fabric),
            "qdt_hardware_profile": str(self.qdt_hardware_profile),
            "qdt_num_slots": int(self.qdt_num_slots),
            "qdt_transformer_layers": int(self.qdt_transformer_layers),
            "qdt_qspin_guarded_shadow": bool(self.qdt_qspin_guarded_shadow),
            "qdt_qspin_live_activation": self.qdt_qspin_live_activation,
            "qdt_qspin_live_kill_switch_enabled": bool(self.qdt_qspin_live_kill_switch_enabled),
            "qdt_qspin_live_max_payload_tokens": int(self.qdt_qspin_live_max_payload_tokens),
        }


@dataclass
class FeatureConfig:
    hgm_enabled: bool = False
    reasoning_bridge_enabled: bool = False


@dataclass
class GlobalConfig:
    stores: Dict[str, StoreConfig]
    ahg: AHGConfig
    distill: DistillConfig = field(default_factory=DistillConfig)


@dataclass
class UnifiedYamlConfig:
    cortex: CortexBuildConfig
    features: FeatureConfig
    global_config: GlobalConfig


def _load_yaml_obj(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to load YAML config files.") from exc

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_global_config(obj: Dict[str, Any]) -> GlobalConfig:
    stores: Dict[str, StoreConfig] = {}
    for name, cfg in (obj.get("stores") or {}).items():
        geo = cfg.get("geometries", {})
        stores[name] = StoreConfig(
            senses=int(cfg.get("senses", 3)),
            conformal_b=float(cfg.get("conformal_b", 0.02)),
            w_h=float(geo.get("hyperbolic", 0.0)),
            w_p=float(geo.get("cproj", geo.get("phase", 0.0))),
            w_e=float(geo.get("fisher", geo.get("euclidean", 0.0))),
        )

    ahg_obj = obj.get("ahg", {}) or {}
    ahg = AHGConfig(
        proto_tau=float(ahg_obj.get("proto_tau", 0.45)),
        fisher_tau=float(ahg_obj.get("fisher_tau", 0.60)),
        phase_rho=float(ahg_obj.get("phase_rho", 0.10)),
        agree_tau=float(ahg_obj.get("agree_tau", 0.15)),
        ask_on_uncertain=_as_bool(ahg_obj.get("ask_on_uncertain", True), True),
        refuse_on_high_risk=_as_bool(ahg_obj.get("refuse_on_high_risk", True), True),
    )

    distill_obj = obj.get("distill", {}) or {}
    distill = DistillConfig(
        enabled=_as_bool(distill_obj.get("enabled", False), False),
        embedding_weight=float(distill_obj.get("embedding_weight", 1.0)),
        mse_weight=float(distill_obj.get("mse_weight", 0.0)),
        neighbor_kl_weight=float(distill_obj.get("neighbor_kl_weight", 0.0)),
        cms_teacher_weight=float(distill_obj.get("cms_teacher_weight", 0.0)),
        teacher_domain=str(distill_obj.get("teacher_domain", "core")),
        student_domains=_as_str_list(distill_obj.get("student_domains")),
        neighbor_k=int(distill_obj.get("neighbor_k", 16)),
        sim_temp=float(distill_obj.get("sim_temp", 0.07)),
    )
    return GlobalConfig(stores=stores, ahg=ahg, distill=distill)


def load_yaml_config(path: str) -> GlobalConfig:
    return _parse_global_config(_load_yaml_obj(path))


def load_unified_yaml_config(path: str) -> UnifiedYamlConfig:
    obj = _load_yaml_obj(path)
    cortex_obj = obj.get("cortex", {}) or {}
    features_obj = obj.get("features", {}) or {}
    global_config = _parse_global_config(obj)

    profile_name = str(cortex_obj.get("capacity_profile", "standard"))
    profile = CapacityProfile.from_name(profile_name)
    cortex_cfg = CortexBuildConfig(
        capacity_profile=profile.name,
        apply_capacity_profile=_as_bool(
            cortex_obj.get("apply_capacity_profile", False), False
        ),
        input_dim=int(cortex_obj.get("input_dim", profile.input_dim)),
        output_dim=int(cortex_obj.get("output_dim", cortex_obj.get("input_dim", profile.output_dim))),
        sensory_buffer_size=int(cortex_obj.get("sensory_buffer_size", profile.sensory_buffer_size)),
        wm_slots=int(cortex_obj.get("wm_slots", profile.wm_slots)),
        wm_slot_dim=int(cortex_obj.get("wm_slot_dim", profile.wm_slot_dim)),
        wm_transformer_layers=int(cortex_obj.get("wm_transformer_layers", 2)),
        fusion=str(cortex_obj.get("fusion", "weighted")),
        hgm_enabled=_as_bool(cortex_obj.get("hgm_enabled", False), False),
        ltm_hg_slots=int(cortex_obj.get("ltm_hg_slots", profile.hg_slots)),
        ltm_cgmn_slots=int(cortex_obj.get("ltm_cgmn_slots", profile.cgmn_slots)),
        ltm_curved_slots=int(cortex_obj.get("ltm_curved_slots", profile.curved_slots)),
        ltm_spatial_slots=int(cortex_obj.get("ltm_spatial_slots", profile.spatial_slots)),
        ltm_n_transformer_layers=int(cortex_obj.get("ltm_n_transformer_layers", 3)),
        ltm_depth_profile=str(cortex_obj.get("ltm_depth_profile", "standard")),
        max_external_context_tokens=int(cortex_obj.get("max_external_context_tokens", profile.max_external_context_tokens)),
        global_hidden_max_layers=int(cortex_obj.get("global_hidden_max_layers", profile.global_hidden_max_layers)),
        max_parameter_tokens=int(cortex_obj.get("max_parameter_tokens", profile.max_parameter_tokens)),
        enable_parameter_storage_loop_stack=_as_bool(cortex_obj.get("enable_parameter_storage_loop_stack", False), False),
        parameter_loop_slots_per_layer=int(cortex_obj.get("parameter_loop_slots_per_layer", 64)),
        parameter_loop_free_hidden_layers=int(cortex_obj.get("parameter_loop_free_hidden_layers", 4)),
        enable_parameter_loop_ltm_context=_as_bool(cortex_obj.get("enable_parameter_loop_ltm_context", True), True),
        enable_parameter_loop_training_writes=_as_bool(cortex_obj.get("enable_parameter_loop_training_writes", False), False),
        parameter_loop_training_write_scale=float(cortex_obj.get("parameter_loop_training_write_scale", 1.0)),
        enable_parameter_loop_auto_consolidation=_as_bool(cortex_obj.get("enable_parameter_loop_auto_consolidation", False), False),
        parameter_loop_consolidation_interval=int(cortex_obj.get("parameter_loop_consolidation_interval", 100)),
        parameter_loop_consolidation_max_bundles=int(cortex_obj.get("parameter_loop_consolidation_max_bundles", 4)),
        parameter_loop_consolidation_min_params=int(cortex_obj.get("parameter_loop_consolidation_min_params", 1)),
        parameter_loop_consolidation_min_total_numel=int(cortex_obj.get("parameter_loop_consolidation_min_total_numel", 1024)),
        parameter_loop_consolidation_include=_as_str_list(cortex_obj.get("parameter_loop_consolidation_include")),
        parameter_loop_consolidation_exclude=_as_str_list(
            cortex_obj.get(
                "parameter_loop_consolidation_exclude",
                ["parameter_storage_loop_stack", "trainable_parameter_cps"],
            )
        ),
        enable_trainable_parameter_cps=_as_bool(
            cortex_obj.get("enable_trainable_parameter_cps", False), False
        ),
        trainable_parameter_cps_include=_as_str_list(
            cortex_obj.get("trainable_parameter_cps_include")
        ),
        trainable_parameter_cps_exclude=_as_str_list(
            cortex_obj.get(
                "trainable_parameter_cps_exclude",
                ["qspin", "trainable_parameter_cps"],
            )
        ),
        trainable_parameter_cps_enable_compression=_as_bool(
            cortex_obj.get("trainable_parameter_cps_enable_compression", False), False
        ),
        trainable_parameter_cps_max_rank=int(
            cortex_obj.get("trainable_parameter_cps_max_rank", 8)
        ),
        trainable_parameter_cps_min_cohort_size=int(
            cortex_obj.get("trainable_parameter_cps_min_cohort_size", 2)
        ),
        trainable_parameter_cps_reconstruction_tolerance=float(
            cortex_obj.get("trainable_parameter_cps_reconstruction_tolerance", 1e-4)
        ),
        trainable_parameter_cps_output_tolerance=float(
            cortex_obj.get("trainable_parameter_cps_output_tolerance", 1e-5)
        ),
        working_memory_fabric=str(cortex_obj.get("working_memory_fabric", "legacy")),
        qdt_hardware_profile=str(cortex_obj.get("qdt_hardware_profile", "single_gpu_8_12gb")),
        qdt_num_slots=int(cortex_obj.get("qdt_num_slots", 0)),
        qdt_transformer_layers=int(cortex_obj.get("qdt_transformer_layers", 0)),
        qdt_qspin_guarded_shadow=_as_bool(cortex_obj.get("qdt_qspin_guarded_shadow", True), True),
        qdt_qspin_live_activation=(
            _as_bool(cortex_obj.get("qdt_qspin_live_activation"), False)
            if "qdt_qspin_live_activation" in cortex_obj
            else False
        ),
        qdt_qspin_live_kill_switch_enabled=_as_bool(cortex_obj.get("qdt_qspin_live_kill_switch_enabled", True), True),
        qdt_qspin_live_max_payload_tokens=int(cortex_obj.get("qdt_qspin_live_max_payload_tokens", 8)),
    )
    features_cfg = FeatureConfig(
        hgm_enabled=_as_bool(features_obj.get("hgm_enabled", False), False),
        reasoning_bridge_enabled=_as_bool(features_obj.get("reasoning_bridge_enabled", False), False),
    )

    # Features may promote constructor defaults for compatibility.
    if features_cfg.hgm_enabled:
        cortex_cfg.hgm_enabled = True

    return UnifiedYamlConfig(cortex=cortex_cfg, features=features_cfg, global_config=global_config)


def apply_config_to_broker(broker, gcfg: GlobalConfig):
    if broker is None or gcfg is None:
        return
    for name, sc in gcfg.stores.items():
        if name in broker.stores:
            broker.store_configs[name] = sc
            store = broker.stores[name]
            with torch.no_grad():
                store.w.copy_(
                    torch.tensor(
                        [sc.w_h, sc.w_p, sc.w_e],
                        device=store.w.device,
                        dtype=store.w.dtype,
                    )
                )


def _apply_unified_features(model: Any, unified: UnifiedYamlConfig, *, config_path: str, broker_vocab_size: int | None = None) -> None:
    model.hgm_enabled = bool(unified.cortex.hgm_enabled or unified.features.hgm_enabled)
    model.distillation_config = unified.global_config.distill

    if unified.global_config.stores and model.consolidation_broker is None and broker_vocab_size is not None:
        model.enable_consolidation_broker(vocab_size=int(broker_vocab_size), config_path=config_path)
    elif model.consolidation_broker is not None:
        apply_config_to_broker(model.consolidation_broker, unified.global_config)


def build_cortex_from_yaml(path: str):
    _attach_cortex_yaml_entrypoints()
    from .cortex import EnhancedMnemonicCortex

    unified = load_unified_yaml_config(path)
    model = EnhancedMnemonicCortex(**unified.cortex.to_cortex_kwargs())
    _apply_unified_features(model, unified, config_path=path)
    return model, unified


def _configure_from_yaml(self, path: str, broker_vocab_size: int | None = None):
    unified = load_unified_yaml_config(path)
    _apply_unified_features(self, unified, config_path=path, broker_vocab_size=broker_vocab_size)
    return unified


@classmethod
def _from_yaml(cls, path: str, broker_vocab_size: int | None = None):
    model, _ = build_cortex_from_yaml(path)
    unified = model.configure_from_yaml(path, broker_vocab_size=broker_vocab_size)
    return model, unified


# Keep compatibility hooks attached once so legacy call sites continue to work.
def _attach_cortex_yaml_entrypoints() -> None:
    from .cortex import EnhancedMnemonicCortex

    if not hasattr(EnhancedMnemonicCortex, "configure_from_yaml"):
        setattr(EnhancedMnemonicCortex, "configure_from_yaml", _configure_from_yaml)
    if not hasattr(EnhancedMnemonicCortex, "from_yaml"):
        setattr(EnhancedMnemonicCortex, "from_yaml", _from_yaml)
    if not hasattr(EnhancedMnemonicCortex, "enable_hypergraph_manifold_bridge"):
        def _enable_hypergraph_manifold_bridge(self, enabled: bool = True):
            self.hgm_enabled = bool(enabled)
            return self

        setattr(EnhancedMnemonicCortex, "enable_hypergraph_manifold_bridge", _enable_hypergraph_manifold_bridge)
