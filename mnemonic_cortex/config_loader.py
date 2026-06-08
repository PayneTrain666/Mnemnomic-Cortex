from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, TYPE_CHECKING

import torch

from .ahg import AHGConfig
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
    input_dim: int = 256
    output_dim: int = 256
    wm_slots: int = 7
    fusion: str = "weighted"
    hgm_enabled: bool = False


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

    cortex_cfg = CortexBuildConfig(
        input_dim=int(cortex_obj.get("input_dim", 256)),
        output_dim=int(cortex_obj.get("output_dim", cortex_obj.get("input_dim", 256))),
        wm_slots=int(cortex_obj.get("wm_slots", 7)),
        fusion=str(cortex_obj.get("fusion", "weighted")),
        hgm_enabled=_as_bool(cortex_obj.get("hgm_enabled", False), False),
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
    model = EnhancedMnemonicCortex(
        input_dim=int(unified.cortex.input_dim),
        output_dim=int(unified.cortex.output_dim),
        wm_slots=int(unified.cortex.wm_slots),
        fusion=str(unified.cortex.fusion),
    )
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
