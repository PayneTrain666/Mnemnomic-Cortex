from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import warnings

import torch

from .ahg import AHGConfig
from .config import CortexConfig
from .consolidation_broker import StoreConfig
from .distillation import DistillationConfig


@dataclass
class GlobalConfig:
    stores: Dict[str, StoreConfig]
    ahg: AHGConfig
    distill: DistillationConfig


@dataclass
class FeatureTogglesConfig:
    hgm_enabled: Optional[bool] = None
    reasoning_bridge_enabled: Optional[bool] = None
    qdt_wm_bridge_enabled: Optional[bool] = None
    shared_memory_enabled: Optional[bool] = None
    hg_episodic_ltm_enabled: Optional[bool] = None


@dataclass
class UnifiedCortexConfig:
    cortex: CortexConfig
    global_config: GlobalConfig
    features: FeatureTogglesConfig


def _parse_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    return bool(default)


def _parse_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    return _parse_bool(value, False)


def _load_yaml_obj(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to load YAML config files.") from exc

    with open(path, "r", encoding="utf-8") as f:
        obj: Dict[str, Any] = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        return {}
    return obj


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
        ask_on_uncertain=_parse_bool(ahg_obj.get("ask_on_uncertain", True), True),
        refuse_on_high_risk=_parse_bool(ahg_obj.get("refuse_on_high_risk", True), True),
    )
    ahg.validate()
    distill_obj = obj.get("distill", {}) or {}
    students = distill_obj.get("student_domains", ("reasoning",))
    if isinstance(students, str):
        students = [s.strip() for s in students.split(",") if s.strip()]
    distill = DistillationConfig(
        enabled=_parse_bool(distill_obj.get("enabled", False), False),
        embedding_weight=float(distill_obj.get("embedding_weight", 0.5)),
        mse_weight=float(distill_obj.get("mse_weight", 0.1)),
        neighbor_kl_weight=float(distill_obj.get("neighbor_kl_weight", 0.2)),
        cms_teacher_weight=float(distill_obj.get("cms_teacher_weight", 0.2)),
        teacher_domain=str(distill_obj.get("teacher_domain", "core")),
        student_domains=tuple(students),
        neighbor_k=int(distill_obj.get("neighbor_k", 16)),
        sim_temp=float(distill_obj.get("sim_temp", 0.07)),
    )
    distill.validate()
    return GlobalConfig(stores=stores, ahg=ahg, distill=distill)


def _parse_cortex_config(obj: Dict[str, Any], features: FeatureTogglesConfig) -> CortexConfig:
    defaults = CortexConfig()
    payload = dict(getattr(defaults, "__dict__", {}))
    root_known = {k: obj[k] for k in payload.keys() if k in obj}
    payload.update(root_known)
    section = obj.get("cortex", {}) or {}
    if isinstance(section, dict):
        payload.update({k: section[k] for k in payload.keys() if k in section})
    # Promote canonical feature toggle into constructor config.
    if features.hgm_enabled is not None:
        payload["hgm_enabled"] = bool(features.hgm_enabled)
    else:
        payload["hgm_enabled"] = bool(payload.get("hgm_enabled", False))
    return CortexConfig(**payload)


def _parse_feature_toggles(obj: Dict[str, Any]) -> FeatureTogglesConfig:
    sec = obj.get("features", {}) or {}
    if not isinstance(sec, dict):
        sec = {}
    return FeatureTogglesConfig(
        hgm_enabled=_parse_optional_bool(sec.get("hgm_enabled", obj.get("hgm_enabled", None))),
        reasoning_bridge_enabled=_parse_optional_bool(sec.get("reasoning_bridge_enabled", None)),
        qdt_wm_bridge_enabled=_parse_optional_bool(sec.get("qdt_wm_bridge_enabled", None)),
        shared_memory_enabled=_parse_optional_bool(sec.get("shared_memory_enabled", None)),
        hg_episodic_ltm_enabled=_parse_optional_bool(sec.get("hg_episodic_ltm_enabled", None)),
    )


def load_unified_yaml_config(path: str) -> UnifiedCortexConfig:
    obj = _load_yaml_obj(path)
    features = _parse_feature_toggles(obj)
    cortex = _parse_cortex_config(obj, features)
    gcfg = _parse_global_config(obj)
    return UnifiedCortexConfig(cortex=cortex, global_config=gcfg, features=features)


def load_yaml_config(path: str) -> GlobalConfig:
    # Backward-compatible broker/AHG/distill-only entry point.
    unified = load_unified_yaml_config(path)
    return unified.global_config


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
                if hasattr(store, "b"):
                    store.b.copy_(torch.tensor(float(sc.conformal_b), device=store.b.device, dtype=store.b.dtype))
                if hasattr(store, "k") and int(getattr(store, "k")) != int(sc.senses):
                    warnings.warn(
                        f"Store '{name}' built with senses={int(getattr(store, 'k'))}, "
                        f"config requests senses={int(sc.senses)}; rebuild required for sense-count changes.",
                        RuntimeWarning,
                    )


def apply_unified_config_to_cortex(model, unified: UnifiedCortexConfig):
    if model is None or unified is None:
        return model
    if hasattr(model, "enable_hypergraph_manifold_bridge") and unified.features.hgm_enabled is not None:
        model.enable_hypergraph_manifold_bridge(enabled=bool(unified.features.hgm_enabled))
    if hasattr(model, "enable_reasoning_controller_bridge") and bool(unified.features.reasoning_bridge_enabled):
        model.enable_reasoning_controller_bridge(enabled=True)
    if hasattr(model, "enable_qdt_working_memory_bridge") and bool(unified.features.qdt_wm_bridge_enabled):
        model.enable_qdt_working_memory_bridge()
    if hasattr(model, "enable_shared_memory_subsystem") and bool(unified.features.shared_memory_enabled):
        model.enable_shared_memory_subsystem()
    if hasattr(model, "enable_hg_episodic_ltm") and bool(unified.features.hg_episodic_ltm_enabled):
        model.enable_hg_episodic_ltm()
    if hasattr(model, "ahg"):
        model.ahg = model.ahg.__class__(unified.global_config.ahg)
    if hasattr(model, "configure_distillation"):
        model.configure_distillation(unified.global_config.distill)
    if hasattr(model, "consolidation_broker"):
        apply_config_to_broker(getattr(model, "consolidation_broker"), unified.global_config)
    return model


def build_cortex_from_yaml(path: str):
    unified = load_unified_yaml_config(path)
    from .cortex import EnhancedMnemonicCortex

    model = EnhancedMnemonicCortex(**unified.cortex.to_cortex_kwargs())
    apply_unified_config_to_cortex(model, unified)
    return model, unified
