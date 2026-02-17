from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch

from .ahg import AHGConfig
from .consolidation_broker import StoreConfig


@dataclass
class GlobalConfig:
    stores: Dict[str, StoreConfig]
    ahg: AHGConfig


def load_yaml_config(path: str) -> GlobalConfig:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to load YAML config files.") from exc

    with open(path, "r", encoding="utf-8") as f:
        obj: Dict[str, Any] = yaml.safe_load(f) or {}

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
        ask_on_uncertain=bool(ahg_obj.get("ask_on_uncertain", True)),
        refuse_on_high_risk=bool(ahg_obj.get("refuse_on_high_risk", True)),
    )
    return GlobalConfig(stores=stores, ahg=ahg)


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
