"""
Plain-language summary
----------------------
What this file is for: Wiring glue between spatial LTM and the cortex / WM lattice.
How it fits in the system: Connects spatial banks so cortex can mirror and use them.
Status: ACTIVE when spatial LTM enabled
Important notes for non-coders: Mostly integration code, not a standalone memory.

Technical notes (original):
Cortex wiring for spatial LTM banks, extension, and WM lattice mirroring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .ltm import DualTransformerPolicy
from .reasoning_depth.depth_lattice_config import DepthLatticeConfig


@dataclass(frozen=True)
class SpatialWMLatticeMirror:
    """Copy of WM shared-slot lattice topology used by spatial LTM."""

    lattice: DepthLatticeConfig
    namespace: str
    slot_count: int
    num_depths: int
    key_dim: int
    value_dim: int
    read_top_k_slots: int
    read_top_k_depths: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "namespace": self.namespace,
            "slot_count": self.slot_count,
            "num_depths": self.num_depths,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "read_top_k_slots": self.read_top_k_slots,
            "read_top_k_depths": self.read_top_k_depths,
            "lattice": self.lattice.to_dict(),
        }


@dataclass(frozen=True)
class SpatialLTMCortexWiringConfig:
    """Resolved transformer/fusion/decoder settings for spatial LTM."""

    input_dim: int
    output_dim: int
    bank_transformer_layers: int = 3
    bank_transformer_layers_cfg: int = 0
    fixed_transformer_layers: int = 3
    fusion_transformer_layers: int = 4
    decoder_transformer_layers: int = 3
    cross_model_attention_layers: int = 4
    prefusion_specialization_layers: int = 2
    n_heads: int = 8
    bank_transformer_heads: int = 0
    fusion_transformer_heads: int = 0
    cross_model_attention_heads: int = 0
    attention_type: str = "multiscale"
    spatial_slots: int = 256
    spatial_value_dim: int = 0
    spatial_key_dim: int = 64
    spatial_ltm_topk: int = 8
    spatial_conformal_b: float = 0.08
    wm_slot_count: int = 8
    wm_key_dim: int = 256
    wm_value_dim: int = 256
    auto_enable_extension: bool = True
    extension_shared_slots: int = 0
    extension_wm_tf_depth: int = 3
    extension_wm_tf_heads: int = 4
    lattice_mirror_namespace: str = "wm.depth_lattice.spatial_mirror"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_cortex(cls, cortex: Any, *, auto_enable_extension: bool = True) -> "SpatialLTMCortexWiringConfig":
        ltm = getattr(cortex, "long_term_memory", None)
        bank_layers = int(getattr(ltm, "n_transformer_layers", 3) if ltm is not None else 3)
        fusion_layers = int(getattr(ltm, "n_transformer_layers", 3) if ltm is not None else 3)
        if ltm is not None:
            bank_layers = int(getattr(ltm, "n_transformer_layers", bank_layers))
            # triple-hybrid resolves fusion_layers internally; mirror that default.
            fusion_layers = max(4, bank_layers + 1)

        spatial_layers_cfg = int(getattr(cortex, "_ltm_spatial_transformer_layers", 0))
        fixed_layers = int(getattr(cortex, "_ltm_spatial_fixed_transformer_layers", 3))
        policy = DualTransformerPolicy.resolve(
            bank_layers_cfg=spatial_layers_cfg,
            fixed_layers_cfg=fixed_layers,
            fusion_layers_cfg=0,
            decoder_layers_cfg=0,
            inherited_bank_layers=bank_layers,
            inherited_fusion_layers=fusion_layers,
        )

        spatial_slots = int(getattr(ltm, "spatial_slots", 256) if ltm is not None else 256)
        wm_slots = int(getattr(cortex, "_wm_slots", 8))
        input_dim = int(getattr(cortex, "input_dim", 256))
        output_dim = int(getattr(cortex, "output_dim", input_dim))
        mem_heads = int(getattr(ltm, "n_heads", 8) if ltm is not None else 8)

        return cls(
            input_dim=input_dim,
            output_dim=output_dim,
            bank_transformer_layers=policy.bank_layers,
            bank_transformer_layers_cfg=spatial_layers_cfg,
            fixed_transformer_layers=policy.fixed_layers,
            fusion_transformer_layers=policy.fusion_layers,
            decoder_transformer_layers=policy.decoder_layers,
            cross_model_attention_layers=int(getattr(ltm, "n_transformer_layers", 3) if ltm is not None else 3) + 1,
            prefusion_specialization_layers=max(2, bank_layers - 1),
            n_heads=mem_heads,
            attention_type=str(getattr(ltm, "attention_type", "multiscale") if ltm is not None else "multiscale"),
            spatial_slots=spatial_slots,
            spatial_value_dim=int(getattr(ltm, "spatial_value_dim", 0) if ltm is not None else 0),
            spatial_key_dim=int(getattr(ltm, "spatial_key_dim", 64) if ltm is not None else 64),
            spatial_ltm_topk=int(getattr(cortex, "_ltm_spatial_ltm_topk", 8)),
            spatial_conformal_b=float(getattr(cortex, "_ltm_spatial_conformal_b", 0.08)),
            wm_slot_count=max(wm_slots, spatial_slots),
            wm_key_dim=input_dim,
            wm_value_dim=min(256, input_dim),
            auto_enable_extension=bool(auto_enable_extension),
            extension_shared_slots=int(spatial_slots),
            extension_wm_tf_depth=policy.bank_layers,
            extension_wm_tf_heads=max(4, mem_heads // 2),
        )

    def build_wm_lattice_mirror(self) -> SpatialWMLatticeMirror:
        lattice = DepthLatticeConfig.wm(
            slot_count=int(self.wm_slot_count),
            key_dim=int(self.wm_key_dim),
            value_dim=int(self.wm_value_dim),
        )
        return SpatialWMLatticeMirror(
            lattice=lattice,
            namespace=self.lattice_mirror_namespace,
            slot_count=lattice.slot_count,
            num_depths=lattice.num_depths,
            key_dim=lattice.key_dim,
            value_dim=lattice.value_dim,
            read_top_k_slots=lattice.read_top_k_slots,
            read_top_k_depths=lattice.read_top_k_depths,
        )


def build_wm_shared_slot_store_mirror(cfg: SpatialLTMCortexWiringConfig):
    """Create a WM-style shared slot store copy for spatial lattice linkage."""
    from .working_memory.wm_shared_slot_store import SharedSlotStore, SharedSlotStoreConfig

    store_cfg = SharedSlotStoreConfig(
        namespace=cfg.lattice_mirror_namespace,
        dim=int(cfg.wm_value_dim),
    )
    return SharedSlotStore(store_cfg)


def apply_spatial_bank_transformer_config(cortex: Any, wiring: SpatialLTMCortexWiringConfig) -> bool:
    """Re-apply resolved transformer/decoder settings on the triple-hybrid spatial bank."""
    ltm = getattr(cortex, "long_term_memory", None)
    spatial = getattr(ltm, "spatial_ltm", None) if ltm is not None else None
    if spatial is None:
        return False

    core = getattr(spatial, "memory_core", spatial)
    core.transformer_layers = int(wiring.bank_transformer_layers)
    core.fixed_transformer_layers = int(wiring.fixed_transformer_layers)
    core.fusion_transformer_layers = int(wiring.fusion_transformer_layers)
    core.decoder_transformer_layers = int(wiring.decoder_transformer_layers)
    core.cross_model_attention_layers = int(wiring.cross_model_attention_layers)
    if hasattr(core, "rebuild_transformer_stacks"):
        core.rebuild_transformer_stacks(
            bank_layers=wiring.bank_transformer_layers_cfg,
            fixed_layers=wiring.fixed_transformer_layers,
            fusion_layers=0,
            decoder_layers=0,
            n_heads=wiring.n_heads if wiring.bank_transformer_heads <= 0 else wiring.bank_transformer_heads,
            attention_type=wiring.attention_type,
            inherited_bank_layers=int(wiring.bank_transformer_layers),
            inherited_fusion_layers=int(wiring.fusion_transformer_layers),
        )
    spatial.n_transformer_layers = int(wiring.bank_transformer_layers)
    return True


def attach_wm_lattice_mirror_to_spatial(cortex: Any, wiring: SpatialLTMCortexWiringConfig) -> SpatialWMLatticeMirror:
    """Attach a WM lattice topology copy and shared-slot mirror to spatial LTM."""
    mirror = wiring.build_wm_lattice_mirror()
    store = build_wm_shared_slot_store_mirror(wiring)

    ltm = getattr(cortex, "long_term_memory", None)
    spatial = getattr(ltm, "spatial_ltm", None) if ltm is not None else None
    core = getattr(spatial, "memory_core", spatial) if spatial is not None else None
    if core is not None and hasattr(core, "attach_wm_lattice_mirror"):
        core.attach_wm_lattice_mirror(mirror, store)

    extension = getattr(cortex, "spatial_ltm_extension", None)
    if extension is not None:
        extension.wm_lattice_mirror = mirror
        extension.wm_shared_slot_store_mirror = store

    cortex.spatial_wm_lattice_mirror = mirror
    cortex.spatial_wm_shared_slot_store_mirror = store
    return mirror


def build_spatial_extension_config(cortex: Any, wiring: SpatialLTMCortexWiringConfig):
    from .ltm import default_config

    shared_slots = int(wiring.extension_shared_slots or wiring.spatial_slots)
    value_dim = int(wiring.spatial_value_dim) if int(wiring.spatial_value_dim) > 0 else min(256, wiring.input_dim)
    ltm = getattr(cortex, "long_term_memory", None)
    inherited_bank = int(getattr(ltm, "n_transformer_layers", wiring.bank_transformer_layers) if ltm is not None else wiring.bank_transformer_layers)
    return default_config(
        input_dim=wiring.input_dim,
        model_dim=wiring.input_dim,
        output_dim=wiring.output_dim,
        value_dim=value_dim,
        key_dim=int(wiring.spatial_key_dim),
        shared_slots=shared_slots,
        mann_hops=3,
        wm_slots=max(8, int(getattr(cortex, "_wm_slots", 8))),
        wm_dim=value_dim,
        wm_use_transformer=True,
        bank_transformer_layers=int(wiring.bank_transformer_layers_cfg),
        fixed_transformer_layers=int(wiring.fixed_transformer_layers),
        fusion_transformer_layers=0,
        decoder_transformer_layers=0,
        inherited_bank_layers=inherited_bank,
        wm_tf_depth=0,
        reasoning_stack_depth=0,
        wm_tf_heads=int(wiring.extension_wm_tf_heads),
        wm_tf_ffn_mult=2,
        wm_tf_max_len=64,
        wm_tf_trace_cap=8,
        depth_slices=8,
        ltm_topk=int(wiring.spatial_ltm_topk),
        conformal_b=float(wiring.spatial_conformal_b),
    )


def enable_spatial_ltm_extension_for_cortex(
    cortex: Any,
    wiring: Optional[SpatialLTMCortexWiringConfig] = None,
) -> Any:
    from .ltm import EnhancedSpatialMnemonicCortex

    cfg_obj = wiring or SpatialLTMCortexWiringConfig.from_cortex(cortex)
    cfg = build_spatial_extension_config(cortex, cfg_obj)
    device = cortex.ctx_proj.weight.device
    dtype = cortex.ctx_proj.weight.dtype
    extension = EnhancedSpatialMnemonicCortex(cfg).to(device=device, dtype=dtype)
    extension.wiring_config = cfg_obj
    extension.wm_lattice_mirror = getattr(cortex, "spatial_wm_lattice_mirror", None)
    extension.wm_shared_slot_store_mirror = getattr(cortex, "spatial_wm_shared_slot_store_mirror", None)
    cortex.spatial_ltm_extension = extension
    cortex.spatial_ltm_wiring = cfg_obj
    return extension


def wire_spatial_ltm_to_cortex(
    cortex: Any,
    *,
    auto_enable_extension: bool = True,
    rebuild_bank_stacks: bool = True,
) -> Dict[str, Any]:
    """Wire spatial LTM into cortex with WM lattice mirror and transformer parity."""
    wiring = SpatialLTMCortexWiringConfig.from_cortex(
        cortex,
        auto_enable_extension=bool(auto_enable_extension),
    )
    cortex.spatial_ltm_wiring = wiring

    mirror = attach_wm_lattice_mirror_to_spatial(cortex, wiring)
    bank_updated = False
    if rebuild_bank_stacks:
        bank_updated = apply_spatial_bank_transformer_config(cortex, wiring)

    extension = None
    if auto_enable_extension and getattr(cortex, "spatial_ltm_extension", None) is None:
        extension = enable_spatial_ltm_extension_for_cortex(cortex, wiring)

    trace = {
        "trace_type": "spatial_ltm_cortex_wiring",
        "bank_transformer_layers": wiring.bank_transformer_layers,
        "bank_transformer_layers_cfg": wiring.bank_transformer_layers_cfg,
        "fixed_transformer_layers": wiring.fixed_transformer_layers,
        "fusion_transformer_layers": wiring.fusion_transformer_layers,
        "decoder_transformer_layers": wiring.decoder_transformer_layers,
        "dual_stack_active": wiring.fixed_transformer_layers > 0,
        "cross_model_attention_layers": wiring.cross_model_attention_layers,
        "wm_lattice_mirror": mirror.to_dict(),
        "bank_stacks_rebuilt": bool(bank_updated),
        "extension_enabled": extension is not None or getattr(cortex, "spatial_ltm_extension", None) is not None,
    }
    cortex.spatial_ltm_wiring_trace = trace
    if hasattr(cortex, "diagnostics"):
        cortex.diagnostics.log("spatial_ltm_wired", trace)
    return trace
