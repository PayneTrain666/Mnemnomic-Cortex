"""Cortex wiring for HG episodic LTM, triple-hybrid HG bank linkage, and WM lattice mirroring."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .ltm import DualTransformerPolicy
from .reasoning_depth.depth_lattice_config import DepthLatticeConfig


@dataclass(frozen=True)
class HGEpisodicWMLatticeMirror:
    """Copy of WM shared-slot lattice topology used by HG episodic LTM."""

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
class HGEpisodicCortexWiringConfig:
    """Resolved transformer/fusion/decoder settings for HG episodic LTM."""

    input_dim: int
    output_dim: int
    bank_transformer_layers: int = 3
    bank_transformer_layers_cfg: int = 0
    fixed_transformer_layers: int = 3
    fusion_transformer_layers: int = 4
    decoder_transformer_layers: int = 3
    cross_model_attention_layers: int = 4
    n_heads: int = 8
    attention_type: str = "multiscale"
    wm_slot_count: int = 8
    wm_key_dim: int = 256
    wm_value_dim: int = 256
    lattice_mirror_namespace: str = "wm.depth_lattice.hg_episodic_mirror"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_cortex(cls, cortex: Any) -> "HGEpisodicCortexWiringConfig":
        ltm = getattr(cortex, "long_term_memory", None)
        bank_layers = int(getattr(ltm, "n_transformer_layers", 3) if ltm is not None else 3)
        fusion_layers = max(4, bank_layers + 1)
        episodic_cfg = int(getattr(cortex, "_ltm_hg_episodic_transformer_layers", 0))
        fixed_layers = int(getattr(cortex, "_ltm_hg_episodic_fixed_transformer_layers", 3))
        policy = DualTransformerPolicy.resolve(
            bank_layers_cfg=episodic_cfg,
            fixed_layers_cfg=fixed_layers,
            fusion_layers_cfg=0,
            decoder_layers_cfg=0,
            inherited_bank_layers=bank_layers,
            inherited_fusion_layers=fusion_layers,
        )
        input_dim = int(getattr(cortex, "input_dim", 256))
        output_dim = int(getattr(cortex, "output_dim", input_dim))
        mem_heads = int(getattr(ltm, "n_heads", 8) if ltm is not None else 8)
        wm_slots = int(getattr(cortex, "_wm_slots", 8))
        return cls(
            input_dim=input_dim,
            output_dim=output_dim,
            bank_transformer_layers=policy.bank_layers,
            bank_transformer_layers_cfg=episodic_cfg,
            fixed_transformer_layers=policy.fixed_layers,
            fusion_transformer_layers=policy.fusion_layers,
            decoder_transformer_layers=policy.decoder_layers,
            cross_model_attention_layers=max(4, bank_layers + 1),
            n_heads=mem_heads,
            attention_type=str(getattr(ltm, "attention_type", "multiscale") if ltm is not None else "multiscale"),
            wm_slot_count=max(wm_slots, 64),
            wm_key_dim=input_dim,
            wm_value_dim=min(256, input_dim),
        )

    def build_wm_lattice_mirror(self) -> HGEpisodicWMLatticeMirror:
        lattice = DepthLatticeConfig.wm(
            slot_count=int(self.wm_slot_count),
            key_dim=int(self.wm_key_dim),
            value_dim=int(self.wm_value_dim),
        )
        return HGEpisodicWMLatticeMirror(
            lattice=lattice,
            namespace=self.lattice_mirror_namespace,
            slot_count=lattice.slot_count,
            num_depths=lattice.num_depths,
            key_dim=lattice.key_dim,
            value_dim=lattice.value_dim,
            read_top_k_slots=lattice.read_top_k_slots,
            read_top_k_depths=lattice.read_top_k_depths,
        )


def build_wm_shared_slot_store_mirror(cfg: HGEpisodicCortexWiringConfig):
    """Create a WM-style shared slot store copy for episodic lattice linkage."""
    from .working_memory.wm_shared_slot_store import SharedSlotStore, SharedSlotStoreConfig

    store_cfg = SharedSlotStoreConfig(
        namespace=cfg.lattice_mirror_namespace,
        dim=int(cfg.wm_value_dim),
    )
    return SharedSlotStore(store_cfg)


def apply_hg_episodic_transformer_config(cortex: Any, wiring: HGEpisodicCortexWiringConfig) -> bool:
    """Re-apply resolved transformer/decoder settings on HG episodic LTM."""
    episodic = getattr(cortex, "hg_episodic_ltm", None)
    if episodic is None:
        return False
    if hasattr(episodic, "rebuild_transformer_stacks"):
        episodic.rebuild_transformer_stacks(
            bank_layers=wiring.bank_transformer_layers_cfg,
            fixed_layers=wiring.fixed_transformer_layers,
            fusion_layers=0,
            decoder_layers=0,
            n_heads=wiring.n_heads,
            attention_type=wiring.attention_type,
            inherited_bank_layers=int(wiring.bank_transformer_layers),
            inherited_fusion_layers=int(wiring.fusion_transformer_layers),
        )
    episodic.transformer_layers = int(wiring.bank_transformer_layers)
    episodic.fixed_transformer_layers = int(wiring.fixed_transformer_layers)
    episodic.fusion_transformer_layers = int(wiring.fusion_transformer_layers)
    episodic.decoder_transformer_layers = int(wiring.decoder_transformer_layers)
    episodic.cross_model_attention_layers = int(wiring.cross_model_attention_layers)
    return True


def attach_wm_lattice_mirror_to_hg_episodic(
    cortex: Any,
    wiring: HGEpisodicCortexWiringConfig,
) -> HGEpisodicWMLatticeMirror:
    """Attach a WM lattice topology copy and shared-slot mirror to HG episodic LTM."""
    mirror = wiring.build_wm_lattice_mirror()
    store = build_wm_shared_slot_store_mirror(wiring)
    episodic = getattr(cortex, "hg_episodic_ltm", None)
    if episodic is not None and hasattr(episodic, "attach_wm_lattice_mirror"):
        episodic.attach_wm_lattice_mirror(mirror, store)
    cortex.hg_episodic_wm_lattice_mirror = mirror
    cortex.hg_episodic_wm_shared_slot_store_mirror = store
    return mirror


def link_hg_episodic_to_triple_hybrid(cortex: Any) -> Dict[str, Any]:
    """Register episodic LTM on triple-hybrid for ingest/sync bridge callbacks."""
    ltm = getattr(cortex, "long_term_memory", None)
    episodic = getattr(cortex, "hg_episodic_ltm", None)
    if ltm is None or episodic is None:
        return {"linked": False}
    if hasattr(ltm, "link_hg_episodic_bridge"):
        ltm.link_hg_episodic_bridge(episodic)
    cortex.hg_episodic_triple_hybrid_bridge = episodic
    return {
        "linked": True,
        "hg_bank_available": getattr(ltm, "hg", None) is not None,
        "episodic_records": len(getattr(episodic, "episode_records", {})),
    }


def wire_hg_episodic_to_cortex(
    cortex: Any,
    *,
    rebuild_stacks: bool = True,
    link_triple_hybrid: bool = True,
) -> Dict[str, Any]:
    """Wire HG episodic LTM into cortex with WM lattice mirror and transformer parity."""
    wiring = HGEpisodicCortexWiringConfig.from_cortex(cortex)
    cortex.hg_episodic_wiring = wiring

    mirror = attach_wm_lattice_mirror_to_hg_episodic(cortex, wiring)
    stacks_updated = False
    if rebuild_stacks:
        stacks_updated = apply_hg_episodic_transformer_config(cortex, wiring)

    bridge = {}
    if link_triple_hybrid:
        bridge = link_hg_episodic_to_triple_hybrid(cortex)

    trace = {
        "trace_type": "hg_episodic_cortex_wiring",
        "bank_transformer_layers": wiring.bank_transformer_layers,
        "bank_transformer_layers_cfg": wiring.bank_transformer_layers_cfg,
        "fixed_transformer_layers": wiring.fixed_transformer_layers,
        "fusion_transformer_layers": wiring.fusion_transformer_layers,
        "decoder_transformer_layers": wiring.decoder_transformer_layers,
        "cross_model_attention_layers": wiring.cross_model_attention_layers,
        "dual_stack_active": wiring.fixed_transformer_layers > 0,
        "wm_lattice_mirror": mirror.to_dict(),
        "stacks_rebuilt": bool(stacks_updated),
        "triple_hybrid_bridge": bridge,
    }
    cortex.hg_episodic_wiring_trace = trace
    if hasattr(cortex, "diagnostics"):
        cortex.diagnostics.log("hg_episodic_wired", trace)
    return trace
