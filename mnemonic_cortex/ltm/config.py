"""Configuration for the Spatial LTM + geometry-aware MANN subsystem."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


DEFAULT_MANN_DEPTH_CHART = ["euclid", "euclid", "euclid", "hyper", "hyper", "sphere", "torus", "spatial"]
DEFAULT_HG_DEPTH_CHART = ["hyper", "hyper", "hyper", "sphere", "sphere", "euclid", "torus", "spatial"]
DEFAULT_CGMN_DEPTH_CHART = ["sphere", "sphere", "hyper", "hyper", "euclid", "euclid", "torus", "torus"]
DEFAULT_CURVED_DEPTH_CHART = ["curved", "hyper", "curved", "euclid", "curved", "torus", "sphere", "curved"]
DEFAULT_SPATIAL_DEPTH_CHART = ["spatial", "spatial", "spatial", "spatial", "sphere", "hyper", "torus", "euclid"]
DEFAULT_PROCEDURAL_DEPTH_CHART = ["sphere", "sphere", "euclid", "torus", "sphere", "euclid", "spatial", "torus"]


@dataclass
class SpatialLtmMannConfig:
    """Runtime-safe defaults for the reconstructed subsystem.

    The dimensions intentionally match the architecture described in the DOCX:
    a shared value store, geometry-specific keys, an 8-slice depth fiber, and
    separate HG/CGMN/Spatial LTM banks plus a traditional MANN reasoning path.
    """

    input_dim: int = 512
    model_dim: int = 512
    value_dim: int = 256
    key_dim: int = 64
    output_dim: int = 512

    shared_slots: int = 256
    depth_slices: int = 8
    phase_bins: int = 8
    scale_bins: int = 4
    spin_bins: int = 8

    ltm_topk: int = 8
    mann_topk: int = 8
    mann_hops: int = 3
    mann_write_lr: float = 0.03
    ltm_write_lr: float = 0.02

    conformal_b: float = 0.08
    min_c: float = 1e-3
    max_c: float = 1.0
    temperature: float = 1.0

    wm_slots: int = 8
    wm_dim: int = 256
    wm_use_transformer: bool = True
    wm_tf_depth: int = 0
    wm_tf_heads: int = 4
    bank_transformer_layers: int = 0
    fixed_transformer_layers: int = 3
    fusion_transformer_layers: int = 0
    decoder_transformer_layers: int = 0
    wm_tf_ffn_mult: int = 2
    wm_tf_max_len: int = 64
    wm_tf_trace_cap: int = 8
    reasoning_stack_depth: int = 0
    inherited_bank_layers: int = 3
    depth_profile: str = "standard"
    dropout: float = 0.0

    sensory_capacity: int = 64
    context_capacity: int = 32

    mann_depth_chart: List[str] = field(default_factory=lambda: list(DEFAULT_MANN_DEPTH_CHART))
    hg_depth_chart: List[str] = field(default_factory=lambda: list(DEFAULT_HG_DEPTH_CHART))
    cgmn_depth_chart: List[str] = field(default_factory=lambda: list(DEFAULT_CGMN_DEPTH_CHART))
    curved_depth_chart: List[str] = field(default_factory=lambda: list(DEFAULT_CURVED_DEPTH_CHART))
    spatial_depth_chart: List[str] = field(default_factory=lambda: list(DEFAULT_SPATIAL_DEPTH_CHART))
    procedural_depth_chart: List[str] = field(default_factory=lambda: list(DEFAULT_PROCEDURAL_DEPTH_CHART))

    def validate(self) -> "SpatialLtmMannConfig":
        if self.depth_slices != 8:
            raise ValueError("Spatial/QSPIN depth-map compatibility expects exactly 8 depth_slices")
        for name, chart in self.depth_charts().items():
            if len(chart) != self.depth_slices:
                raise ValueError(f"{name} chart must have {self.depth_slices} entries")
        if min(self.input_dim, self.model_dim, self.value_dim, self.key_dim, self.shared_slots) <= 0:
            raise ValueError("dimensions and shared_slots must be positive")
        if self.mann_hops <= 0 or self.mann_topk <= 0 or self.ltm_topk <= 0:
            raise ValueError("hop/topk values must be positive")
        return self

    def depth_charts(self) -> Dict[str, List[str]]:
        return {
            "mann": self.mann_depth_chart,
            "hg": self.hg_depth_chart,
            "cgmn": self.cgmn_depth_chart,
            "curved": self.curved_depth_chart,
            "spatial": self.spatial_depth_chart,
            "procedural": self.procedural_depth_chart,
        }

    def resolve_transformer_policy(
        self,
        *,
        inherited_bank_layers: int | None = None,
        inherited_fusion_layers: int | None = None,
    ):
        from .transformer_policy import DualTransformerPolicy

        bank_inherit = int(self.inherited_bank_layers) if inherited_bank_layers is None else int(inherited_bank_layers)
        return DualTransformerPolicy.resolve(
            bank_layers_cfg=int(self.bank_transformer_layers),
            fixed_layers_cfg=int(self.fixed_transformer_layers),
            fusion_layers_cfg=int(self.fusion_transformer_layers),
            decoder_layers_cfg=int(self.decoder_transformer_layers),
            inherited_bank_layers=bank_inherit,
            inherited_fusion_layers=inherited_fusion_layers,
            depth_profile=str(self.depth_profile),
        )

    def effective_wm_tf_depth(self, *, inherited_bank_layers: int | None = None) -> int:
        if int(self.wm_tf_depth) > 0:
            return int(self.wm_tf_depth)
        return int(self.resolve_transformer_policy(inherited_bank_layers=inherited_bank_layers).bank_layers)

    def effective_reasoning_stack_depth(self, *, inherited_bank_layers: int | None = None) -> int:
        if int(self.reasoning_stack_depth) > 0:
            return int(self.reasoning_stack_depth)
        return int(self.resolve_transformer_policy(inherited_bank_layers=inherited_bank_layers).fixed_layers)


def default_config(**overrides) -> SpatialLtmMannConfig:
    cfg = SpatialLtmMannConfig(**overrides)
    return cfg.validate()
