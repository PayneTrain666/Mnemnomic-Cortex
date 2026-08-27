"""
Plain-language summary
----------------------
What this file is for: Opt-in dual-fusion policy testbed. Hardcoded
scenario/chart priors plus residual trainable logits and runtime condition
overlays. Gate 0 is the hardcoded recipe. Off leaves the historical four
weights in place.
How it fits in the system: When enabled, it supersedes WMDualFusionController's
fixed WM/LTM/MANN/SPCP mix. Fusion still happens in the shared tangent.
Status: OPT-IN testbed for finetuning / recipe search. Not production.
Important notes: Does not write LTM/MANN/shared slots/QH or activate QSPIN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .context_geometry_maps import build_default_context_geometry_maps


FUSION_SYSTEMS: Tuple[str, ...] = ("wm", "ltm", "mann", "spcp")

# Historical dual-fusion mix. Unknown maps and policy-off both fall back here
# unless DualFusionConfig overrides the four scalars.
LEGACY_FUSION_WEIGHTS: Tuple[float, float, float, float] = (0.55, 0.18, 0.18, 0.09)

# Blend between role prior (scenario purpose) and chart-histogram prior.
SCENARIO_CHART_BLEND: float = 0.40

CHART_ALIASES: Dict[str, str] = {
    "euclid": "euclidean",
    "sphere": "spherical",
    "grassmannian": "grassmann",
    "cp": "complex_projective",
    "poincare": "poincare",
}


def canonicalize_chart(name: Optional[str]) -> str:
    raw = str(name or "euclidean").strip().lower()
    return CHART_ALIASES.get(raw, raw) or "euclidean"


def _norm4(values: Sequence[float], eps: float = 1e-8) -> Tuple[float, float, float, float]:
    vals = [max(0.0, float(v)) for v in list(values)[:4]]
    while len(vals) < 4:
        vals.append(0.0)
    total = sum(vals)
    if total <= eps:
        return LEGACY_FUSION_WEIGHTS
    return tuple(v / total for v in vals)  # type: ignore[return-value]


# Per-chart pull on the four fusion sources. Euclidean stays near the historical
# mix. Hyperbolic/Poincaré prefer LTM (trees/facts). Spherical/quaternion prefer
# MANN (hops/pose). Torus/SPCP prefer procedural memory. CP/phase prefer LTM.
CHART_SYSTEM_BIAS: Dict[str, Tuple[float, float, float, float]] = {
    "euclidean": (0.55, 0.18, 0.18, 0.09),
    "tangent_bridge": (0.50, 0.20, 0.18, 0.12),
    "hyperbolic": (0.20, 0.50, 0.18, 0.12),
    "poincare": (0.18, 0.52, 0.18, 0.12),
    "fiber_bundle": (0.22, 0.36, 0.24, 0.18),
    "spherical": (0.22, 0.18, 0.42, 0.18),
    "quaternion": (0.24, 0.14, 0.46, 0.16),
    "dual_quaternion": (0.22, 0.12, 0.48, 0.18),
    "spatial_se3": (0.30, 0.12, 0.42, 0.16),
    "torus": (0.22, 0.16, 0.22, 0.40),
    "spcp": (0.18, 0.14, 0.22, 0.46),
    "complex": (0.24, 0.28, 0.20, 0.28),
    "complex_projective": (0.22, 0.40, 0.16, 0.22),
    "cp_kahler": (0.20, 0.42, 0.16, 0.22),
    "holographic_phase": (0.24, 0.32, 0.18, 0.26),
    "subspace": (0.28, 0.32, 0.22, 0.18),
    "grassmann": (0.28, 0.32, 0.22, 0.18),
    "product": (0.28, 0.28, 0.22, 0.22),
}

# Role priors from each context-map purpose. These supersede the historical
# four-scalar mix when the policy is on.
SCENARIO_ROLE_PRIORS: Dict[str, Tuple[float, float, float, float]] = {
    "literal": (0.70, 0.12, 0.10, 0.08),
    "hierarchical": (0.28, 0.42, 0.18, 0.12),
    "temporal": (0.30, 0.18, 0.22, 0.30),
    "spatial_mechanical": (0.32, 0.12, 0.40, 0.16),
    "symbolic_mathematical": (0.28, 0.38, 0.16, 0.18),
    "procedural": (0.22, 0.14, 0.26, 0.38),
    "conflict_verification": (0.34, 0.36, 0.18, 0.12),
    "creative_synthesis": (0.24, 0.22, 0.28, 0.26),
    "policy_governance": (0.40, 0.32, 0.12, 0.16),
    "quantum_holographic": (0.26, 0.30, 0.20, 0.24),
}

# Additive log-space condition directions. Scaled by runtime stats and
# condition_mix. Disagreement trusts current WM tokens; low confidence
# leans on canonical LTM.
CONDITION_DISAGREEMENT: Tuple[float, float, float, float] = (0.80, -0.30, -0.30, -0.20)
CONDITION_LOW_CONFIDENCE: Tuple[float, float, float, float] = (-0.20, 0.70, -0.20, -0.30)


def chart_bias(name: str) -> Tuple[float, float, float, float]:
    key = canonicalize_chart(name)
    return _norm4(CHART_SYSTEM_BIAS.get(key, LEGACY_FUSION_WEIGHTS))


def chart_histogram_prior(
    charts: Sequence[str],
    depth_weights: Optional[Sequence[float]] = None,
) -> Tuple[float, float, float, float]:
    names = [canonicalize_chart(name) for name in charts if str(name).strip()]
    if not names:
        return LEGACY_FUSION_WEIGHTS
    weights = list(depth_weights or [])
    if len(weights) < len(names):
        weights = weights + [1.0] * (len(names) - len(weights))
    acc = [0.0, 0.0, 0.0, 0.0]
    for name, weight in zip(names, weights[: len(names)]):
        bias = chart_bias(name)
        scale = max(0.0, float(weight))
        for idx in range(4):
            acc[idx] += scale * bias[idx]
    return _norm4(acc)


def scenario_prior(
    map_name: Optional[str],
    *,
    charts: Optional[Sequence[str]] = None,
    depth_weights: Optional[Sequence[float]] = None,
    legacy: Sequence[float] = LEGACY_FUSION_WEIGHTS,
    chart_blend: float = SCENARIO_CHART_BLEND,
) -> Tuple[float, float, float, float]:
    key = str(map_name or "").strip().lower()
    role = SCENARIO_ROLE_PRIORS.get(key)
    if role is None and not charts:
        return _norm4(legacy)
    if role is None:
        role = _norm4(legacy)
    hist = chart_histogram_prior(charts or [], depth_weights=depth_weights)
    mix = float(max(0.0, min(1.0, chart_blend)))
    blended = [(1.0 - mix) * role[i] + mix * hist[i] for i in range(4)]
    return _norm4(blended)


def build_scenario_prior_table(
    *,
    num_depths: int = 8,
    legacy: Sequence[float] = LEGACY_FUSION_WEIGHTS,
    chart_blend: float = SCENARIO_CHART_BLEND,
) -> Dict[str, Tuple[float, float, float, float]]:
    maps = build_default_context_geometry_maps(num_depths)
    table = {"default": _norm4(legacy)}
    for name, spec in maps.items():
        table[name] = scenario_prior(
            name,
            charts=spec.geometry_by_depth,
            depth_weights=spec.depth_weights,
            legacy=legacy,
            chart_blend=chart_blend,
        )
    return table


@dataclass
class WMChartFusionPolicyConfig:
    """Opt-in fusion-policy testbed configuration.

    enable=False keeps historical four-scalar fusion. enable=True supersedes
    those scalars with scenario/chart priors. gate=0 is the hardcoded recipe.
    Open the gate (or call open_for_finetune) before training residuals.
    """

    enable: bool = False
    gate_init: float = 0.0
    condition_mix: float = 0.15
    chart_blend: float = SCENARIO_CHART_BLEND
    num_depths: int = 8
    eps: float = 1e-8
    legacy_weights: Tuple[float, float, float, float] = LEGACY_FUSION_WEIGHTS

    def validate(self) -> None:
        if not 0.0 <= float(self.gate_init) <= 1.0:
            raise ValueError("gate_init must be in [0,1]")
        if not 0.0 <= float(self.condition_mix) <= 1.0:
            raise ValueError("condition_mix must be in [0,1]")
        if not 0.0 <= float(self.chart_blend) <= 1.0:
            raise ValueError("chart_blend must be in [0,1]")
        if int(self.num_depths) <= 0:
            raise ValueError("num_depths must be positive")
        if float(self.eps) <= 0:
            raise ValueError("eps must be positive")
        _norm4(self.legacy_weights, eps=self.eps)


@dataclass
class WMChartFusionPolicyOutput:
    weights: torch.Tensor
    prior: torch.Tensor
    gate: float
    map_name: str
    trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weight_shape": list(self.weights.shape),
            "prior": self.prior.detach().cpu().tolist(),
            "gate": float(self.gate),
            "map_name": self.map_name,
            "trace": self.trace,
        }


class WMChartFusionPolicy(nn.Module):
    """Hardcoded scenario/chart fusion priors with residual trainable logits.

    weights = softmax(log(prior) + gate * learned + condition_mix * conditions)

    Learned terms are global + per-map + mean per-chart residuals, all zero
    at init. Priors are buffers (not trained unless an outer loop writes them).
    """

    def __init__(self, config: Optional[WMChartFusionPolicyConfig] = None):
        super().__init__()
        cfg = config or WMChartFusionPolicyConfig()
        cfg.validate()
        self.config = cfg
        table = build_scenario_prior_table(
            num_depths=cfg.num_depths,
            legacy=cfg.legacy_weights,
            chart_blend=cfg.chart_blend,
        )
        self._maps = build_default_context_geometry_maps(cfg.num_depths)
        self.map_names: List[str] = sorted(table.keys())
        self.map_index = {name: idx for idx, name in enumerate(self.map_names)}
        self.chart_names: List[str] = sorted(set(CHART_SYSTEM_BIAS) | set(CHART_ALIASES.values()))
        self.chart_index = {name: idx for idx, name in enumerate(self.chart_names)}

        prior_rows = [table[name] for name in self.map_names]
        self.register_buffer("prior_table", torch.tensor(prior_rows, dtype=torch.float32))
        chart_rows = [chart_bias(name) for name in self.chart_names]
        self.register_buffer("chart_bias_table", torch.tensor(chart_rows, dtype=torch.float32))

        self.gate = nn.Parameter(torch.tensor(float(cfg.gate_init)))
        self.global_logits = nn.Parameter(torch.zeros(len(FUSION_SYSTEMS)))
        self.scenario_logits = nn.Parameter(torch.zeros(len(self.map_names), len(FUSION_SYSTEMS)))
        self.chart_logits = nn.Parameter(torch.zeros(len(self.chart_names), len(FUSION_SYSTEMS)))
        self.last_output: Optional[WMChartFusionPolicyOutput] = None
        self.last_stats: Dict[str, float] = {}

    def _map_key(self, map_name: Optional[str]) -> str:
        key = str(map_name or "").strip().lower()
        if key in self.map_index:
            return key
        return "default"

    def hardcoded_prior(
        self,
        map_name: Optional[str],
        charts: Optional[Sequence[str]] = None,
        depth_weights: Optional[Sequence[float]] = None,
    ) -> Tuple[float, float, float, float]:
        spec = self._maps.get(str(map_name or "").strip().lower())
        use_charts = list(charts or [])
        weights = list(depth_weights or [])
        if spec is not None and not use_charts:
            use_charts = list(spec.geometry_by_depth)
            weights = list(spec.depth_weights)
        return scenario_prior(
            map_name,
            charts=use_charts,
            depth_weights=weights,
            legacy=self.config.legacy_weights,
            chart_blend=self.config.chart_blend,
        )

    def _learned_delta(
        self,
        map_key: str,
        charts: Sequence[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        delta = self.global_logits.to(device=device, dtype=dtype)
        delta = delta + self.scenario_logits[self.map_index[map_key]].to(device=device, dtype=dtype)
        chart_ids = []
        for name in charts:
            idx = self.chart_index.get(canonicalize_chart(name))
            if idx is not None:
                chart_ids.append(idx)
        if chart_ids:
            idx_t = torch.tensor(chart_ids, device=device, dtype=torch.long)
            delta = delta + self.chart_logits.index_select(0, idx_t).to(dtype=dtype).mean(dim=0)
        return delta

    def _condition_logits(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        confidence: Optional[torch.Tensor],
        disagreement: Optional[torch.Tensor],
    ) -> torch.Tensor:
        mix = float(self.config.condition_mix)
        zeros = torch.zeros(batch, len(FUSION_SYSTEMS), device=device, dtype=dtype)
        if mix <= 0.0:
            return zeros
        if confidence is None and disagreement is None:
            return zeros
        if confidence is None:
            conf = torch.ones(batch, device=device, dtype=dtype)
        else:
            conf = confidence.to(device=device, dtype=dtype).reshape(-1)
            if conf.numel() == 1 and batch > 1:
                conf = conf.expand(batch)
            elif conf.numel() != batch:
                conf = conf.mean().expand(batch)
            conf = conf.clamp(0.0, 1.0)
        if disagreement is None:
            disagree = torch.zeros(batch, device=device, dtype=dtype)
        else:
            disagree = disagreement.to(device=device, dtype=dtype).reshape(-1)
            if disagree.numel() == 1 and batch > 1:
                disagree = disagree.expand(batch)
            elif disagree.numel() != batch:
                disagree = disagree.mean().expand(batch)
            disagree = torch.tanh(disagree.clamp_min(0.0))
        low_conf = (1.0 - conf).unsqueeze(-1)
        disagree_n = disagree.unsqueeze(-1)
        disagree_dir = torch.tensor(CONDITION_DISAGREEMENT, device=device, dtype=dtype)
        low_conf_dir = torch.tensor(CONDITION_LOW_CONFIDENCE, device=device, dtype=dtype)
        return mix * (disagree_n * disagree_dir + low_conf * low_conf_dir)

    def mix_sources(
        self,
        weights: torch.Tensor,
        wm_context: torch.Tensor,
        ltm_context: torch.Tensor,
        mann_context: torch.Tensor,
        spcp_context: torch.Tensor,
    ) -> torch.Tensor:
        """Tangent-space mix of the four fusion sources. weights [4] or [B,4]."""
        sources = torch.stack([wm_context, ltm_context, mann_context, spcp_context], dim=1)
        if weights.dim() == 1:
            return torch.einsum("s,bsd->bd", weights, sources)
        return torch.einsum("bs,bsd->bd", weights, sources)

    def open_for_finetune(self, gate: float = 0.10) -> None:
        """Raise the residual gate so learned logits can receive gradients."""
        value = float(max(0.0, min(1.0, gate)))
        with torch.no_grad():
            self.gate.fill_(value)

    def freeze_priors(self) -> None:
        """Priors are buffers; this only freezes the residual parameters."""
        self.global_logits.requires_grad_(False)
        self.scenario_logits.requires_grad_(False)
        self.chart_logits.requires_grad_(False)
        self.gate.requires_grad_(False)

    def unfreeze_residuals(self) -> None:
        self.global_logits.requires_grad_(True)
        self.scenario_logits.requires_grad_(True)
        self.chart_logits.requires_grad_(True)
        self.gate.requires_grad_(True)

    def finetune_parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Named groups for a finetune optimizer. Train gate before or with logits."""
        return {
            "gate": [self.gate],
            "global_logits": [self.global_logits],
            "scenario_logits": [self.scenario_logits],
            "chart_logits": [self.chart_logits],
        }

    def snapshot_recipe(self) -> Dict[str, Any]:
        """Serialization-safe recipe for outer-loop search / eval logging."""
        gate = float(self.gate.detach().clamp(0.0, 1.0).item())
        priors = {
            name: self.prior_table[idx].detach().cpu().tolist()
            for name, idx in self.map_index.items()
        }
        return {
            "testbed": "wm_chart_fusion_policy",
            "enabled": bool(self.config.enable),
            "gate": gate,
            "condition_mix": float(self.config.condition_mix),
            "chart_blend": float(self.config.chart_blend),
            "systems": list(FUSION_SYSTEMS),
            "priors": priors,
            "role_priors": {k: list(v) for k, v in SCENARIO_ROLE_PRIORS.items()},
            "chart_bias": {k: list(v) for k, v in CHART_SYSTEM_BIAS.items()},
            "learned": {
                "global_logits": self.global_logits.detach().cpu().tolist(),
                "scenario_logits": {
                    name: self.scenario_logits[idx].detach().cpu().tolist()
                    for name, idx in self.map_index.items()
                },
            },
            "qspin_live_routing": False,
            "shared_slot_writes": False,
        }

    def forward(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        map_name: Optional[str] = None,
        charts: Optional[Sequence[str]] = None,
        depth_weights: Optional[Sequence[float]] = None,
        confidence: Optional[torch.Tensor] = None,
        disagreement: Optional[torch.Tensor] = None,
    ) -> WMChartFusionPolicyOutput:
        map_key = self._map_key(map_name)
        spec = self._maps.get(str(map_name or "").strip().lower())
        use_charts = [canonicalize_chart(name) for name in (charts or []) if str(name).strip()]
        use_weights = list(depth_weights or [])
        if spec is not None and not use_charts:
            use_charts = list(spec.geometry_by_depth)
            use_weights = list(spec.depth_weights)
        if spec is not None:
            prior_vals = scenario_prior(
                spec.name,
                charts=use_charts,
                depth_weights=use_weights,
                legacy=self.config.legacy_weights,
                chart_blend=self.config.chart_blend,
            )
        elif use_charts:
            prior_vals = scenario_prior(
                None,
                charts=use_charts,
                depth_weights=use_weights,
                legacy=self.config.legacy_weights,
                chart_blend=self.config.chart_blend,
            )
        else:
            prior_vals = _norm4(self.config.legacy_weights, eps=self.config.eps)
        prior_t = torch.tensor(prior_vals, device=device, dtype=dtype)

        gate_t = self.gate.clamp(0.0, 1.0)
        log_prior = torch.log(prior_t.clamp_min(self.config.eps))
        delta = self._learned_delta(map_key, use_charts, device, dtype)
        conditions = self._condition_logits(batch, device, dtype, confidence, disagreement)
        logits = log_prior.unsqueeze(0) + gate_t * delta.unsqueeze(0) + conditions
        weights = torch.softmax(logits, dim=-1)
        gate = float(gate_t.detach().item())
        prior_b = prior_t.unsqueeze(0).expand(batch, -1)
        kl = (weights * (torch.log(weights.clamp_min(self.config.eps)) - torch.log(prior_b.clamp_min(self.config.eps)))).sum(dim=-1).mean()
        trace = {
            "enabled": True,
            "map_name": map_key if spec is not None else str(map_name or "default"),
            "resolved_map": map_key,
            "charts": list(use_charts),
            "prior": prior_t.detach().cpu().tolist(),
            "weights_mean": weights.mean(dim=0).detach().cpu().tolist(),
            "gate": gate,
            "condition_mix": float(self.config.condition_mix),
            "chart_blend": float(self.config.chart_blend),
            "kl_to_prior": float(kl.detach().cpu()),
            "systems": list(FUSION_SYSTEMS),
            "qspin_live_routing": False,
            "shared_slot_writes": False,
            "testbed": True,
        }
        out = WMChartFusionPolicyOutput(
            weights=weights,
            prior=prior_t,
            gate=gate,
            map_name=trace["map_name"],
            trace=trace,
        )
        self.last_output = out
        self.last_stats = {
            "enabled": 1.0,
            "gate": gate,
            "condition_mix": float(self.config.condition_mix),
            "kl_to_prior": float(kl.detach().cpu()),
        }
        return out


def fusion_policy_contract() -> dict:
    return {
        "trace_type": "wm_chart_fusion_policy_contract",
        "module": __name__,
        "opt_in": True,
        "supersedes_legacy_fusion_when_enabled": True,
        "gate_zero_is_hardcoded_prior": True,
        "qspin_live_routing": False,
        "shared_slot_writes": False,
        "testbed": True,
    }
