"""Native geometry-map charts: project tensors onto a named manifold and score with its metric.

Maps are schedules of chart names (euclidean, hyperbolic, spherical, ...). This
module turns those names into residual projections and pairwise distances.
Euclidean projection is identity. Mix 0 leaves tensors and distances unchanged.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from geometry.manifold_utils import (
    Geom,
    distance,
    geometry_name_to_geom,
    project_to_manifold,
)

# GeometryBlender extended mode order: hyperbolic, spherical, euclidean, fractal, torus, cp
_BLENDER_MODE_INDEX: Dict[str, int] = {
    "hyperbolic": 0,
    "poincare": 0,
    "curved": 0,
    "curved_associative": 0,
    "fiber_bundle": 0,
    "spherical": 1,
    "sphere": 1,
    "quaternion": 1,
    "dual_quaternion": 1,
    "euclidean": 2,
    "euclid": 2,
    "spatial_se3": 2,
    "tangent_bridge": 2,
    "subspace": 3,
    "grassmann": 3,
    "grassmannian": 3,
    "fractal": 3,
    "torus": 4,
    "product": 4,
    "complex": 5,
    "complex_projective": 5,
    "cp_kahler": 5,
    "cp": 5,
    "holographic_phase": 5,
    "spcp": 5,
}


def clamp_mix(mix: float) -> float:
    return float(max(0.0, min(1.0, mix)))


def resolve_chart_geom(geometry: str, x: torch.Tensor) -> Geom:
    """Pick a vector-safe chart. CP/Grassmann log/exp stay unused; Grassmann needs matrices."""
    geom = geometry_name_to_geom(str(geometry).strip().lower())
    feat = int(x.size(-1)) if x.dim() >= 1 else 0
    if geom == "grassmann":
        return "sphere"
    if geom == "cp" and feat != 2:
        return "sphere"
    if geom == "hyperbolic" and feat > 32:
        return "sphere"
    return geom


def majority_chart(charts: Optional[Sequence[str]], default: str = "euclidean") -> str:
    names = [str(name).strip().lower() for name in (charts or []) if str(name).strip()]
    if not names:
        return default
    return Counter(names).most_common(1)[0][0]


def fit_chart_list(charts: Sequence[str], count: int, fill: str = "euclidean") -> List[str]:
    vals = [str(name).strip().lower() or fill for name in charts]
    if len(vals) >= count:
        return vals[:count]
    return vals + [fill] * (count - len(vals))


def blender_priors_from_charts(charts: Sequence[str]) -> torch.Tensor:
    """Histogram prior over GeometryBlender's six extended modes."""
    counts = torch.zeros(6, dtype=torch.float32)
    names = list(charts) or ["euclidean"]
    for name in names:
        idx = _BLENDER_MODE_INDEX.get(str(name).strip().lower(), 2)
        counts[idx] += 1.0
    return counts / counts.sum().clamp_min(1.0)


def project_to_chart(
    x: torch.Tensor,
    geometry: str,
    *,
    kappa: float = -0.1,
) -> Tuple[torch.Tensor, Geom]:
    geom = resolve_chart_geom(geometry, x)
    projected = project_to_manifold(geom, x, kappa=kappa)
    if not torch.isfinite(projected).all():
        projected = torch.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)
    return projected, geom


def project_with_residual(
    x: torch.Tensor,
    geometry: str,
    mix: float = 1.0,
    *,
    kappa: float = -0.1,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Residual chart projection. Mix 0 is identity. Euclidean proj == x even at mix 1."""
    mix_v = clamp_mix(mix)
    geom = resolve_chart_geom(geometry, x)
    meta: Dict[str, Any] = {
        "geometry": str(geometry),
        "geom": geom,
        "mix": mix_v,
        "native_chart": True,
    }
    if mix_v <= 0.0:
        return x, meta
    projected, geom = project_to_chart(x, geometry, kappa=kappa)
    meta["geom"] = geom
    if mix_v >= 1.0:
        out = projected
    else:
        out = x + mix_v * (projected - x)
    if not torch.isfinite(out).all():
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out, meta


def pairwise_chart_distance(
    query: torch.Tensor,
    keys: torch.Tensor,
    geometry: str,
    *,
    kappa: float = -0.1,
) -> torch.Tensor:
    """Manifold distance. query [..., D], keys [K, D] or [..., K, D] -> [..., K]."""
    if keys.size(-1) != query.size(-1):
        raise ValueError("query/keys last dim must match for chart distance")
    q = query.unsqueeze(-2)
    k = keys
    q, k = torch.broadcast_tensors(q, k)
    geom = resolve_chart_geom(geometry, query)
    dist = distance(geom, q, k, kappa=kappa)
    if dist.dim() > 0 and dist.size(-1) == 1:
        dist = dist.squeeze(-1)
    return torch.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)


def pairwise_chart_affinity(
    query: torch.Tensor,
    keys: torch.Tensor,
    geometry: str,
    *,
    kappa: float = -0.1,
) -> torch.Tensor:
    """Higher is better, roughly cosine-scaled via tanh(-distance)."""
    dist = pairwise_chart_distance(query, keys, geometry, kappa=kappa)
    return torch.tanh(-dist)


def mix_chart_distance(
    query: torch.Tensor,
    keys: torch.Tensor,
    geometry: str,
    base_distance: torch.Tensor,
    mix: float,
    *,
    kappa: float = -0.1,
) -> torch.Tensor:
    mix_v = clamp_mix(mix)
    if mix_v <= 0.0 or keys.size(-1) != query.size(-1):
        return base_distance
    native = pairwise_chart_distance(query, keys, geometry, kappa=kappa)
    if native.shape != base_distance.shape:
        try:
            native = native.reshape_as(base_distance)
        except RuntimeError:
            return base_distance
    mixed = (1.0 - mix_v) * base_distance + mix_v * native
    return torch.nan_to_num(mixed, nan=0.0, posinf=0.0, neginf=0.0)


def mix_chart_affinity(
    cosine_scores: torch.Tensor,
    query: torch.Tensor,
    keys: torch.Tensor,
    geometry: str,
    mix: float,
    *,
    kappa: float = -0.1,
) -> torch.Tensor:
    mix_v = clamp_mix(mix)
    if mix_v <= 0.0:
        return cosine_scores
    geom = resolve_chart_geom(geometry, query)
    if geom == "euclid":
        return cosine_scores
    native = pairwise_chart_affinity(query, keys, geometry, kappa=kappa)
    if native.shape != cosine_scores.shape:
        if native.dim() == cosine_scores.dim() + 1 and native.size(-1) == 1:
            native = native.squeeze(-1)
        elif native.dim() == cosine_scores.dim() + 1 and native.size(-2) == 1:
            native = native.squeeze(-2)
        else:
            try:
                native = native.reshape_as(cosine_scores)
            except RuntimeError:
                return cosine_scores
    mixed = (1.0 - mix_v) * cosine_scores + mix_v * native
    return torch.nan_to_num(mixed, nan=0.0, posinf=0.0, neginf=0.0)


def configure_memory_bank_charts(
    bank: Any,
    charts: Sequence[str],
    mix: float = 0.15,
    *,
    blender_prior_mix: float = 0.25,
) -> None:
    """Attach chart list/mix to an LTM bank and nudge GeometryBlender priors."""
    chart_list = [str(name).strip().lower() for name in charts]
    bank.native_chart_list = chart_list
    bank.native_chart_mix = clamp_mix(mix)
    blender = getattr(bank, "geom_blender", None)
    if blender is not None and hasattr(blender, "set_mode_priors") and chart_list:
        blender.set_mode_priors(blender_priors_from_charts(chart_list), mix=float(blender_prior_mix))


def mix_bank_chart_distance(
    bank: Any,
    query: torch.Tensor,
    keys: torch.Tensor,
    base_distance: torch.Tensor,
) -> torch.Tensor:
    mix_v = clamp_mix(float(getattr(bank, "native_chart_mix", 0.0) or 0.0))
    charts = getattr(bank, "native_chart_list", None)
    if mix_v <= 0.0 or not charts:
        return base_distance
    return mix_chart_distance(query, keys, majority_chart(charts), base_distance, mix_v)
