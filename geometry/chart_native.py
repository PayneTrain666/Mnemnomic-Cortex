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
    exp_map,
    geometry_name_to_geom,
    log_map,
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


def chart_origin(geom: Geom, like: torch.Tensor) -> torch.Tensor:
    """Canonical basepoint used to identify each chart's tangent space with R^D."""
    origin = torch.zeros_like(like)
    if geom == "sphere":
        origin[..., 0] = 1.0
    return origin


def tangent_at_origin(
    x: torch.Tensor,
    geometry: str,
    *,
    kappa: float = -0.1,
) -> torch.Tensor:
    """Log-map x to the tangent space at the chart origin. Euclidean is identity."""
    projected, geom = project_to_chart(x, geometry, kappa=kappa)
    origin = chart_origin(geom, projected)
    mapped = log_map(geom, origin, projected, kappa=kappa)
    if mapped is NotImplemented:
        projected, geom = project_to_chart(x, "spherical", kappa=kappa)
        origin = chart_origin(geom, projected)
        mapped = log_map(geom, origin, projected, kappa=kappa)
    if not torch.is_tensor(mapped):
        mapped = projected
    if not torch.isfinite(mapped).all():
        mapped = torch.nan_to_num(mapped, nan=0.0, posinf=0.0, neginf=0.0)
    return mapped


def retract_from_origin(
    tangent: torch.Tensor,
    geometry: str,
    *,
    kappa: float = -0.1,
) -> torch.Tensor:
    """Exp-map a tangent vector at the chart origin back onto the manifold."""
    geom = resolve_chart_geom(geometry, tangent)
    origin = chart_origin(geom, tangent)
    out = exp_map(geom, origin, tangent, kappa=kappa)
    if out is NotImplemented:
        geom = "sphere"
        origin = chart_origin(geom, tangent)
        out = exp_map(geom, origin, tangent, kappa=kappa)
    if not torch.is_tensor(out):
        out = tangent
    if not torch.isfinite(out).all():
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def batched_tangent_at_origin(
    tokens: torch.Tensor,
    charts: Sequence[str],
    *,
    kappa: float = -0.1,
) -> torch.Tensor:
    """tokens [B, M, D] -> tangents [B, M, D]."""
    if tokens.dim() != 3:
        raise ValueError(f"expected tokens [B,M,D], got {tuple(tokens.shape)}")
    names = fit_chart_list(charts, int(tokens.size(1)))
    parts = [tangent_at_origin(tokens[:, idx], names[idx], kappa=kappa) for idx in range(int(tokens.size(1)))]
    return torch.stack(parts, dim=1)


def batched_retract_from_origin(
    tangents: torch.Tensor,
    charts: Sequence[str],
    *,
    kappa: float = -0.1,
) -> torch.Tensor:
    """tangents [B, M, D] -> manifold points [B, M, D]."""
    if tangents.dim() != 3:
        raise ValueError(f"expected tangents [B,M,D], got {tuple(tangents.shape)}")
    names = fit_chart_list(charts, int(tangents.size(1)))
    parts = [retract_from_origin(tangents[:, idx], names[idx], kappa=kappa) for idx in range(int(tangents.size(1)))]
    return torch.stack(parts, dim=1)


def manifold_pairwise_affinity(
    tokens: torch.Tensor,
    charts: Sequence[str],
    *,
    kappa: float = -0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chart-aware affinity [B,M,M] plus shared-origin tangents [B,M,D].

    Same resolved chart uses geodesic tanh(-distance). Different charts compare
    log-maps at the origin so Poincaré/spherical points are not added in ambient space.
    """
    if tokens.dim() != 3:
        raise ValueError(f"expected tokens [B,M,D], got {tuple(tokens.shape)}")
    names = fit_chart_list(charts, int(tokens.size(1)))
    tangents = batched_tangent_at_origin(tokens, names, kappa=kappa)
    diff = tangents.unsqueeze(2) - tangents.unsqueeze(1)
    scores = torch.tanh(-diff.norm(dim=-1).clamp_min(1e-6))
    geoms = [resolve_chart_geom(name, tokens[:, 0]) for name in names]
    for geom in set(geoms):
        idx = [i for i, g in enumerate(geoms) if g == geom]
        if len(idx) < 2:
            continue
        idx_t = torch.tensor(idx, device=tokens.device, dtype=torch.long)
        sub = tokens.index_select(1, idx_t)
        dist = distance(geom, sub.unsqueeze(2), sub.unsqueeze(1), kappa=kappa)
        if dist.dim() > 0 and dist.size(-1) == 1:
            dist = dist.squeeze(-1)
        aff = torch.tanh(-torch.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0))
        scores[:, idx_t.unsqueeze(1), idx_t.unsqueeze(0)] = aff
    return torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0), tangents


def query_key_manifold_affinity(
    query: torch.Tensor,
    keys: torch.Tensor,
    query_chart: str,
    key_charts: Optional[Sequence[str]] = None,
    *,
    kappa: float = -0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cross-attention chart scores [B,K] plus query/key tangents at each origin.

    Same resolved chart uses geodesic tanh(-distance). Different charts compare
    log-maps at the origin so Poincaré/spherical keys are not dotted in ambient space.
    """
    if query.dim() != 2 or keys.dim() != 3:
        raise ValueError(
            f"expected query [B,D] and keys [B,K,D], got {tuple(query.shape)} and {tuple(keys.shape)}"
        )
    if query.size(0) != keys.size(0) or query.size(-1) != keys.size(-1):
        raise ValueError("query/keys batch and last dim must match")
    names = fit_chart_list(key_charts or [query_chart], int(keys.size(1)))
    q_tan = tangent_at_origin(query, query_chart, kappa=kappa)
    k_tan = batched_tangent_at_origin(keys, names, kappa=kappa)
    diff = q_tan.unsqueeze(1) - k_tan
    scores = torch.tanh(-diff.norm(dim=-1).clamp_min(1e-6))
    q_geom = resolve_chart_geom(query_chart, query)
    key_geoms = [resolve_chart_geom(name, keys[:, 0]) for name in names]
    same_idx = [i for i, geom in enumerate(key_geoms) if geom == q_geom]
    if same_idx:
        idx_t = torch.tensor(same_idx, device=query.device, dtype=torch.long)
        sub = keys.index_select(1, idx_t)
        aff = pairwise_chart_affinity(query, sub, query_chart, kappa=kappa)
        scores[:, idx_t] = aff
    return torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0), q_tan, k_tan


def transport_query_key_messages(
    weights: torch.Tensor,
    key_tangents: torch.Tensor,
    query_chart: str,
    *,
    kappa: float = -0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Weighted tangent combination of keys, then exp-map onto the query chart."""
    if weights.dim() != 2 or key_tangents.dim() != 3:
        raise ValueError(
            f"expected weights [B,K] and key_tangents [B,K,D], got {tuple(weights.shape)} and {tuple(key_tangents.shape)}"
        )
    if weights.shape[:2] != key_tangents.shape[:2]:
        raise ValueError("weights [B,K] must match key_tangents [B,K]")
    message = torch.einsum("bk,bkd->bd", weights, key_tangents)
    if not torch.isfinite(message).all():
        message = torch.nan_to_num(message, nan=0.0, posinf=0.0, neginf=0.0)
    retracted = retract_from_origin(message, query_chart, kappa=kappa)
    return message, retracted
