"""Stable geometry utilities for Spatial LTM and geometry-aware MANN.

The spatial channel is a compact SE(3)-inspired metric: translation distance,
unit-quaternion orientation distance, and optional residual feature distance.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

EPS = 1e-8


def _as_query_key(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if q.ndim != 2:
        raise ValueError("q must be [B,D]")
    if k.ndim not in (2, 3):
        raise ValueError("k must be [S,D] or [B,S,D]")
    if k.ndim == 2:
        k = k.unsqueeze(0).expand(q.size(0), -1, -1)
    if q.size(0) != k.size(0) or q.size(-1) != k.size(-1):
        raise ValueError("q and k batch/dim mismatch")
    return q, k


def wrap_angles(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def euclid_dist(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    q, k = _as_query_key(q, k)
    return torch.cdist(q.unsqueeze(1), k, p=2).squeeze(1)


def spherical_dist(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    q, k = _as_query_key(q, k)
    qn = F.normalize(q, dim=-1, eps=EPS)
    kn = F.normalize(k, dim=-1, eps=EPS)
    dot = (qn.unsqueeze(1) * kn).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(dot)


def torus_dist(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    q, k = _as_query_key(q, k)
    delta = wrap_angles(q.unsqueeze(1) - k)
    return torch.linalg.norm(delta, dim=-1)


def _poincare_project(x: torch.Tensor, c: float | torch.Tensor = 1.0) -> torch.Tensor:
    if not torch.is_tensor(c):
        c = torch.tensor(float(c), device=x.device, dtype=x.dtype)
    radius = (1.0 / torch.sqrt(c.clamp_min(EPS))) * (1.0 - 1e-5)
    norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(EPS)
    scale = torch.minimum(torch.ones_like(norm), radius / norm)
    return x * scale


def hyperbolic_dist(q: torch.Tensor, k: torch.Tensor, c: float | torch.Tensor = 1.0) -> torch.Tensor:
    q, k = _as_query_key(q, k)
    if not torch.is_tensor(c):
        c_t = torch.tensor(float(c), device=q.device, dtype=q.dtype)
    else:
        c_t = c.to(device=q.device, dtype=q.dtype)
    q = _poincare_project(q, c_t)
    k = _poincare_project(k, c_t)
    diff2 = ((q.unsqueeze(1) - k) ** 2).sum(dim=-1)
    q2 = (q ** 2).sum(dim=-1, keepdim=True)
    k2 = (k ** 2).sum(dim=-1)
    denom = (1.0 - c_t * q2).clamp_min(EPS) * (1.0 - c_t * k2).clamp_min(EPS)
    z = 1.0 + 2.0 * c_t * diff2 / denom
    return torch.acosh(z.clamp_min(1.0 + 1e-6)) / torch.sqrt(c_t.clamp_min(EPS))


def quaternion_normalize(q: torch.Tensor) -> torch.Tensor:
    if q.size(-1) != 4:
        raise ValueError("quaternion tensor must have last dimension 4")
    return F.normalize(q, dim=-1, eps=EPS)


def quaternion_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    q1 = quaternion_normalize(q1)
    q2 = quaternion_normalize(q2)
    dot = (q1 * q2).sum(dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


def spatial_dist(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """SE(3)-inspired distance.

    If D >= 7: first 3 dims are translation and next 4 dims are quaternion.
    Remaining dims are treated as residual landmark/affordance features.
    If D < 7: fallback to Euclidean distance.
    """
    q, k = _as_query_key(q, k)
    d = q.size(-1)
    if d < 7:
        return euclid_dist(q, k)
    pos_q, quat_q, rem_q = q[:, :3], q[:, 3:7], q[:, 7:]
    pos_k, quat_k, rem_k = k[:, :, :3], k[:, :, 3:7], k[:, :, 7:]
    d_pos = torch.linalg.norm(pos_q.unsqueeze(1) - pos_k, dim=-1)
    d_quat = quaternion_distance(quat_q.unsqueeze(1).expand_as(quat_k), quat_k)
    if rem_q.numel() == 0:
        d_rem = torch.zeros_like(d_pos)
    else:
        d_rem = torch.linalg.norm(rem_q.unsqueeze(1) - rem_k, dim=-1) / math.sqrt(max(1, rem_q.size(-1)))
    return d_pos + 0.5 * d_quat + 0.25 * d_rem


def conformal_scale(phi: torch.Tensor, b: float = 0.08) -> torch.Tensor:
    return torch.exp(float(b) * torch.tanh(phi))


def metric_distance(name: str, q: torch.Tensor, k: torch.Tensor, c: Optional[float | torch.Tensor] = None) -> torch.Tensor:
    name = name.lower()
    if name == "euclid":
        return euclid_dist(q, k)
    if name == "sphere":
        return spherical_dist(q, k)
    if name == "torus":
        return torus_dist(q, k)
    if name == "hyper":
        return hyperbolic_dist(q, k, 1.0 if c is None else c)
    if name == "spatial":
        return spatial_dist(q, k)
    raise ValueError(f"unknown geometry: {name}")
