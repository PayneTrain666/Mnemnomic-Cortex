from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


def _safe_norm(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    return torch.linalg.norm(x, ord=2, dim=dim, keepdim=keepdim).clamp_min(EPS)


def hyperbolic_project_ball(x: torch.Tensor, max_norm: float = 0.9) -> torch.Tensor:
    n = _safe_norm(x, dim=-1, keepdim=True)
    scale = (max_norm / n).clamp_max(1.0)
    return x * scale


def hyperbolic_distance_poincare(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    diff2 = ((x - y) * (x - y)).sum(dim=-1).clamp_min(EPS)
    denom = ((1.0 - x2).clamp_min(1e-6) * (1.0 - y2).clamp_min(1e-6)).squeeze(-1)
    z = 1.0 + 2.0 * diff2 / denom.clamp_min(1e-6)
    return torch.acosh(z.clamp_min(1.0 + 1e-6))


def spherical_project_unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def spherical_distance(x: torch.Tensor, y: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    xy = (x * y).sum(dim=-1)
    cosang = xy.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return radius * torch.acos(cosang)


def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _safe_norm(x - y, dim=-1)


def _multi_scale_pool(x: torch.Tensor, n_scales: int = 3):
    outs = [x]
    for s in range(1, n_scales):
        k = 2 ** s
        d = x.size(-1)
        if d // k < 1:
            outs.append(x)
            continue
        new_d = (d // k) * k
        x_cut = x[..., :new_d]
        xv = x_cut.view(*x_cut.shape[:-1], new_d // k, k).mean(dim=-1)
        pad = torch.zeros(*xv.shape[:-1], d - xv.size(-1), device=x.device, dtype=x.dtype)
        outs.append(torch.cat([xv, pad], dim=-1))
    return outs


def fractal_distance_ms(x: torch.Tensor, y: torch.Tensor, n_scales: int = 3) -> torch.Tensor:
    xs = _multi_scale_pool(x, n_scales=n_scales)
    ys = _multi_scale_pool(y, n_scales=n_scales)
    d = 0.0
    wsum = 0.0
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        w = 1.0 / (2.0 ** i)
        d = d + w * euclidean_distance(xi, yi)
        wsum += w
    return d / max(wsum, 1e-8)


class EuclideanHead(nn.Module):
    def __init__(self, d_in: int, d_metric: int = 64):
        super().__init__()
        self.proj = nn.Linear(d_in, d_metric, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SphericalHead(nn.Module):
    def __init__(self, d_in: int, d_metric: int = 64):
        super().__init__()
        self.proj = nn.Linear(d_in, d_metric, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return spherical_project_unit(self.proj(x))


class HyperbolicHead(nn.Module):
    def __init__(self, d_in: int, d_metric: int = 64, max_norm: float = 0.9):
        super().__init__()
        self.proj = nn.Linear(d_in, d_metric, bias=False)
        self.max_norm = float(max_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return hyperbolic_project_ball(self.proj(x), max_norm=self.max_norm)


class FractalHead(nn.Module):
    def __init__(self, d_in: int, d_metric: int = 64):
        super().__init__()
        self.proj = nn.Linear(d_in, d_metric, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class GeometryMetric(nn.Module):
    """
    Geometry-aware top-k re-scoring.
    """

    def __init__(self, d_query: int, d_key: int, d_metric: int = 64):
        super().__init__()
        self.q_e = EuclideanHead(d_query, d_metric)
        self.q_s = SphericalHead(d_query, d_metric)
        self.q_h = HyperbolicHead(d_query, d_metric)
        self.q_f = FractalHead(d_query, d_metric)

        self.k_e = EuclideanHead(d_key, d_metric)
        self.k_s = SphericalHead(d_key, d_metric)
        self.k_h = HyperbolicHead(d_key, d_metric)
        self.k_f = FractalHead(d_key, d_metric)

    def distances(self, q: torch.Tensor, k: torch.Tensor, mode_weights: torch.Tensor) -> torch.Tensor:
        # q: (B,Dq), k: (B,K,Dk), mode_weights: (4,) or (B,4)
        qe, qs, qh, qf = self.q_e(q), self.q_s(q), self.q_h(q), self.q_f(q)
        ke, ks, kh, kf = self.k_e(k), self.k_s(k), self.k_h(k), self.k_f(k)

        qe = qe.unsqueeze(1).expand_as(ke)
        qs = qs.unsqueeze(1).expand_as(ks)
        qh = qh.unsqueeze(1).expand_as(kh)
        qf = qf.unsqueeze(1).expand_as(kf)

        de = euclidean_distance(qe, ke).clamp_min(EPS)
        ds = spherical_distance(qs, ks).clamp_min(EPS)
        dh = hyperbolic_distance_poincare(qh, kh).clamp_min(EPS)
        df = fractal_distance_ms(qf, kf).clamp_min(EPS)

        if mode_weights.ndim == 1:
            w = mode_weights
            return w[0] * dh + w[1] * ds + w[2] * de + w[3] * df
        w = mode_weights
        return (
            w[:, 0].unsqueeze(-1) * dh
            + w[:, 1].unsqueeze(-1) * ds
            + w[:, 2].unsqueeze(-1) * de
            + w[:, 3].unsqueeze(-1) * df
        )

