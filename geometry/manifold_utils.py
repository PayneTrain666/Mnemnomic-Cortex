from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F

EPS = 1e-6


def conformal_scale(phi: torch.Tensor, b: float = 0.1) -> torch.Tensor:
    """Tiny conformal magnification Ω(x) = 1 + b * tanh(phi(x)). Keep b <= 0.2."""
    b = float(max(0.0, min(0.2, b)))
    raw = 1.0 + b * torch.tanh(phi)
    return raw.clamp(1.0 - b, 1.0 + b).clamp_min(1e-4)


def safe_norm(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    return torch.linalg.norm(x, ord=2, dim=dim, keepdim=keepdim).clamp_min(EPS)


def euc_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y).norm(dim=-1, keepdim=True).clamp_min(EPS)


def euc_log_map(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return y - x


def euc_exp_map(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return x + v


def euc_project(x: torch.Tensor) -> torch.Tensor:
    return x


def poincare_proj(x: torch.Tensor, c: float, eps: float = 1e-5) -> torch.Tensor:
    sqrt_c = torch.tensor(c, dtype=x.dtype, device=x.device).sqrt()
    max_norm = (1.0 - eps) / sqrt_c
    norm = safe_norm(x, dim=-1, keepdim=True)
    factor = torch.where(norm < max_norm, torch.ones_like(norm), max_norm / norm)
    return x * factor


def poincare_project_ball(x: torch.Tensor, max_norm: float = 0.9) -> torch.Tensor:
    n = safe_norm(x, dim=-1, keepdim=True)
    scale = (max_norm / n).clamp_max(1.0)
    return x * scale


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + c * c * x2 * y2
    return num / den.clamp_min(EPS)


def poincare_exp_map(x: torch.Tensor, v: torch.Tensor, c: float) -> torch.Tensor:
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(EPS)
    sqrt_c = torch.tensor(c, dtype=x.dtype, device=x.device).sqrt()
    coef = torch.tanh(sqrt_c * v_norm / 2.0) * v / (sqrt_c * v_norm)
    return poincare_proj(mobius_add(x, coef, c), c)


def poincare_log_map(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    sqrt_c = torch.tensor(c, dtype=x.dtype, device=x.device).sqrt()
    diff = mobius_add(-x, y, c)
    d = diff.norm(dim=-1, keepdim=True).clamp_min(EPS)
    factor = (2.0 / sqrt_c) * torch.atanh((sqrt_c * d).clamp(max=1 - 1e-6))
    return diff * (factor / d)


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    sqrt_c = torch.tensor(c, dtype=x.dtype, device=x.device).sqrt()
    diff = mobius_add(-x, y, c)
    d = diff.norm(dim=-1, keepdim=True).clamp_min(EPS)
    arg = (sqrt_c * d).clamp(max=1 - 1e-6)
    return (2.0 / sqrt_c) * torch.atanh(arg).clamp_min(EPS)


def sphere_project(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def sphere_exp_map(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    v_par = v - (v * x).sum(dim=-1, keepdim=True) * x
    th = v_par.norm(dim=-1, keepdim=True).clamp_min(EPS)
    y = x * torch.cos(th) + v_par * torch.sin(th) / th
    return sphere_project(y)


def sphere_log_map(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = sphere_project(x)
    y = sphere_project(y)
    dot = (x * y).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    th = torch.arccos(dot).clamp_min(EPS)
    v = y - dot * x
    return th * v / v.norm(dim=-1, keepdim=True).clamp_min(EPS)


def sphere_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = sphere_project(x)
    y = sphere_project(y)
    dot = (x * y).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.arccos(dot).clamp_min(EPS)


def wrap_angles(theta: torch.Tensor) -> torch.Tensor:
    return (theta + torch.pi) % (2 * torch.pi) - torch.pi


def torus_add(theta: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return wrap_angles(theta + delta)


def torus_log_map(theta_x: torch.Tensor, theta_y: torch.Tensor) -> torch.Tensor:
    return wrap_angles(theta_y - theta_x)


def torus_exp_map(theta_x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return wrap_angles(theta_x + delta)


def torus_distance(
    theta_x: torch.Tensor,
    theta_y: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    *,
    keepdim: bool = True,
) -> torch.Tensor:
    d = wrap_angles(theta_x - theta_y)
    if weights is not None:
        d = d * weights
    return safe_norm(d, dim=-1, keepdim=keepdim)


def torus_sincos_embed(theta: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)


def product_et_distance(
    q_e: torch.Tensor,
    q_t: torch.Tensor,
    k_e: torch.Tensor,
    k_t: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Combined distance on E×T: sqrt( ||q_e-k_e||^2 + alpha * ||Δ_torus||^2 )
    Shapes:
      q_e: (B,De), q_t: (B,Kt), k_e: (B,K,De), k_t: (B,K,Kt) -> (B,K)
    """
    de = safe_norm(q_e.unsqueeze(1) - k_e, dim=-1)
    dt = torus_distance(q_t.unsqueeze(1).expand_as(k_t), k_t, keepdim=False)
    return torch.sqrt((de ** 2) + float(alpha) * (dt ** 2)).clamp_min(EPS)


def symplectic_leapfrog(
    q: torch.Tensor,
    p: torch.Tensor,
    dH_dq: torch.Tensor,
    dH_dp: torch.Tensor,
    step: float = 1e-2,
):
    """
    Tiny stable Hamiltonian update used as optional WM patch.
    """
    h = float(step)
    p_half = p - 0.5 * h * dH_dq
    q_new = q + h * dH_dp
    p_new = p_half - 0.5 * h * dH_dq
    return q_new, p_new


def complex_normalize(z: torch.Tensor) -> torch.Tensor:
    re, im = z[..., 0], z[..., 1]
    nrm = torch.sqrt(re.pow(2) + im.pow(2)).sum(dim=-1, keepdim=True).clamp_min(EPS)
    re = re / nrm
    im = im / nrm
    return torch.stack([re, im], dim=-1)


def complex_inner_prod(z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    zr, zi = z[..., 0], z[..., 1]
    wr, wi = w[..., 0], w[..., 1]
    re = zr * wr + zi * wi
    im = -zr * wi + zi * wr
    return torch.stack([re, im], dim=-1)


def normalize_complex(z_re: torch.Tensor, z_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    norm = torch.sqrt((z_re ** 2 + z_im ** 2).sum(dim=-1, keepdim=True)).clamp_min(1e-8)
    return z_re / norm, z_im / norm


def fubini_study_distance(
    z1_re: torch.Tensor,
    z1_im: torch.Tensor,
    z2_re: torch.Tensor,
    z2_im: torch.Tensor,
) -> torch.Tensor:
    ip_re = (z1_re * z2_re + z1_im * z2_im).sum(dim=-1)
    ip_im = (z1_im * z2_re - z1_re * z2_im).sum(dim=-1)
    ip_abs = torch.sqrt(ip_re ** 2 + ip_im ** 2).clamp(0, 1.0)
    return torch.acos(ip_abs.clamp(0, 1.0 - 1e-6))


def cp_distance(z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    z = complex_normalize(z)
    w = complex_normalize(w)
    inner = complex_inner_prod(z, w)
    mag = torch.sqrt(inner[..., 0].pow(2) + inner[..., 1].pow(2)).clamp(0.0, 1.0)
    return torch.arccos(mag).unsqueeze(-1).clamp_min(EPS)


def grassmann_project(u: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(u, mode="reduced")
    return q


def grassmann_principal_angles(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    utv = torch.matmul(u.transpose(-2, -1), v)
    _, s, _ = torch.linalg.svd(utv)
    return torch.arccos(s.clamp(0, 1))


def grassmann_distance(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    theta = grassmann_principal_angles(u, v)
    return torch.norm(theta, dim=-1, keepdim=True).clamp_min(EPS)


Geom = Literal["euclid", "hyperbolic", "sphere", "torus", "cp", "grassmann"]

GEOMETRY_NAME_TO_GEOM: dict[str, Geom] = {
    "curved": "hyperbolic",
    "curved_associative": "hyperbolic",
    "euclidean": "euclid",
    "euclid": "euclid",
    "hyperbolic": "hyperbolic",
    "poincare": "hyperbolic",
    "spherical": "sphere",
    "sphere": "sphere",
    "torus": "torus",
    "complex": "cp",
    "complex_projective": "cp",
    "cp_kahler": "cp",
    "cp": "cp",
    "grassmann": "grassmann",
    "grassmannian": "grassmann",
    "subspace": "grassmann",
    "product": "torus",
    "holographic_phase": "cp",
    "quaternion": "sphere",
    "spatial_se3": "euclid",
    "spcp": "cp",
    "fiber_bundle": "hyperbolic",
    "tangent_bridge": "euclid",
    "dual_quaternion": "sphere",
}


def geometry_name_to_geom(name: str) -> Geom:
    return GEOMETRY_NAME_TO_GEOM.get(str(name).lower(), "euclid")


def project_to_manifold(geom: Geom, x: torch.Tensor, **kwargs) -> torch.Tensor:
    if geom == "euclid":
        return euc_project(x)
    if geom == "hyperbolic":
        kappa = kwargs.get("kappa", -0.1)
        c = float(abs(min(kappa, -EPS)))
        return poincare_proj(x, c)
    if geom == "sphere":
        return sphere_project(x)
    if geom == "torus":
        return wrap_angles(x)
    if geom == "cp":
        return complex_normalize(x)
    if geom == "grassmann":
        return grassmann_project(x)
    raise ValueError(f"Unknown geom: {geom}")


def log_map(geom: Geom, x: torch.Tensor, y: torch.Tensor, **kwargs):
    if geom == "euclid":
        return euc_log_map(x, y)
    if geom == "hyperbolic":
        kappa = kwargs.get("kappa", -0.1)
        c = float(abs(min(kappa, -EPS)))
        return poincare_log_map(poincare_proj(x, c), poincare_proj(y, c), c)
    if geom == "sphere":
        return sphere_log_map(sphere_project(x), sphere_project(y))
    if geom == "torus":
        return torus_log_map(x, y)
    if geom in ("cp", "grassmann"):
        return NotImplemented
    raise ValueError(f"Unknown geom: {geom}")


def exp_map(geom: Geom, x: torch.Tensor, v: torch.Tensor, **kwargs):
    if geom == "euclid":
        return euc_exp_map(x, v)
    if geom == "hyperbolic":
        kappa = kwargs.get("kappa", -0.1)
        c = float(abs(min(kappa, -EPS)))
        return poincare_exp_map(poincare_proj(x, c), v, c)
    if geom == "sphere":
        return sphere_exp_map(sphere_project(x), v)
    if geom == "torus":
        return torus_exp_map(x, v)
    if geom in ("cp", "grassmann"):
        return NotImplemented
    raise ValueError(f"Unknown geom: {geom}")


def distance(geom: Geom, x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
    if geom == "euclid":
        return euc_distance(x, y)
    if geom == "hyperbolic":
        kappa = kwargs.get("kappa", -0.1)
        c = float(abs(min(kappa, -EPS)))
        return poincare_distance(poincare_proj(x, c), poincare_proj(y, c), c)
    if geom == "sphere":
        return sphere_distance(x, y)
    if geom == "torus":
        return torus_distance(x, y)
    if geom == "cp":
        return cp_distance(x, y)
    if geom == "grassmann":
        return grassmann_distance(grassmann_project(x), grassmann_project(y))
    raise ValueError(f"Unknown geom: {geom}")


def retract_sphere(x: torch.Tensor) -> torch.Tensor:
    return sphere_project(x)


def retract_poincare(x: torch.Tensor, max_norm: float = 0.98) -> torch.Tensor:
    return poincare_project_ball(x, max_norm=max_norm)


def retract_torus(theta: torch.Tensor) -> torch.Tensor:
    return wrap_angles(theta)


def retract_unit_complex(z_re: torch.Tensor, z_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return normalize_complex(z_re, z_im)


def proj_to_sphere(x: torch.Tensor) -> torch.Tensor:
    return sphere_project(x)


def retract_to_manifold(geom: Geom, x: torch.Tensor, **kwargs) -> torch.Tensor:
    return project_to_manifold(geom, x, **kwargs)


def _chart_geom_for_tensor(geom: Geom, x: torch.Tensor) -> Geom:
    """Pick a slot-vector-safe chart when the symbolic geometry needs higher rank."""
    if x.dim() == 2 and geom in ("cp", "grassmann"):
        return "sphere"
    if x.dim() == 2 and geom == "hyperbolic" and x.size(-1) > 32:
        return "sphere"
    return geom


def chart_transform(
    x: torch.Tensor,
    *,
    geometry: str,
    depth_index: int = 0,
    gain: float = 1.0,
    kappa: float = -0.1,
) -> Tuple[torch.Tensor, dict]:
    """Project onto the named geometry chart, then apply a gentle depth gain."""
    geom = _chart_geom_for_tensor(geometry_name_to_geom(geometry), x)
    depth_scale = 1.0 + 0.04 * float(depth_index + 1)
    projected = project_to_manifold(geom, x, kappa=kappa)
    scaled = torch.tanh(projected * (gain * depth_scale)) + 0.1 * torch.sin(projected * depth_scale)
    retracted = retract_to_manifold(geom, scaled, kappa=kappa)
    return retracted, {
        "geometry": geometry,
        "geom": geom,
        "depth_index": int(depth_index),
        "gain": float(gain),
        "depth_scale": float(depth_scale),
    }


@torch.no_grad()
def frechet_mean(geom: Geom, points: torch.Tensor, iters: int = 15, step: float = 1.0, **kwargs):
    if points.ndim < 2:
        raise ValueError("points must be at least (N, D...)")
    if geom == "euclid":
        return points.mean(dim=0)
    if geom == "torus":
        s = torch.sin(points).mean(dim=0)
        c = torch.cos(points).mean(dim=0)
        return torch.atan2(s, c)
    if geom == "sphere":
        m = sphere_project(points.mean(dim=0))
        for _ in range(iters):
            v = sphere_log_map(m, points).mean(dim=0)
            m = sphere_exp_map(m, step * v)
        return sphere_project(m)
    if geom == "hyperbolic":
        kappa = kwargs.get("kappa", -0.1)
        c = float(abs(min(kappa, -EPS)))
        m = poincare_proj(points.mean(dim=0), c)
        for _ in range(iters):
            v = poincare_log_map(m, poincare_proj(points, c), c).mean(dim=0)
            m = poincare_exp_map(m, step * v, c)
        return poincare_proj(m, c)
    raise ValueError(f"frechet_mean not implemented for {geom}")


def gather_curvature(curv_per_slot: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """curv_per_slot: (M,), indices: (B,K) -> (B,K)"""
    curv_sel = curv_per_slot[indices.view(-1)].view(*indices.shape)
    return torch.clamp(curv_sel, -1.0, 1.0)


def warp_distances(
    distances: torch.Tensor,
    curvature_idx: Optional[torch.Tensor] = None,
    conf_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Stable quadratic curvature warp + conformal magnification:
    d' = Ω * [ d * (1 + c * d^2) ]
    """
    d = distances.clamp_min(EPS)
    if curvature_idx is not None:
        c = torch.clamp(curvature_idx, -1.0, 1.0)
        d = d * (1.0 + c * (d ** 2))
    if conf_scale is not None:
        d = conf_scale * d
    return d.clamp_min(1e-7)

