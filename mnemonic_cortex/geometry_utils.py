"""
Plain-language summary
----------------------
What this file is for: Low-level geometry helpers (quaternions, angles, etc.).
How it fits in the system: Math toolkit used by topology and manifold code.
Status: WORKING
Important notes for non-coders: Foundation utilities; not a full memory system.
"""

import math
import torch
import torch.nn as nn

def qnormalize(q, eps=1e-9):
    return q / (q.norm(dim=-1, keepdim=True) + eps)

def qmul(q1, q2):
    # (w,x,y,z) Hamilton product
    w1,x1,y1,z1 = q1.unbind(-1)
    w2,x2,y2,z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w,x,y,z], dim=-1)

def qexp(omega):
    # omega: [..., 3], exp : unit quaternion
    theta = omega.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    half = 0.5 * theta
    v = (torch.sin(half)/theta) * omega
    return qnormalize(torch.cat([torch.cos(half), v], dim=-1))

def qangle(q1, q2, eps=1e-7):
    q1 = qnormalize(q1); q2 = qnormalize(q2)
    dot = (q1*q2).sum(-1).abs().clamp(0.0, 1.0 - eps)
    return 2.0 * torch.acos(dot)  # radians in [0, pi]

# ===== metric warps (monotone, stable) =====
def _d_euc(d):  return d
def _d_hyp(d, slot_c, alpha=0.6):
    return torch.asinh((1.0 + alpha*torch.tanh(slot_c)) * d)
def _d_sph(d, gamma=0.9):
    return (2.0 * torch.sin(0.5 * gamma * d)).abs()


def lorentz_lift(x, eps=1e-6):
    # x:[...,D] -> X:[...,D+1] on H^D
    x_norm2 = (x * x).sum(dim=-1, keepdim=True)
    x0 = torch.sqrt(1.0 + x_norm2 + eps)
    return torch.cat([x0, x], dim=-1)


def mink_dot(x, y):
    # Lorentzian dot, time-like 0
    return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(-1)


def lorentz_dist(x, y, eps=1e-6):
    z = (-mink_dot(x, y)).clamp_min(1.0 + eps)
    return torch.acosh(z)


def sph_warp(d, gamma=0.9):
    return (2.0 * torch.sin(0.5 * gamma * d)).abs()


def fubini_study_batched(psi, phi):
    # psi:[B,d] complex; phi:[B,K,d] complex -> [B,K]
    # Avoid torch.nan_to_num on complex (no autograd support); sanitize via real/imag.
    def _finite_complex(z: torch.Tensor) -> torch.Tensor:
        r = torch.where(torch.isfinite(z.real), z.real, torch.zeros_like(z.real))
        i = torch.where(torch.isfinite(z.imag), z.imag, torch.zeros_like(z.imag))
        return torch.complex(r, i)

    psi = _finite_complex(psi)
    phi = _finite_complex(phi)
    psi = psi / (psi.abs().pow(2).sum(-1, keepdim=True).sqrt() + 1e-9)
    phi = phi / (phi.abs().pow(2).sum(-1, keepdim=True).sqrt() + 1e-9)
    ip = (psi.unsqueeze(1).conj() * phi).sum(-1).abs().clamp(0.0, 1.0 - 1e-6)
    return torch.arccos(ip)

class HolonomyProbe(nn.Module):
    """Diagnostics-only holonomy estimator over tiny latent loops."""

    def __init__(self, spin_conn_fn):
        super().__init__()
        self.spin_conn_fn = spin_conn_fn

    @torch.no_grad()
    def forward(self, manifold_patch, steps=3, dt=1.0):
        # manifold_patch: [B, D, 3]
        bsz = manifold_patch.size(0)
        q = qnormalize(
            torch.cat(
                [
                    torch.ones(
                        bsz, 1, device=manifold_patch.device, dtype=manifold_patch.dtype
                    ),
                    torch.zeros(
                        bsz, 3, device=manifold_patch.device, dtype=manifold_patch.dtype
                    ),
                ],
                dim=-1,
            )
        )
        for _ in range(steps):
            for sgn in (+1.0, -1.0, +1.0, -1.0):
                omega = self.spin_conn_fn(manifold_patch) * sgn
                dq = qexp(dt * omega)
                q = qmul(q, dq)
        ref = torch.tensor([1.0, 0.0, 0.0, 0.0], device=q.device, dtype=q.dtype).expand_as(q)
        ang = qangle(q, ref)
        return {
            "mean_holonomy_deg": (ang.mean() * 180.0 / math.pi).item(),
            "p95_holonomy_deg": (torch.quantile(ang, 0.95) * 180.0 / math.pi).item(),
        }