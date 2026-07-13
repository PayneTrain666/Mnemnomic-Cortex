"""
Plain-language summary
----------------------
What this file is for: Shared-slot memory subsystem module: conformal.
How it fits in the system: Manages shared memory slots that multiple systems can read/write under rules.
Status: OPT-IN
Important notes for non-coders: Not always enabled in standard capacity profiles.
"""

import torch
import torch.nn as nn

from geometry.manifold_utils import conformal_scale, gather_curvature, warp_distances


class ConformalMLP(nn.Module):
    """Tiny conformal lens scorer phi(query)."""

    def __init__(self, d_in: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@torch.no_grad()
def _ensure_shapes(distances: torch.Tensor, indices: torch.Tensor, query_vec: torch.Tensor):
    if distances.ndim != 2 or indices.ndim != 2:
        raise ValueError("distances and indices must be (N,k)")
    if query_vec.ndim != 2 or query_vec.size(0) != distances.size(0):
        raise ValueError("query_vec must be (N,D) with same N as distances")


def warp_knn(
    distances: torch.Tensor,
    indices: torch.Tensor,
    query_vec: torch.Tensor,
    curv_per_slot: torch.Tensor,
    conformal_mlp: ConformalMLP,
    b: float = 0.1,
) -> torch.Tensor:
    _ensure_shapes(distances, indices, query_vec)
    phi = conformal_mlp(query_vec).unsqueeze(-1).expand_as(distances)
    omega = conformal_scale(phi, b=b)
    csel = gather_curvature(curv_per_slot, indices)
    return warp_distances(distances, curvature_idx=csel, conf_scale=omega)


def warp_knn_with_stats(
    distances: torch.Tensor,
    indices: torch.Tensor,
    query_vec: torch.Tensor,
    curv_per_slot: torch.Tensor,
    conformal_mlp: ConformalMLP,
    b: float = 0.1,
):
    _ensure_shapes(distances, indices, query_vec)
    phi = conformal_mlp(query_vec).unsqueeze(-1).expand_as(distances)
    omega = conformal_scale(phi, b=b)
    csel = gather_curvature(curv_per_slot, indices)
    warped = warp_distances(distances, curvature_idx=csel, conf_scale=omega)
    aux = {
        "omega_mean": omega.mean(dim=-1).detach(),
        "curv_mean": csel.mean(dim=-1).detach(),
    }
    return warped, aux

