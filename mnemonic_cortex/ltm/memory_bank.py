"""
Plain-language summary
----------------------
What this file is for: Long-term memory package module: memory bank.
How it fits in the system: Supports LTM banks, MANN/geometry helpers, or package wiring used with cortex LTM.
Status: ACTIVE / LEGACY depending on file
Important notes for non-coders: Some files are local copies or aliases; prefer top-level cortex + triple_hybrid for product runtime.

Technical notes (original):
Geometry-specific key bank over a shared Euclidean value store.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .manifold_utils import conformal_scale, metric_distance


GEOMETRY_CHOICES = ["euclid", "hyper", "sphere", "torus", "spatial", "curved"]


@dataclass
class BankReadTrace:
    geometry: str
    depth_slice: int
    indices: torch.Tensor
    weights: torch.Tensor
    distances: torch.Tensor
    selected_keys: torch.Tensor
    selected_values: torch.Tensor


class GeometryMemoryBank(nn.Module):
    """Per-depth geometry key bank.

    Keys are geometry-specific; values come from a shared value tensor. This is the
    core mechanism needed by both LTM-native reasoning and MANN slot attention.
    """

    geom_choices: List[str] = list(GEOMETRY_CHOICES)

    def __init__(self, slots: int, key_dim: int, default_geom: str, depth_slices: int = 8, conformal_b: float = 0.08, min_c: float = 1e-3, max_c: float = 1.0):
        super().__init__()
        if default_geom not in self.geom_choices:
            raise ValueError(f"unknown default geometry: {default_geom}")
        self.slots = int(slots)
        self.key_dim = int(key_dim)
        self.default_geom = default_geom
        self.depth_slices = int(depth_slices)
        self.min_c = float(min_c)
        self.max_c = float(max_c)
        self.keys = nn.Parameter(torch.randn(slots, key_dim) * 0.02)
        self.register_buffer("geom_per_slice", torch.zeros(depth_slices, dtype=torch.long))
        self.curv_per_slice = nn.Parameter(torch.ones(depth_slices) * 0.05)
        self.b_per_slice = nn.Parameter(torch.ones(depth_slices) * float(conformal_b))
        self.phi = nn.Sequential(nn.Linear(key_dim, 32), nn.GELU(), nn.Linear(32, 1), nn.Tanh())
        self.set_slice_geometries([default_geom] * depth_slices)

    @torch.no_grad()
    def set_slice_geometries(self, geom_names: Iterable[str]) -> None:
        names = list(geom_names)
        if len(names) != self.depth_slices:
            raise ValueError(f"expected {self.depth_slices} geometry names")
        idxs = []
        for g in names:
            idxs.append(self.geom_choices.index(g if g in self.geom_choices else self.default_geom))
        self.geom_per_slice.copy_(torch.tensor(idxs, device=self.geom_per_slice.device, dtype=torch.long))

    def _depth_idx(self, depth_slice: int | torch.Tensor) -> int:
        if torch.is_tensor(depth_slice):
            if depth_slice.numel() == 0:
                return 0
            return int(depth_slice.reshape(-1)[0].item()) % self.depth_slices
        return int(depth_slice) % self.depth_slices

    def _geom_name(self, depth_slice: int | torch.Tensor) -> str:
        ds = self._depth_idx(depth_slice)
        return self.geom_choices[int(self.geom_per_slice[ds].item())]

    def distances(self, q: torch.Tensor, depth_slice: int | torch.Tensor = 0, b_override: Optional[float] = None) -> Tuple[torch.Tensor, str, int]:
        ds = self._depth_idx(depth_slice)
        geom = self._geom_name(ds)
        c = self.curv_per_slice[ds].clamp(self.min_c, self.max_c)
        d = metric_distance(geom, q, self.keys, c=c if geom == "hyper" else None)
        b = float(b_override) if b_override is not None else float(self.b_per_slice[ds].clamp(0.0, 0.2).detach().cpu())
        omega = conformal_scale(self.phi(q), b=b)
        return d * omega, geom, ds

    def retrieve(self, q: torch.Tensor, shared_values: torch.Tensor, topk: int = 8, depth_slice: int | torch.Tensor = 0, b_override: Optional[float] = None):
        if shared_values.ndim != 2 or shared_values.size(0) != self.slots:
            raise ValueError("shared_values must be [slots,value_dim]")
        d, geom, ds = self.distances(q, depth_slice=depth_slice, b_override=b_override)
        k = min(int(topk), self.slots)
        idx = torch.topk(-d, k=k, dim=-1).indices
        sel_d = torch.gather(d, dim=1, index=idx)
        w = torch.softmax(-sel_d, dim=-1)
        sel_keys = self.keys[idx]
        sel_values = shared_values[idx]
        out = torch.sum(w.unsqueeze(-1) * sel_values, dim=1)
        trace = BankReadTrace(geometry=geom, depth_slice=ds, indices=idx, weights=w, distances=sel_d, selected_keys=sel_keys, selected_values=sel_values)
        return out, trace

    def snapshot(self) -> Dict[str, object]:
        return {
            "slots": self.slots,
            "key_dim": self.key_dim,
            "default_geom": self.default_geom,
            "depth_slices": self.depth_slices,
            "chart": [self._geom_name(i) for i in range(self.depth_slices)],
        }
