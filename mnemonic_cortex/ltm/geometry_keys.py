"""Geometry key projection and depth routing."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthRouter(nn.Module):
    """Predict phase/scale/spin bins plus true 8-depth slice id."""

    def __init__(self, dim: int, phase_bins: int, scale_bins: int, spin_bins: int, depth_slices: int = 8):
        super().__init__()
        self.phase_bins = phase_bins
        self.scale_bins = scale_bins
        self.spin_bins = spin_bins
        self.depth_slices = depth_slices
        self.phase_head = nn.Linear(dim, phase_bins)
        self.scale_head = nn.Linear(dim, scale_bins)
        self.spin_head = nn.Linear(dim, spin_bins)
        self.slice_head = nn.Linear(dim, depth_slices)

    def forward(self, x: torch.Tensor, return_logits: bool = False):
        if x.ndim != 2:
            raise ValueError("DepthRouter expects [B,D]")
        phase_logits = self.phase_head(x)
        scale_logits = self.scale_head(x)
        spin_logits = self.spin_head(x)
        depth_logits = self.slice_head(x)
        phase = torch.argmax(phase_logits, dim=-1)
        scale = torch.argmax(scale_logits, dim=-1)
        spin = torch.argmax(spin_logits, dim=-1)
        depth_slice = torch.argmax(depth_logits, dim=-1)
        if return_logits:
            return phase, scale, spin, depth_slice, {
                "phase_logits": phase_logits,
                "scale_logits": scale_logits,
                "spin_logits": spin_logits,
                "depth_logits": depth_logits,
            }
        return phase, scale, spin, depth_slice


class GeometryKeyProjector(nn.Module):
    """Project a value/query vector into named geometry key charts."""

    def __init__(self, in_dim: int, key_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.key_dim = key_dim
        self.euclid = nn.Linear(in_dim, key_dim)
        self.hyper = nn.Linear(in_dim, key_dim)
        self.sphere = nn.Linear(in_dim, key_dim)
        self.torus = nn.Linear(in_dim, key_dim)
        self.spatial = nn.Linear(in_dim, key_dim)
        self.complex = nn.Linear(in_dim, key_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError("GeometryKeyProjector expects [B,D]")
        return {
            "euclid": self.euclid(x),
            "hyper": torch.tanh(self.hyper(x)) * 0.5,
            "sphere": F.normalize(self.sphere(x), dim=-1, eps=1e-8),
            "torus": self.torus(x),
            "spatial": self.spatial(x),
            "complex": self.complex(x),
        }
