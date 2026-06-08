"""Traditional MANN reasoning path over geometry-specific keys."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import SpatialLtmMannConfig
from .geometry_keys import DepthRouter, GeometryKeyProjector
from .manifold_ops import log_map
from .memory_bank import GeometryMemoryBank
from .shared_memory import SharedValueStore


@dataclass
class MANNReadResult:
    output: torch.Tensor
    confidence: torch.Tensor
    traces: Dict[str, object]


class MANNReasoner(nn.Module):
    """Multi-hop key/value slot-attention memory using all LTM geometries.

    This is the traditional MANN path requested in the DOCX: it does not replace
    HG/CGMN/Spatial native retrieval. It operates alongside them and can be fused
    by WM or a higher-level cortex controller.
    """

    def __init__(self, cfg: SpatialLtmMannConfig, shared_store: SharedValueStore):
        super().__init__()
        self.cfg = cfg.validate()
        self.shared = shared_store
        self.projector = GeometryKeyProjector(cfg.value_dim, cfg.key_dim)
        self.depth_router = DepthRouter(cfg.value_dim, cfg.phase_bins, cfg.scale_bins, cfg.spin_bins, cfg.depth_slices)
        self.banks = nn.ModuleDict({
            "euclid": GeometryMemoryBank(cfg.shared_slots, cfg.key_dim, "euclid", cfg.depth_slices, cfg.conformal_b, cfg.min_c, cfg.max_c),
            "hyper": GeometryMemoryBank(cfg.shared_slots, cfg.key_dim, "hyper", cfg.depth_slices, cfg.conformal_b, cfg.min_c, cfg.max_c),
            "sphere": GeometryMemoryBank(cfg.shared_slots, cfg.key_dim, "sphere", cfg.depth_slices, cfg.conformal_b, cfg.min_c, cfg.max_c),
            "torus": GeometryMemoryBank(cfg.shared_slots, cfg.key_dim, "torus", cfg.depth_slices, cfg.conformal_b, cfg.min_c, cfg.max_c),
            "spatial": GeometryMemoryBank(cfg.shared_slots, cfg.key_dim, "spatial", cfg.depth_slices, cfg.conformal_b, cfg.min_c, cfg.max_c),
            "complex": GeometryMemoryBank(cfg.shared_slots, cfg.key_dim, "euclid", cfg.depth_slices, cfg.conformal_b, cfg.min_c, cfg.max_c),
        })
        for bank in self.banks.values():
            bank.set_slice_geometries(cfg.mann_depth_chart)
        self.bank_gate = nn.Sequential(nn.Linear(cfg.value_dim, 64), nn.GELU(), nn.Linear(64, len(self.banks)))
        self.hop = nn.GRUCell(cfg.value_dim, cfg.value_dim)
        self.stability_proj = nn.Linear(cfg.key_dim, cfg.value_dim, bias=False)
        self.conf_head = nn.Sequential(nn.Linear(cfg.value_dim, 1), nn.Sigmoid())

    def _collapse(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.mean(dim=1)
        if x.ndim != 2 or x.size(-1) != self.cfg.value_dim:
            raise ValueError(f"MANNReasoner expects [B,{self.cfg.value_dim}] or [B,T,{self.cfg.value_dim}]")
        return x

    def read(self, x: torch.Tensor, do_write: bool = False, depth_slice: int | torch.Tensor | None = None, return_traces: bool = True) -> MANNReadResult:
        q = self._collapse(x)
        B = q.size(0)
        traces = {"hops": [], "do_write": bool(do_write)}
        if depth_slice is None:
            phase, scale, spin, depth_slice, router_logits = self.depth_router(q, return_logits=True)
        else:
            phase = scale = spin = torch.zeros(B, dtype=torch.long, device=q.device)
            router_logits = {}
        for hop_i in range(self.cfg.mann_hops):
            keys = self.projector(q)
            bank_logits = self.bank_gate(q)
            bank_weights = torch.softmax(bank_logits, dim=-1)
            bank_outs = []
            bank_traces = {}
            for j, name in enumerate(self.banks.keys()):
                out, trace = self.banks[name].retrieve(keys[name], self.shared.values, topk=self.cfg.mann_topk, depth_slice=depth_slice)
                # Mild hop-stability term via log-map into Euclidean anchor tangent.
                geom = trace.geometry
                anchor = keys["euclid"]
                proxy = torch.sum(trace.weights.unsqueeze(-1) * trace.selected_keys, dim=1)
                v_anchor = log_map(geom, anchor, proxy, c=float(self.cfg.min_c) if geom == "hyper" else None)
                out = out + 0.02 * self.stability_proj(v_anchor).detach()
                bank_outs.append(out)
                bank_traces[name] = trace
            stacked = torch.stack(bank_outs, dim=1)
            v = torch.sum(bank_weights.unsqueeze(-1) * stacked, dim=1)
            q = self.hop(v, q)
            if do_write:
                # Scratch-only MANN write, not durable LTM consolidation.
                slot_idx = bank_traces["euclid"].indices[:, 0]
                self.shared.scratch_write(slot_idx, v, lr=self.cfg.mann_write_lr)
            if return_traces:
                traces["hops"].append({
                    "hop": hop_i,
                    "bank_weights": bank_weights,
                    "bank_traces": bank_traces,
                    "phase": phase,
                    "scale": scale,
                    "spin": spin,
                    "depth_slice": depth_slice,
                })
        confidence = self.conf_head(q)
        traces["router_logits"] = router_logits
        return MANNReadResult(output=q, confidence=confidence, traces=traces)

    def snapshot(self) -> Dict[str, object]:
        return {name: bank.snapshot() for name, bank in self.banks.items()}
