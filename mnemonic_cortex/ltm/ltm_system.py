"""Triple-hybrid LTM with explicit Spatial LTM and per-depth geometry charts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

from .config import SpatialLtmMannConfig
from .geometry_keys import DepthRouter
from .manifold_ops import frechet_mean
from .memory_bank import GeometryMemoryBank
from .shared_memory import SharedValueStore


@dataclass
class LTMReadResult:
    output: torch.Tensor
    confidence: torch.Tensor
    traces: Dict[str, object]


class LTMSubsystem(nn.Module):
    """Geometry-aware LTM subsystem backed by a shared value store."""

    def __init__(self, name: str, shared: SharedValueStore, key_dim: int, default_geom: str, cfg: SpatialLtmMannConfig, depth_chart: Iterable[str]):
        super().__init__()
        self.name = name
        self.cfg = cfg
        self.shared = shared
        self.key_encoder = nn.Sequential(
            nn.Linear(cfg.value_dim, key_dim),
            nn.LayerNorm(key_dim),
            nn.GELU(),
            nn.Linear(key_dim, key_dim),
        )
        self.value_encoder = nn.Identity()
        self.bank = GeometryMemoryBank(
            slots=cfg.shared_slots,
            key_dim=key_dim,
            default_geom=default_geom,
            depth_slices=cfg.depth_slices,
            conformal_b=cfg.conformal_b,
            min_c=cfg.min_c,
            max_c=cfg.max_c,
        )
        self.bank.set_slice_geometries(depth_chart)

    def _collapse(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.mean(dim=1)
        if x.ndim != 2:
            raise ValueError("LTMSubsystem expects [B,V] or [B,T,V]")
        if x.size(-1) != self.cfg.value_dim:
            raise ValueError(f"expected last dim {self.cfg.value_dim}")
        return x

    def read(self, x: torch.Tensor, depth_slice: int | torch.Tensor = 0, b_override: Optional[float] = None) -> LTMReadResult:
        x_in = self._collapse(x)
        qk = self.key_encoder(x_in)
        out, trace = self.bank.retrieve(qk, self.shared.values, topk=self.cfg.ltm_topk, depth_slice=depth_slice, b_override=b_override)
        confidence = trace.weights.max(dim=-1, keepdim=True).values
        return LTMReadResult(output=out, confidence=confidence, traces={"subsystem": self.name, "bank": trace, "query_key": qk})

    @torch.no_grad()
    def write(self, x: torch.Tensor, depth_slice: int | torch.Tensor = 0, importance: float = 1.0) -> Dict[str, object]:
        x_in = self._collapse(x)
        qk = self.key_encoder(x_in)
        _, trace = self.bank.retrieve(qk, self.shared.values, topk=self.cfg.ltm_topk, depth_slice=depth_slice)
        top1 = trace.indices[:, 0]
        lr = self.cfg.ltm_write_lr * float(0.5 + 0.5 * max(0.0, min(1.0, importance)))
        self.shared.consolidate_write(top1, x_in, lr=lr)
        for b in range(qk.size(0)):
            s = int(top1[b].item())
            self.bank.keys[s].mul_(0.98).add_(0.02 * qk[b])
        return {"subsystem": self.name, "depth_slice": trace.depth_slice, "written_slots": top1.detach().cpu()}

    @torch.no_grad()
    def consolidate(self, x: torch.Tensor, depth_slice: int | torch.Tensor = 0) -> Dict[str, object]:
        x_in = self._collapse(x)
        qk = self.key_encoder(x_in)
        _, trace = self.bank.retrieve(qk, self.shared.values, topk=self.cfg.ltm_topk, depth_slice=depth_slice)
        ds = trace.depth_slice
        geom = trace.geometry
        c = float(self.bank.curv_per_slice[ds].clamp(self.cfg.min_c, self.cfg.max_c).item())
        mu_k = frechet_mean(geom, trace.selected_keys, trace.weights, c=c if geom == "hyper" else None, iters=6, lr=0.5)
        mu_v = torch.sum(trace.weights.unsqueeze(-1) * trace.selected_values, dim=1)
        top1 = trace.indices[:, 0]
        for b in range(mu_k.size(0)):
            s = int(top1[b].item())
            self.bank.keys[s].mul_(0.98).add_(0.02 * mu_k[b])
        self.shared.consolidate_write(top1, mu_v, lr=self.cfg.ltm_write_lr)
        return {"subsystem": self.name, "depth_slice": ds, "geom": geom, "consolidated_slots": top1.detach().cpu()}


class TripleHybridLTM(nn.Module):
    """HG episodic + CGMN semantic + Curved + Spatial + Procedural LTM over shared values."""

    def __init__(self, cfg: SpatialLtmMannConfig, shared_store: Optional[SharedValueStore] = None):
        super().__init__()
        self.cfg = cfg.validate()
        self.shared = shared_store if shared_store is not None else SharedValueStore(cfg.shared_slots, cfg.value_dim)
        self.depth_router = DepthRouter(cfg.value_dim, cfg.phase_bins, cfg.scale_bins, cfg.spin_bins, cfg.depth_slices)
        self.hg = LTMSubsystem("episodic_hg", self.shared, cfg.key_dim, "hyper", cfg, cfg.hg_depth_chart)
        self.cgmn = LTMSubsystem("semantic_cgmn", self.shared, cfg.key_dim, "sphere", cfg, cfg.cgmn_depth_chart)
        self.curved = LTMSubsystem("curved_associative", self.shared, cfg.key_dim, "curved", cfg, cfg.curved_depth_chart)
        self.spatial = LTMSubsystem("spatial_atlas", self.shared, cfg.key_dim, "spatial", cfg, cfg.spatial_depth_chart)
        self.procedural = LTMSubsystem(
            "procedural_spcp",
            self.shared,
            cfg.key_dim,
            "sphere",
            cfg,
            cfg.procedural_depth_chart,
        )
        self.bank_gate = nn.Sequential(nn.Linear(cfg.value_dim, 64), nn.GELU(), nn.Linear(64, 5))

    def _collapse(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1) if x.ndim == 3 else x

    def read(self, x: torch.Tensor, bank: str = "all", depth_slice: int | torch.Tensor | None = None) -> LTMReadResult:
        x_in = self._collapse(x)
        if depth_slice is None:
            *_bins, depth_slice = self.depth_router(x_in)
        subs = {
            "hg": self.hg,
            "cgmn": self.cgmn,
            "curved": self.curved,
            "spatial": self.spatial,
            "procedural": self.procedural,
        }
        if bank != "all":
            if bank == "curved_associative":
                bank = "curved"
            if bank not in subs:
                raise ValueError(f"unknown LTM bank {bank}")
            return subs[bank].read(x_in, depth_slice=depth_slice)
        reads = {name: sub.read(x_in, depth_slice=depth_slice) for name, sub in subs.items()}
        logits = self.bank_gate(x_in)
        weights = torch.softmax(logits, dim=-1)
        stacked = torch.stack(
            [
                reads["hg"].output,
                reads["cgmn"].output,
                reads["curved"].output,
                reads["spatial"].output,
                reads["procedural"].output,
            ],
            dim=1,
        )
        fused = torch.sum(weights.unsqueeze(-1) * stacked, dim=1)
        conf_stack = torch.cat(
            [
                reads["hg"].confidence,
                reads["cgmn"].confidence,
                reads["curved"].confidence,
                reads["spatial"].confidence,
                reads["procedural"].confidence,
            ],
            dim=-1,
        )
        confidence = torch.sum(weights * conf_stack, dim=-1, keepdim=True)
        disagreement = torch.var(stacked, dim=1).mean(dim=-1, keepdim=True)
        return LTMReadResult(output=fused, confidence=confidence, traces={"reads": reads, "bank_weights": weights, "disagreement": disagreement})

    @torch.no_grad()
    def write(self, x: torch.Tensor, target: str = "all", depth_slice: int | torch.Tensor | None = None, importance: float = 1.0) -> Dict[str, object]:
        x_in = self._collapse(x)
        if depth_slice is None:
            *_bins, depth_slice = self.depth_router(x_in)
        targets = [target] if target != "all" else ["hg", "cgmn", "curved", "spatial", "procedural"]
        targets = ["curved" if t == "curved_associative" else t for t in targets]
        subs = {
            "hg": self.hg,
            "cgmn": self.cgmn,
            "curved": self.curved,
            "curved_associative": self.curved,
            "spatial": self.spatial,
            "procedural": self.procedural,
        }
        return {t: subs[t].write(x_in, depth_slice=depth_slice, importance=importance) for t in targets}

    @torch.no_grad()
    def consolidate(self, x: torch.Tensor, target: str = "all", depth_slice: int | torch.Tensor | None = None) -> Dict[str, object]:
        x_in = self._collapse(x)
        if depth_slice is None:
            *_bins, depth_slice = self.depth_router(x_in)
        targets = [target] if target != "all" else ["hg", "cgmn", "curved", "spatial", "procedural"]
        targets = ["curved" if t == "curved_associative" else t for t in targets]
        subs = {
            "hg": self.hg,
            "cgmn": self.cgmn,
            "curved": self.curved,
            "curved_associative": self.curved,
            "spatial": self.spatial,
            "procedural": self.procedural,
        }
        return {t: subs[t].consolidate(x_in, depth_slice=depth_slice) for t in targets}

    def snapshot(self) -> Dict[str, object]:
        return {
            "shared": self.shared.snapshot(),
            "hg": self.hg.bank.snapshot(),
            "cgmn": self.cgmn.bank.snapshot(),
            "curved": self.curved.bank.snapshot(),
            "spatial": self.spatial.bank.snapshot(),
            "procedural": self.procedural.bank.snapshot(),
        }
