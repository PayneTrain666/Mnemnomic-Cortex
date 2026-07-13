"""
Plain-language summary
----------------------
What this file is for: Stores memories as overlapping hologram patterns in slots, with codes for depth/bank/role.
How it fits in the system: Used by LTM banks and consolidated memory to stack several memories in one address.
Status: WORKING
Important notes for non-coders: Codebook tensors move with the module via _apply when you call .to(device).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _stable_seed(*parts: str) -> int:
    s = "|".join(parts).encode("utf-8")
    return int(hashlib.sha256(s).hexdigest()[:16], 16) % (2**31 - 1)


def _unit_real(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


@dataclass
class QuantumHologramConfig:
    enabled: bool = True
    hrr_dim: int = 256
    num_slots: int = 1024
    num_depths: int = 8
    bank_name: str = "default_bank"
    interference_threshold: float = 0.90
    max_triplets_per_slot: int = 8
    eps: float = 1e-8

    def validate(self) -> None:
        if int(self.hrr_dim) <= 0:
            raise ValueError("hrr_dim must be positive")
        if int(self.num_slots) <= 0:
            raise ValueError("num_slots must be positive")
        if int(self.num_depths) <= 0:
            raise ValueError("num_depths must be positive")
        if not str(self.bank_name):
            raise ValueError("bank_name must be non-empty")
        if not (0.0 <= float(self.interference_threshold) <= 1.0):
            raise ValueError("interference_threshold must be in [0,1]")
        if int(self.max_triplets_per_slot) <= 0:
            raise ValueError("max_triplets_per_slot must be positive")


class QuantumHologramCodebook:
    """Deterministic full codebook for depth/bank/triplet/slot codes."""

    TRIPLETS = ("anchor", "direction", "phase")

    def __init__(self, cfg: QuantumHologramConfig, bank_names: Optional[Sequence[str]] = None):
        cfg.validate()
        self.cfg = cfg
        banks = sorted(set([cfg.bank_name, *(bank_names or [])]))
        self.depth_codes = {d: self._make_code("depth", str(d)) for d in range(cfg.num_depths)}
        self.bank_codes = {b: self._make_code("bank", b) for b in banks}
        self.triplet_codes = {t: self._make_code("triplet", t) for t in self.TRIPLETS}
        self.slot_codes = {s: self._make_code("slot", str(s)) for s in range(cfg.num_slots)}

    def _make_code(self, namespace: str, token: str) -> torch.Tensor:
        g = torch.Generator()
        g.manual_seed(_stable_seed(namespace, token, str(self.cfg.hrr_dim)))
        raw = torch.randn(self.cfg.hrr_dim, generator=g, dtype=torch.float32)
        return _unit_real(raw)

    def bank_code(self, bank_name: str) -> torch.Tensor:
        if bank_name not in self.bank_codes:
            code = self._make_code("bank", bank_name)
            # Keep lazily created codes on the same device as the rest of the codebook.
            for existing in self.bank_codes.values():
                if isinstance(existing, torch.Tensor):
                    code = code.to(device=existing.device, dtype=existing.dtype)
                    break
            self.bank_codes[bank_name] = code
        return self.bank_codes[bank_name]


class FFTHRRTripletStacker:
    """FFT-HRR binder for triplet stacking with codebook role separation."""

    def __init__(self, cfg: QuantumHologramConfig, codebook: QuantumHologramCodebook):
        self.cfg = cfg
        self.codebook = codebook

    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        af = torch.fft.rfft(a.float(), dim=-1)
        bf = torch.fft.rfft(b.float(), dim=-1)
        return torch.fft.irfft(af * bf, n=a.size(-1), dim=-1)

    def superpose(self, xs: Iterable[torch.Tensor]) -> torch.Tensor:
        acc = None
        for x in xs:
            acc = x if acc is None else acc + x
        if acc is None:
            acc = torch.zeros(self.cfg.hrr_dim)
        return _unit_real(acc)

    def stack_triplet(
        self,
        *,
        anchor: torch.Tensor,
        direction: torch.Tensor,
        phase: torch.Tensor,
        slot_index: int,
        depth_index: int,
        bank_name: str,
    ) -> torch.Tensor:
        slot_index = int(slot_index) % int(self.cfg.num_slots)
        depth_index = int(depth_index) % int(self.cfg.num_depths)
        dev = anchor.device
        dt = anchor.dtype
        slot_code = self.codebook.slot_codes[slot_index].to(device=dev, dtype=dt)
        depth_code = self.codebook.depth_codes[depth_index].to(device=dev, dtype=dt)
        bank_code = self.codebook.bank_code(str(bank_name)).to(device=dev, dtype=dt)
        anchor_code = self.codebook.triplet_codes["anchor"].to(device=dev, dtype=dt)
        direction_code = self.codebook.triplet_codes["direction"].to(device=dev, dtype=dt)
        phase_code = self.codebook.triplet_codes["phase"].to(device=dev, dtype=dt)
        a = self.bind(_unit_real(anchor), anchor_code)
        d = self.bind(_unit_real(direction), direction_code)
        p = self.bind(_unit_real(phase), phase_code)
        content = self.superpose((a, d, p))
        context = self.superpose((slot_code, depth_code, bank_code))
        return self.bind(content, context)

    def interference_score(self, x: torch.Tensor, y: torch.Tensor) -> float:
        xn = _unit_real(x.reshape(1, -1), eps=self.cfg.eps)[0]
        yn = _unit_real(y.reshape(1, -1), eps=self.cfg.eps)[0]
        return float(torch.abs(torch.dot(xn, yn)).item())


class QuantumHologramSlotBank(nn.Module):
    """Slot-level complex hologram store with crosstalk-aware stacking."""

    def __init__(self, cfg: QuantumHologramConfig, bank_names: Optional[Sequence[str]] = None):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.codebook = QuantumHologramCodebook(cfg, bank_names=bank_names)
        self.stacker = FFTHRRTripletStacker(cfg, self.codebook)
        self.register_buffer("holograms", torch.zeros(cfg.num_slots, cfg.hrr_dim, dtype=torch.float32))
        self.register_buffer("triplet_counts", torch.zeros(cfg.num_slots, dtype=torch.long))
        self.register_buffer("interference_flags", torch.zeros(cfg.num_slots, dtype=torch.bool))

    def _apply(self, fn, recurse=True):
        """Migrate registered buffers and plain-dict codebook tensors together."""
        ret = super()._apply(fn, recurse=recurse)
        codebook = getattr(self, "codebook", None)
        if codebook is not None:
            for attr in ("depth_codes", "bank_codes", "triplet_codes", "slot_codes"):
                table = getattr(codebook, attr, None)
                if not isinstance(table, dict):
                    continue
                for key, value in list(table.items()):
                    if isinstance(value, torch.Tensor):
                        table[key] = fn(value)
        return ret

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.view(1, -1)
        if x.size(-1) == self.cfg.hrr_dim:
            return _unit_real(x)
        if x.size(-1) > self.cfg.hrr_dim:
            return _unit_real(x[..., : self.cfg.hrr_dim])
        pad = torch.zeros(x.size(0), self.cfg.hrr_dim - x.size(-1), device=x.device, dtype=x.dtype)
        return _unit_real(torch.cat([x, pad], dim=-1))

    @torch.no_grad()
    def store_batch(
        self,
        *,
        slot_indices: torch.Tensor,
        anchor: torch.Tensor,
        direction: torch.Tensor,
        phase: torch.Tensor,
        depth_index: int = 0,
        bank_name: Optional[str] = None,
    ) -> Dict[str, float]:
        if not self.cfg.enabled:
            return {"stored": 0.0, "interference_rate": 0.0}
        slots = slot_indices.reshape(-1).tolist()
        a = self._project(anchor)
        d = self._project(direction)
        p = self._project(phase)
        n = min(len(slots), a.size(0), d.size(0), p.size(0))
        interferences = 0
        stored = 0
        for i in range(n):
            sid = int(slots[i]) % int(self.cfg.num_slots)
            if int(self.triplet_counts[sid].item()) >= int(self.cfg.max_triplets_per_slot):
                continue
            h = self.stacker.stack_triplet(
                anchor=a[i],
                direction=d[i],
                phase=p[i],
                slot_index=sid,
                depth_index=int(depth_index),
                bank_name=str(bank_name or self.cfg.bank_name),
            )
            prev = self.holograms[sid]
            if prev.abs().sum().item() > 0:
                inter = self.stacker.interference_score(prev, h)
                if inter >= float(self.cfg.interference_threshold):
                    self.interference_flags[sid] = True
                    interferences += 1
                    # Phase decorrelation fallback reduces crosstalk accumulation.
                    h = _unit_real(h - inter * prev)
            self.holograms[sid] = _unit_real(prev + h)
            self.triplet_counts[sid] += 1
            stored += 1
        return {
            "stored": float(stored),
            "interference_rate": float(interferences) / float(max(1, stored)),
            "active_slots": float((self.triplet_counts > 0).sum().item()),
        }

    @torch.no_grad()
    def trace_summary(self) -> Dict[str, float]:
        active = int((self.triplet_counts > 0).sum().item())
        flagged = int(self.interference_flags.sum().item())
        return {
            "enabled": float(bool(self.cfg.enabled)),
            "hrr_dim": float(self.cfg.hrr_dim),
            "num_slots": float(self.cfg.num_slots),
            "active_slots": float(active),
            "triplets_stored": float(self.triplet_counts.sum().item()),
            "interference_flagged_slots": float(flagged),
            "interference_slot_rate": float(flagged) / float(max(1, active)),
        }
