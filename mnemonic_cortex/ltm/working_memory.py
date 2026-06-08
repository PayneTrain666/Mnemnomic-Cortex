"""Working memory fusion surface for LTM-native and MANN reasoning paths."""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

from .config import SpatialLtmMannConfig
from .transformer_utils import TransformerStack


class WorkingMemory(nn.Module):
    """Fast scratch store + dual fusion gate.

    WM remains a separate fast scratchpad; it receives MANN and LTM readouts and
    decides how to fuse them. This follows the DOCX instruction that WM should
    retain its current role while interfacing correctly with Spatial LTM/MANN.
    """

    def __init__(self, cfg: SpatialLtmMannConfig, *, inherited_bank_layers: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.slots = cfg.wm_slots
        self.dim = cfg.wm_dim
        self.wm_tf_depth = int(cfg.effective_wm_tf_depth(inherited_bank_layers=inherited_bank_layers))
        self.wm = nn.Parameter(torch.randn(self.slots, self.dim) * 0.02)
        self.input_proj = nn.Linear(cfg.value_dim, self.dim)
        self.output_proj = nn.Linear(self.dim, cfg.value_dim)
        self.read_q = nn.Linear(self.dim, self.dim)
        self.read_k = nn.Linear(self.dim, self.dim)
        self.state = nn.GRUCell(self.dim, self.dim)
        self.use_wm_transformer = cfg.wm_use_transformer
        self.wm_stack = (
            TransformerStack(
                cfg.wm_dim,
                self.wm_tf_depth,
                cfg.wm_tf_heads,
                cfg.dropout,
                cfg.wm_tf_ffn_mult,
                cfg.wm_tf_max_len,
            )
            if cfg.wm_use_transformer and self.wm_tf_depth > 0
            else None
        )
        self.trace_proj = nn.Linear(cfg.value_dim, cfg.wm_dim)
        self.fuse_gate = nn.Sequential(nn.Linear(cfg.wm_dim, 64), nn.GELU(), nn.Linear(64, 3))
        self._last: Dict[str, object] = {}

    def read(self, x: torch.Tensor):
        s = self.input_proj(x)
        q = self.read_q(s)
        k = self.read_k(self.wm)
        attn = torch.softmax((q @ k.t()) / (self.dim ** 0.5), dim=-1)
        r = attn @ self.wm
        return r, attn

    @torch.no_grad()
    def write(self, content: torch.Tensor, rate: float = 0.1) -> torch.Tensor:
        content = self.input_proj(content)
        sim = content @ self.wm.t()
        idx = torch.argmax(sim, dim=-1)
        for b in range(content.size(0)):
            s = int(idx[b].item())
            self.wm[s].mul_(1.0 - rate).add_(rate * content[b])
        return idx.detach().cpu()

    def _build_tokens(self, wm_read: torch.Tensor, mann_vec: torch.Tensor, ltm_vec: torch.Tensor, traces: Optional[Iterable[torch.Tensor]] = None) -> torch.Tensor:
        tokens = [wm_read.unsqueeze(1), self.input_proj(mann_vec).unsqueeze(1), self.input_proj(ltm_vec).unsqueeze(1)]
        if traces is not None:
            trace_tokens = []
            for t in list(traces)[: self.cfg.wm_tf_trace_cap]:
                if t.size(-1) != self.dim:
                    t = self.trace_proj(t)
                trace_tokens.append(t.unsqueeze(1))
            if trace_tokens:
                tokens.append(torch.cat(trace_tokens, dim=1))
        return torch.cat(tokens, dim=1)

    def dual_fuse(self, seed: torch.Tensor, mann_vec: torch.Tensor, ltm_vec: torch.Tensor, traces: Optional[Iterable[torch.Tensor]] = None, need_tf_attn: bool = False) -> torch.Tensor:
        wm_read, wm_attn = self.read(seed)
        tf_attn = None
        if self.wm_stack is not None:
            toks = self._build_tokens(wm_read, mann_vec, ltm_vec, traces)
            toks2, tf_attn = self.wm_stack(toks, need_weights=need_tf_attn)
            pooled = toks2[:, 0, :]
        else:
            pooled = wm_read
        gates = torch.softmax(self.fuse_gate(pooled), dim=-1)
        fused_value = gates[:, 0:1] * mann_vec + gates[:, 1:2] * ltm_vec + gates[:, 2:3] * self.output_proj(pooled)
        self._last = {"wm_slot_attn": wm_attn, "gates": gates, "tf_attn": tf_attn}
        return fused_value

    def trace(self) -> Dict[str, object]:
        return dict(self._last)
