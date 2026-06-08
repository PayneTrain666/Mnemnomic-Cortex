"""Integrated additive cortex wrapper for Spatial LTM + MANN reconstruction."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import SpatialLtmMannConfig, default_config
from .ltm_system import TripleHybridLTM
from .mann_reasoner import MANNReasoner
from .reasoning_stack import ReasoningStack
from .sensory_buffer import SensoryContextBuffer
from .shared_memory import SharedValueStore
from .working_memory import WorkingMemory


class EnhancedSpatialMnemonicCortex(nn.Module):
    """Self-contained integration of shared store, Spatial LTM, MANN, and WM.

    It is additive and can be instantiated independently from the QD6A WM system.
    """

    def __init__(self, cfg: Optional[SpatialLtmMannConfig] = None):
        super().__init__()
        self.cfg = (cfg or default_config()).validate()
        self.transformer_policy = self.cfg.resolve_transformer_policy()
        reasoning_depth = int(self.cfg.effective_reasoning_stack_depth())
        self.input_encoder = nn.Sequential(nn.Linear(self.cfg.input_dim, self.cfg.value_dim), nn.LayerNorm(self.cfg.value_dim), nn.GELU())
        self.output_decoder = nn.Linear(self.cfg.value_dim, self.cfg.output_dim)
        self.shared = SharedValueStore(self.cfg.shared_slots, self.cfg.value_dim)
        self.ltm = TripleHybridLTM(self.cfg, self.shared)
        self.mann = MANNReasoner(self.cfg, self.shared)
        self.wm = WorkingMemory(self.cfg, inherited_bank_layers=self.cfg.inherited_bank_layers)
        self.sensory = SensoryContextBuffer(self.cfg.sensory_capacity, self.cfg.context_capacity)
        self.reasoning_stack = (
            ReasoningStack(
                self.cfg.value_dim,
                depth=reasoning_depth,
                heads=self.cfg.wm_tf_heads,
                dropout=self.cfg.dropout,
                ffn_mult=self.cfg.wm_tf_ffn_mult,
                max_len=self.cfg.wm_tf_max_len,
            )
            if reasoning_depth > 0
            else None
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            B, T, D = x.shape
            return self.input_encoder(x.reshape(B * T, D)).reshape(B, T, self.cfg.value_dim)
        if x.ndim != 2:
            raise ValueError("input must be [B,D] or [B,T,D]")
        return self.input_encoder(x)

    def forward(self, x: torch.Tensor, operation: str = "process", write: bool = False, target_ltm: str = "all", return_traces: bool = True):
        z = self.encode(x)
        z_seed = z.mean(dim=1) if z.ndim == 3 else z
        if operation == "write":
            with torch.no_grad():
                ltm_write = self.ltm.write(z_seed, target=target_ltm, importance=1.0)
                wm_idx = self.wm.write(z_seed)
            out = self.output_decoder(z_seed)
            traces = {"operation": "write", "ltm_write": ltm_write, "wm_write_slots": wm_idx}
            return (out, traces) if return_traces else out
        if operation not in {"process", "reason", "read"}:
            raise ValueError("operation must be process/read/reason/write")
        mann = self.mann.read(z_seed, do_write=write, return_traces=return_traces)
        ltm = self.ltm.read(z_seed, bank="all")
        fused = self.wm.dual_fuse(z_seed, mann.output, ltm.output, traces=[mann.output, ltm.output])
        reasoning_attn = None
        if self.reasoning_stack is not None:
            mem_tokens = torch.stack([mann.output, ltm.output], dim=1)
            refined, reasoning_attn = self.reasoning_stack(fused.unsqueeze(1), mem_tokens=mem_tokens, need_attn=True)
            fused = refined[:, 0, :]
        out = self.output_decoder(fused)
        traces = {
            "operation": operation,
            "mann_confidence": mann.confidence,
            "ltm_confidence": ltm.confidence,
            "mann": mann.traces,
            "ltm": ltm.traces,
            "wm": self.wm.trace(),
            "shared": self.shared.snapshot(),
            "transformer_policy": self.transformer_policy.to_dict(),
            "reasoning_stack_depth": int(self.cfg.effective_reasoning_stack_depth()),
            "reasoning_attn": reasoning_attn,
        }
        return (out, traces) if return_traces else out

    def snapshot(self) -> Dict[str, object]:
        return {"shared": self.shared.snapshot(), "ltm": self.ltm.snapshot(), "mann": self.mann.snapshot(), "sensory": self.sensory.snapshot()}
