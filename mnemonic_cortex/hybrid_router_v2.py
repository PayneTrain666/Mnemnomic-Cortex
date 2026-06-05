from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridRouterV2(nn.Module):
    """Attention router over subsystem memory tokens."""

    def __init__(self, input_dim: int, n_subsystems: int = 5, n_heads: int = 8):
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_subsystems = int(n_subsystems)
        heads = int(n_heads)
        if self.input_dim % heads != 0:
            for h in (8, 4, 2, 1):
                if self.input_dim % h == 0:
                    heads = h
                    break
        self.n_heads = heads
        self.context_proj = nn.Linear(self.input_dim, self.input_dim)
        self.token_attn = nn.MultiheadAttention(
            self.input_dim, num_heads=self.n_heads, batch_first=True
        )
        self.refine_attn = nn.MultiheadAttention(
            self.input_dim, num_heads=self.n_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(self.input_dim)
        self.weight_head = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        context: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # tokens: [B, M, D]
        bsz, n_tok, dim = tokens.shape
        if n_tok != self.n_subsystems:
            raise ValueError(
                f"expected {self.n_subsystems} subsystem tokens, got {n_tok}"
            )
        if context is None:
            ctx = tokens.mean(dim=1, keepdim=True)
        else:
            ctx = self.context_proj(context).unsqueeze(1)
        refined, _ = self.token_attn(
            tokens,
            tokens,
            tokens,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        refined = self.norm(tokens + refined)
        ctx_mix, _ = self.refine_attn(
            refined,
            ctx.expand(-1, n_tok, -1),
            ctx.expand(-1, n_tok, -1),
            need_weights=False,
        )
        refined = self.norm(refined + ctx_mix)
        logits = self.weight_head(refined).squeeze(-1)
        weights = torch.softmax(logits, dim=-1)
        return weights, refined
