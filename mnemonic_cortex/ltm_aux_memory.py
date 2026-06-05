from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsolidatedLTMBank(nn.Module):
    """Attention-based consolidated trace bank for cross-memory writes/reads."""

    def __init__(self, dim: int, slots: int = 512, n_heads: int = 8):
        super().__init__()
        self.dim = int(dim)
        self.slots = int(slots)
        heads = int(n_heads)
        if self.dim % heads != 0:
            for h in (8, 4, 2, 1):
                if self.dim % h == 0:
                    heads = h
                    break
        self.register_buffer("memory", torch.randn(self.slots, self.dim) * 0.02)
        self.register_buffer("usage", torch.zeros(self.slots))
        self.write_attn = nn.MultiheadAttention(self.dim, num_heads=heads, batch_first=True)
        self.read_attn = nn.MultiheadAttention(self.dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(self.dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        operation: str = "read",
        lightbulb_intensity: float = 0.0,
    ) -> torch.Tensor:
        if operation == "write":
            if x.dim() == 2:
                x = x.unsqueeze(1)
            pooled = x.mean(dim=1, keepdim=True)
            mem = self.memory.unsqueeze(0).expand(pooled.size(0), -1, -1)
            attn_out, _ = self.write_attn(pooled, mem, mem, need_weights=False)
            write_vec = self.norm(pooled + attn_out).squeeze(1)
            with torch.no_grad():
                slot = int(torch.argmin(self.usage).item())
                alpha = max(0.05, min(0.95, 0.2 + 0.6 * float(lightbulb_intensity)))
                self.memory[slot] = (1.0 - alpha) * self.memory[slot] + alpha * write_vec.mean(dim=0)
                self.usage[slot] = self.usage[slot] + alpha
            return write_vec.unsqueeze(1)

        if x.dim() == 2:
            x = x.unsqueeze(1)
        mem = self.memory.clone().unsqueeze(0).expand(x.size(0), -1, -1)
        read_out, _ = self.read_attn(x, mem, mem, need_weights=False)
        boost = 1.0 + 0.25 * float(lightbulb_intensity)
        return self.norm(x + boost * read_out)


class NeuralFieldMemory(nn.Module):
    """Lightweight neural-field buffer ingested during writes, read as fusion token."""

    def __init__(self, dim: int, slots: int = 64):
        super().__init__()
        self.dim = int(dim)
        self.slots = int(slots)
        self.register_buffer("field", torch.randn(self.slots, self.dim) * 0.02)
        self.register_buffer("usage", torch.zeros(self.slots))
        self.ingest_gate = nn.Parameter(torch.tensor(0.30))

    @torch.no_grad()
    def ingest(self, x: torch.Tensor) -> None:
        if x.dim() == 3:
            pooled = x.mean(dim=1)
        elif x.dim() == 2:
            pooled = x
        else:
            return
        slot = int(torch.argmin(self.usage).item())
        alpha = float(torch.sigmoid(self.ingest_gate).item())
        self.field[slot] = (1.0 - alpha) * self.field[slot] + alpha * pooled.mean(dim=0)
        self.usage[slot] = self.usage[slot] + alpha

    def read_token(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            q = x.mean(dim=1)
        else:
            q = x
        field = self.field.clone()
        mem = F.normalize(field, dim=-1)
        qn = F.normalize(q, dim=-1)
        scores = torch.matmul(qn, mem.transpose(0, 1))
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, field)
