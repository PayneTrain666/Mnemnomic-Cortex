"""Opt-in, benchmark-only task decoder capacity components.

These modules are intentionally isolated from the mnemonic cortex runtime and
its working-memory implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _checkpoint_enabled(enabled: bool, training: bool, *inputs: torch.Tensor) -> bool:
    return bool(enabled and training and any(value.requires_grad for value in inputs))


def checkpoint_encoder(
    encoder: nn.TransformerEncoder,
    src: torch.Tensor,
    *,
    enabled: bool,
    mask: Optional[torch.Tensor] = None,
    src_key_padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run a standard encoder with optional non-reentrant layer checkpointing."""
    if not _checkpoint_enabled(enabled, encoder.training, src):
        return encoder(src, mask=mask, src_key_padding_mask=src_key_padding_mask)
    output = src
    for layer in encoder.layers:
        output = checkpoint(
            lambda value, layer=layer: layer(
                value,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            ),
            output,
            use_reentrant=False,
        )
    return encoder.norm(output) if encoder.norm is not None else output


def checkpoint_decoder(
    decoder: nn.TransformerDecoder,
    tgt: torch.Tensor,
    memory: torch.Tensor,
    *,
    enabled: bool,
    tgt_mask: Optional[torch.Tensor] = None,
    memory_mask: Optional[torch.Tensor] = None,
    tgt_key_padding_mask: Optional[torch.Tensor] = None,
    memory_key_padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run a standard decoder with optional non-reentrant layer checkpointing."""
    if not _checkpoint_enabled(enabled, decoder.training, tgt, memory):
        return decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
    output = tgt
    for layer in decoder.layers:
        output = checkpoint(
            lambda value, mem, layer=layer: layer(
                value,
                mem,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            ),
            output,
            memory,
            use_reentrant=False,
        )
    return decoder.norm(output) if decoder.norm is not None else output


@dataclass(frozen=True)
class SharedCapacityConfig:
    d_model: int
    num_heads: int
    num_kv_heads: int
    dim_feedforward: int
    dropout: float
    low_rank: int

    def validate(self) -> bool:
        return (
            self.d_model > 0
            and self.num_heads > 0
            and self.num_kv_heads > 0
            and self.d_model % self.num_heads == 0
            and self.num_heads % self.num_kv_heads == 0
            and self.low_rank > 0
        )


class _AttentionTemplate(nn.Module):
    def __init__(self, config: SharedCapacityConfig) -> None:
        super().__init__()
        head_dim = config.d_model // config.num_heads
        kv_dim = config.num_kv_heads * head_dim
        self.q = nn.Linear(config.d_model, config.d_model)
        self.k = nn.Linear(config.d_model, kv_dim)
        self.v = nn.Linear(config.d_model, kv_dim)
        self.out = nn.Linear(config.d_model, config.d_model)


class _FFNTemplate(nn.Module):
    def __init__(self, config: SharedCapacityConfig) -> None:
        super().__init__()
        self.up = nn.Linear(config.d_model, config.dim_feedforward)
        self.down = nn.Linear(config.dim_feedforward, config.d_model)


class _LowRankDeltas(nn.Module):
    def __init__(self, shapes: dict[str, tuple[int, int]], rank: int) -> None:
        super().__init__()
        self.rank = rank
        self.left = nn.ParameterDict()
        self.right = nn.ParameterDict()
        for name, (out_features, in_features) in shapes.items():
            self.left[name] = nn.Parameter(torch.empty(out_features, rank))
            self.right[name] = nn.Parameter(torch.empty(rank, in_features))
            nn.init.zeros_(self.left[name])
            nn.init.kaiming_uniform_(self.right[name], a=math.sqrt(5))

    def apply_delta(
        self,
        name: str,
        source: nn.Linear,
        value: torch.Tensor,
    ) -> torch.Tensor:
        delta = self.left[name] @ self.right[name]
        return F.linear(value, source.weight + delta / self.rank, source.bias)


class SharedGroupedQueryAttention(nn.Module):
    """GQA using a shared projection template and layer-local low-rank deltas."""

    def __init__(
        self,
        config: SharedCapacityConfig,
        template: _AttentionTemplate,
    ) -> None:
        super().__init__()
        self.config = config
        # Avoid registering the shared template once per layer. Its owner passes
        # it into forward, while this weak reference is only for direct use.
        object.__setattr__(self, "_template", template)
        self.deltas = _LowRankDeltas(
            {
                "q": tuple(template.q.weight.shape),
                "k": tuple(template.k.weight.shape),
                "v": tuple(template.v.weight.shape),
                "out": tuple(template.out.weight.shape),
            },
            config.low_rank,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        template: Optional[_AttentionTemplate] = None,
    ) -> torch.Tensor:
        template = template or self.__dict__["_template"]
        batch, target_len, _ = query.shape
        source_len = key.size(1)
        head_dim = self.config.d_model // self.config.num_heads
        q = self.deltas.apply_delta("q", template.q, query).view(
            batch, target_len, self.config.num_heads, head_dim
        ).transpose(1, 2)
        k = self.deltas.apply_delta("k", template.k, key).view(
            batch, source_len, self.config.num_kv_heads, head_dim
        ).transpose(1, 2)
        v = self.deltas.apply_delta("v", template.v, value).view(
            batch, source_len, self.config.num_kv_heads, head_dim
        ).transpose(1, 2)
        groups = self.config.num_heads // self.config.num_kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        if attn_mask is not None:
            mask = attn_mask
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.ndim == 3:
                mask = mask.unsqueeze(1)
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
            else:
                scores = scores + mask.to(dtype=scores.dtype)
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :].to(torch.bool),
                torch.finfo(scores.dtype).min,
            )
        weights = self.dropout(torch.softmax(scores, dim=-1))
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(
            batch, target_len, self.config.d_model
        )
        return self.deltas.apply_delta("out", template.out, attended)


class _SharedFFN(nn.Module):
    def __init__(self, config: SharedCapacityConfig, template: _FFNTemplate) -> None:
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.deltas = _LowRankDeltas(
            {
                "up": tuple(template.up.weight.shape),
                "down": tuple(template.down.weight.shape),
            },
            config.low_rank,
        )

    def forward(self, value: torch.Tensor, template: _FFNTemplate) -> torch.Tensor:
        value = self.deltas.apply_delta("up", template.up, value)
        value = self.dropout(F.gelu(value))
        return self.deltas.apply_delta("down", template.down, value)


class _SharedDecoderLayer(nn.Module):
    def __init__(
        self,
        config: SharedCapacityConfig,
        self_template: _AttentionTemplate,
        cross_template: _AttentionTemplate,
        ffn_template: _FFNTemplate,
    ) -> None:
        super().__init__()
        self.self_attention = SharedGroupedQueryAttention(config, self_template)
        self.cross_attention = SharedGroupedQueryAttention(config, cross_template)
        self.ffn = _SharedFFN(config, ffn_template)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        self_template: _AttentionTemplate,
        cross_template: _AttentionTemplate,
        ffn_template: _FFNTemplate,
        tgt_mask: Optional[torch.Tensor],
        memory_mask: Optional[torch.Tensor],
        tgt_key_padding_mask: Optional[torch.Tensor],
        memory_key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        value = self.norm1(
            tgt
            + self.dropout(
                self.self_attention(
                    tgt,
                    tgt,
                    tgt,
                    attn_mask=tgt_mask,
                    key_padding_mask=tgt_key_padding_mask,
                    template=self_template,
                )
            )
        )
        value = self.norm2(
            value
            + self.dropout(
                self.cross_attention(
                    value,
                    memory,
                    memory,
                    attn_mask=memory_mask,
                    key_padding_mask=memory_key_padding_mask,
                    template=cross_template,
                )
            )
        )
        return self.norm3(value + self.dropout(self.ffn(value, ffn_template)))


class SharedGQADecoder(nn.Module):
    """Decoder with templates shared across layers and low-rank layer deltas."""

    def __init__(self, config: SharedCapacityConfig, num_layers: int) -> None:
        super().__init__()
        if not config.validate():
            raise ValueError("incompatible shared GQA capacity configuration")
        self.config = config
        self.self_template = _AttentionTemplate(config)
        self.cross_template = _AttentionTemplate(config)
        self.ffn_template = _FFNTemplate(config)
        self.layers = nn.ModuleList(
            [
                _SharedDecoderLayer(
                    config,
                    self.self_template,
                    self.cross_template,
                    self.ffn_template,
                )
                for _ in range(max(1, num_layers))
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        *,
        activation_checkpointing: bool = False,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        value = tgt
        for layer in self.layers:
            args = (
                memory,
                self.self_template,
                self.cross_template,
                self.ffn_template,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
            )
            if _checkpoint_enabled(activation_checkpointing, self.training, value, memory):
                value = checkpoint(layer, value, *args, use_reentrant=False)
            else:
                value = layer(value, *args)
        return self.norm(value)
