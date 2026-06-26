from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ltm.transformer_utils import TransformerBlock, TransformerStack
from .memory_attention import MultiScaleAttention


DEFAULT_CMS_DEPTH_MANIFOLDS: Tuple[str, ...] = (
    "euclidean",
    "hyperbolic",
    "spatial_s3",
    "complex_projective",
    "complex_projective_kahler",
    "toroidal",
    "grassmann_subspace",
    "quaternion_spatial_loop",
)

CMS_SUPER_PRODUCT_MANIFOLD = "complex_projective_super_product"

DEFAULT_CMS_VISIBLE_MANIFOLD_STACK: Tuple[str, ...] = DEFAULT_CMS_DEPTH_MANIFOLDS + (
    CMS_SUPER_PRODUCT_MANIFOLD,
)

CMS_MANIFOLD_STORAGE_FACTORS: Dict[str, float] = {
    "euclidean": 1.00,
    "hyperbolic": 2.50,
    "spatial_s3": 4.00,
    "complex_projective": 5.50,
    "complex_projective_kahler": 6.00,
    "toroidal": 3.00,
    "grassmann_subspace": 5.00,
    "quaternion_spatial_loop": 8.00,
    CMS_SUPER_PRODUCT_MANIFOLD: 12.00,
}


@dataclass(frozen=True)
class ConsolidatedMemoryDepthCfg:
    """Depth-stack configuration for the consolidated memory store."""

    model_dim: int = 512
    visible_layers: int = 9
    hidden_storage_layers: int = 9
    free_hidden_layers: int = 32
    memory_slots_per_layer: int = 64
    num_heads: int = 8
    dropout: float = 0.0
    ffn_mult: int = 2
    depth_manifold_stack: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_CMS_DEPTH_MANIFOLDS)
    visible_manifold_stack: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_CMS_VISIBLE_MANIFOLD_STACK)
    storage_dtype_bits: int = 16
    include_super_product_token: bool = True
    qh_num_depths: int = 8

    def validate(self) -> "ConsolidatedMemoryDepthCfg":
        if self.model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if self.visible_layers != 9:
            raise ValueError("consolidated memory depth stack expects exactly 9 visible layers")
        if self.hidden_storage_layers != 9:
            raise ValueError("consolidated memory depth stack expects exactly 9 hidden storage layers")
        if self.free_hidden_layers < 0:
            raise ValueError("free_hidden_layers must be non-negative")
        if self.memory_slots_per_layer <= 0:
            raise ValueError("memory_slots_per_layer must be positive")
        if len(self.depth_manifold_stack) != 8:
            raise ValueError("depth_manifold_stack must contain exactly 8 depth manifolds")
        if len(self.visible_manifold_stack) != self.visible_layers:
            raise ValueError("visible_manifold_stack must contain exactly 9 entries")
        if self.visible_manifold_stack[-1] != CMS_SUPER_PRODUCT_MANIFOLD:
            raise ValueError("visible_manifold_stack must end with complex_projective_super_product")
        if list(self.visible_manifold_stack[:8]) != list(self.depth_manifold_stack):
            raise ValueError("first 8 visible manifolds must match depth_manifold_stack order")
        missing = [m for m in self.visible_manifold_stack if m not in CMS_MANIFOLD_STORAGE_FACTORS]
        if missing:
            raise ValueError(f"unknown manifold names: {missing}")
        if self.qh_num_depths != len(self.depth_manifold_stack):
            raise ValueError("qh_num_depths must match depth_manifold_stack length (8)")
        return self


class ConsolidatedMemoryDepthStack(nn.Module):
    """Nine-layer visible CMS depth stack with mirrored hidden storage.

    Depth order (layers 1-8):
      euclidean -> hyperbolic -> spatial_s3 -> complex_projective ->
      complex_projective_kahler -> toroidal -> grassmann_subspace ->
      quaternion_spatial_loop

    Layer 9 forms the complex projective super product manifold over the full
    loop stack, maximizing quantum hologram address density in product space.
    """

    def __init__(self, config: Optional[ConsolidatedMemoryDepthCfg] = None):
        super().__init__()
        self.config = (config or ConsolidatedMemoryDepthCfg()).validate()
        dim = int(self.config.model_dim)
        heads = self._resolve_heads(dim, int(self.config.num_heads))
        self.num_heads = heads

        self.visible_memory_slots = nn.Parameter(
            torch.randn(self.config.visible_layers, self.config.memory_slots_per_layer, dim) * 0.02
        )
        self.hidden_memory_slots = nn.Parameter(
            torch.randn(self.config.hidden_storage_layers, self.config.memory_slots_per_layer, dim) * 0.02
        )
        self.depth_link_tokens = nn.Parameter(torch.randn(self.config.visible_layers, dim) * 0.02)
        self.super_product_token = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.loop_closure_token = nn.Parameter(torch.randn(1, dim) * 0.02)

        self.visible_layers = nn.ModuleList(
            [
                TransformerBlock(dim, heads=heads, dropout=self.config.dropout, ffn_mult=self.config.ffn_mult)
                for _ in range(self.config.visible_layers)
            ]
        )
        self.hidden_storage_layers = nn.ModuleList(
            [
                TransformerBlock(dim, heads=heads, dropout=self.config.dropout, ffn_mult=self.config.ffn_mult)
                for _ in range(self.config.hidden_storage_layers)
            ]
        )
        self.free_processor = (
            TransformerStack(
                dim,
                depth=self.config.free_hidden_layers,
                heads=heads,
                dropout=self.config.dropout,
                ffn_mult=self.config.ffn_mult,
                max_len=max(128, 2 * self.config.memory_slots_per_layer + 64),
            )
            if self.config.free_hidden_layers > 0
            else None
        )

        self.loop_attention = MultiScaleAttention(dim, heads, scales=(1, 2, 5, 8))
        self.memory_observer_attention = MultiScaleAttention(dim, heads, scales=(1, 4, 9))
        self.super_product_mixer = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.loop_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1), nn.Sigmoid())
        self.output_norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, dim)
        self.last_context_tokens: Optional[torch.Tensor] = None
        self.last_depth_trace: Dict[str, object] = {}

    @staticmethod
    def _resolve_heads(dim: int, requested: int) -> int:
        requested = max(1, int(requested))
        if dim % requested == 0:
            return requested
        for h in (8, 4, 2, 1):
            if dim % h == 0:
                return h
        return 1

    @property
    def depth_manifold_stack(self) -> Tuple[str, ...]:
        return tuple(self.config.depth_manifold_stack)

    @property
    def visible_manifold_stack(self) -> Tuple[str, ...]:
        return tuple(self.config.visible_manifold_stack)

    def forward(
        self,
        x: torch.Tensor,
        *,
        recall_boost: float = 0.0,
        return_trace: bool = False,
    ):
        self._validate_input(x)
        bsz = x.size(0)
        visible_contexts: List[torch.Tensor] = []
        hidden_contexts: List[torch.Tensor] = []
        visible_trace: List[Dict[str, object]] = []
        hidden_trace: List[Dict[str, object]] = []

        state = x
        depth_layer_contexts: List[torch.Tensor] = []
        for idx, (block, manifold) in enumerate(zip(self.visible_layers, self.visible_manifold_stack)):
            slots = self._expand_slots(self.visible_memory_slots[idx], bsz, x)
            link = self.depth_link_tokens[idx].to(device=x.device, dtype=x.dtype).view(1, 1, -1).expand(bsz, 1, -1)
            tokens = torch.cat([state, slots, link], dim=1)
            tokens, _ = block(tokens)
            state = self._project_to_manifold(tokens[:, : state.size(1), :], manifold, idx)
            slot_view = self._project_to_manifold(
                tokens[:, state.size(1) : state.size(1) + slots.size(1), :],
                manifold,
                idx,
            )
            ctx = slot_view.mean(dim=1)
            visible_contexts.append(ctx)
            if idx < len(self.depth_manifold_stack):
                depth_layer_contexts.append(ctx)
            visible_trace.append(
                {
                    "layer": idx + 1,
                    "manifold": manifold,
                    "storage_factor": CMS_MANIFOLD_STORAGE_FACTORS[manifold],
                    "slot_tokens": int(slots.size(1)),
                    "depth_layer": idx < len(self.depth_manifold_stack),
                }
            )

        hidden_state = state
        for idx, (block, manifold) in enumerate(zip(self.hidden_storage_layers, self.visible_manifold_stack)):
            slots = self._expand_slots(self.hidden_memory_slots[idx], bsz, x)
            visible_ctx = visible_contexts[idx].unsqueeze(1)
            tokens = torch.cat([hidden_state, slots, visible_ctx], dim=1)
            tokens, _ = block(tokens)
            hidden_state = self._project_to_manifold(tokens[:, : hidden_state.size(1), :], manifold, idx)
            slot_view = self._project_to_manifold(
                tokens[:, hidden_state.size(1) : hidden_state.size(1) + slots.size(1), :],
                manifold,
                idx,
            )
            hidden_contexts.append(slot_view.mean(dim=1))
            hidden_trace.append(
                {
                    "layer": idx + 1,
                    "manifold": manifold,
                    "storage_factor": CMS_MANIFOLD_STORAGE_FACTORS[manifold],
                    "mirrors_visible_layer": idx + 1,
                }
            )

        memory_tokens = torch.stack(visible_contexts + hidden_contexts, dim=1)
        link_tokens = self.depth_link_tokens.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(bsz, -1, -1)
        product_token = self._build_super_product_token(memory_tokens, depth_layer_contexts)
        closure = self.loop_closure_token.to(device=x.device, dtype=x.dtype).view(1, 1, -1).expand(bsz, 1, -1)
        attention_tokens = torch.cat([memory_tokens, link_tokens, product_token, closure], dim=1)
        self.last_context_tokens = attention_tokens.detach()

        loop_out, _ = self.loop_attention(hidden_state, attention_tokens, attention_tokens, need_weights=False)
        observed, _ = self.memory_observer_attention(
            loop_out, attention_tokens, attention_tokens, need_weights=False
        )
        boost = 1.0 + float(max(0.0, recall_boost)) * 0.35
        gate = self.loop_gate(torch.cat([hidden_state, observed], dim=-1))
        processed = hidden_state + gate * boost * observed

        if self.free_processor is not None:
            processed, _ = self.free_processor(processed, need_weights=False)
        out = self.output_proj(self.output_norm(processed))

        trace = {
            "trace_type": "consolidated_memory_depth_stack",
            "visible_layers": visible_trace,
            "hidden_storage_layers": hidden_trace,
            "free_hidden_layers": int(self.config.free_hidden_layers),
            "loop_attention_tokens": int(attention_tokens.size(1)),
            "depth_manifold_stack": list(self.depth_manifold_stack),
            "super_product_manifold": CMS_SUPER_PRODUCT_MANIFOLD,
            "qh_num_depths": int(self.config.qh_num_depths),
            "capacity_estimate": self.estimate_storage_capacity(),
        }
        self.last_depth_trace = dict(trace)
        if not return_trace:
            return out
        return out, trace

    @torch.no_grad()
    def build_ltm_context_tokens(
        self,
        query: Optional[torch.Tensor] = None,
        *,
        max_tokens: int = 48,
    ) -> torch.Tensor:
        ref = query if query is not None else self.visible_memory_slots
        device = ref.device
        dtype = ref.dtype if torch.is_floating_point(ref) else self.visible_memory_slots.dtype
        bsz = int(query.size(0)) if query is not None and query.dim() >= 2 else 1
        visible = self.visible_memory_slots.to(device=device, dtype=dtype).mean(dim=1)
        hidden = self.hidden_memory_slots.to(device=device, dtype=dtype).mean(dim=1)
        links = self.depth_link_tokens.to(device=device, dtype=dtype)
        product = self.super_product_token.to(device=device, dtype=dtype)
        closure = self.loop_closure_token.to(device=device, dtype=dtype)
        tokens = torch.cat([visible, hidden, links, product, closure], dim=0)
        tokens = tokens[: max(1, int(max_tokens))]
        return tokens.unsqueeze(0).expand(bsz, -1, -1).detach()

    @torch.no_grad()
    def encode_depth_view(
        self,
        candidate: torch.Tensor,
        *,
        depth_index: int = 0,
    ) -> torch.Tensor:
        """Project a candidate vector onto a specific depth manifold."""
        manifold = self.depth_manifold_stack[int(depth_index) % len(self.depth_manifold_stack)]
        vec = candidate.reshape(1, 1, -1)
        return self._project_to_manifold(vec, manifold, int(depth_index)).reshape(-1)

    def estimate_storage_capacity(self) -> Dict[str, object]:
        cfg = self.config
        per_layer_scalars = int(cfg.memory_slots_per_layer * cfg.model_dim)
        factors = [CMS_MANIFOLD_STORAGE_FACTORS[m] for m in cfg.visible_manifold_stack]
        factor_sum = float(sum(factors))
        pair_factor_sum = float(
            sum(factors[i] * factors[j] for i in range(len(factors)) for j in range(i + 1, len(factors)))
        )
        product_factor = float(math.prod(CMS_MANIFOLD_STORAGE_FACTORS[m] for m in cfg.depth_manifold_stack))
        super_factor = CMS_MANIFOLD_STORAGE_FACTORS[CMS_SUPER_PRODUCT_MANIFOLD]

        visible_units = float(per_layer_scalars * factor_sum)
        hidden_units = float(per_layer_scalars * factor_sum)
        pair_loop_units = float(per_layer_scalars * pair_factor_sum)
        depth_product_units = float(per_layer_scalars * product_factor)
        super_product_units = float(per_layer_scalars * super_factor) if cfg.include_super_product_token else 0.0
        effective_units = visible_units + hidden_units + pair_loop_units + depth_product_units + super_product_units

        physical_slot_scalars = int((cfg.visible_layers + cfg.hidden_storage_layers) * per_layer_scalars)
        physical_slot_bytes = float(physical_slot_scalars * cfg.storage_dtype_bits / 8.0)
        effective_bytes = float(effective_units * cfg.storage_dtype_bits / 8.0)

        return {
            "accounting": "cms_depth_loop_stack_super_product_manifold",
            "normal_physical_slot_scalars": physical_slot_scalars,
            "normal_physical_slot_bytes": physical_slot_bytes,
            "effective_memory_storage_units": effective_units,
            "effective_storage_bytes": effective_bytes,
            "effective_to_physical_ratio": float(effective_units / max(1, physical_slot_scalars)),
            "depth_product_manifold_factor": product_factor,
            "super_product_manifold_factor": super_factor,
            "visible_effective_units": visible_units,
            "hidden_effective_units": hidden_units,
            "pair_loop_effective_units": pair_loop_units,
            "depth_product_effective_units": depth_product_units,
            "super_product_effective_units": super_product_units,
            "depth_manifold_stack": list(cfg.depth_manifold_stack),
            "visible_manifold_stack": list(cfg.visible_manifold_stack),
            "manifold_storage_factors": {m: CMS_MANIFOLD_STORAGE_FACTORS[m] for m in cfg.visible_manifold_stack},
            "qh_num_depths": int(cfg.qh_num_depths),
            "note": "Estimate is addressable scalar-equivalent in product manifold space.",
        }

    def _build_super_product_token(
        self,
        memory_tokens: torch.Tensor,
        depth_layer_contexts: List[torch.Tensor],
    ) -> torch.Tensor:
        if not bool(self.config.include_super_product_token):
            return memory_tokens.mean(dim=1, keepdim=True)
        seed = self.super_product_token.to(device=memory_tokens.device, dtype=memory_tokens.dtype)
        seed = seed.view(1, 1, -1).expand(memory_tokens.size(0), 1, -1)
        if not depth_layer_contexts:
            return F.normalize(seed, dim=-1, eps=1e-8)
        depth_stack = torch.stack(depth_layer_contexts, dim=1)
        depth_mean = depth_stack.mean(dim=1, keepdim=True)
        visible = memory_tokens[:, : self.config.visible_layers, :].mean(dim=1, keepdim=True)
        hidden = memory_tokens[:, self.config.visible_layers :, :].mean(dim=1, keepdim=True)
        mixed = self.super_product_mixer(
            torch.cat([depth_mean, visible, hidden], dim=-1)
        )
        return self._project_to_manifold(
            F.normalize(seed + mixed, dim=-1, eps=1e-8),
            CMS_SUPER_PRODUCT_MANIFOLD,
            self.config.visible_layers - 1,
        )

    @staticmethod
    def _expand_slots(slots: torch.Tensor, batch_size: int, ref: torch.Tensor) -> torch.Tensor:
        return slots.to(device=ref.device, dtype=ref.dtype).unsqueeze(0).expand(batch_size, -1, -1)

    def _project_to_manifold(self, x: torch.Tensor, manifold: str, layer_index: int) -> torch.Tensor:
        if manifold == "euclidean":
            return x
        if manifold == "hyperbolic":
            scale = 0.55 + 0.03 * float(layer_index)
            return torch.tanh(x * scale)
        if manifold == "spatial_s3":
            if x.size(-1) < 4:
                return F.normalize(x, dim=-1, eps=1e-8)
            quat = F.normalize(x[..., :4], dim=-1, eps=1e-8)
            return torch.cat([quat, x[..., 4:]], dim=-1)
        if manifold == "complex_projective":
            if x.size(-1) % 2 != 0:
                x = F.pad(x, (0, 1))
            half = x.size(-1) // 2
            re, im = x[..., :half], x[..., half:]
            mag = torch.sqrt(re * re + im * im + 1e-8)
            return torch.cat([re / mag, im / mag], dim=-1)[..., : x.size(-1)]
        if manifold == "complex_projective_kahler":
            even = x[..., 0::2]
            odd = x[..., 1::2]
            if even.size(-1) != odd.size(-1):
                odd = F.pad(odd, (0, even.size(-1) - odd.size(-1)))
            radius = torch.sqrt(even * even + odd * odd + 1e-8)
            even = even / radius
            odd = odd / radius
            out = torch.empty_like(x)
            out[..., 0::2] = even[..., : out[..., 0::2].size(-1)]
            out[..., 1::2] = odd[..., : out[..., 1::2].size(-1)]
            return out
        if manifold == "toroidal":
            return torch.atan2(torch.sin(x), torch.cos(x))
        if manifold == "grassmann_subspace":
            centered = x - x.mean(dim=-1, keepdim=True)
            return F.normalize(centered, dim=-1, eps=1e-8)
        if manifold == "quaternion_spatial_loop":
            if x.size(-1) < 4:
                return F.normalize(x, dim=-1, eps=1e-8)
            quat = F.normalize(x[..., :4], dim=-1, eps=1e-8)
            residual = torch.tanh(x[..., 4:])
            return torch.cat([quat, residual], dim=-1)
        if manifold == CMS_SUPER_PRODUCT_MANIFOLD:
            if x.size(-1) % 2 != 0:
                x = F.pad(x, (0, 1))
            half = x.size(-1) // 2
            re = x[..., :half]
            im = torch.roll(x[..., half:], shifts=1, dims=-1)
            mag = torch.sqrt(re * re + im * im + 1e-8)
            bound = torch.cat([re / mag, im / mag], dim=-1)
            if x.size(-1) >= 4:
                loop_quat = F.normalize(bound[..., :4], dim=-1, eps=1e-8)
                return torch.cat([loop_quat, bound[..., 4:]], dim=-1)
            return bound
        raise ValueError(f"unknown manifold: {manifold}")

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.dim() != 3:
            raise ValueError("x must be [B,T,D]")
        if x.size(-1) != self.config.model_dim:
            raise ValueError(f"x last dim must be {self.config.model_dim}")
        if not torch.isfinite(x).all():
            raise ValueError("x contains NaN/Inf")


def estimate_consolidated_memory_depth_capacity(
    config: Optional[ConsolidatedMemoryDepthCfg] = None,
) -> Dict[str, object]:
    cfg = (config or ConsolidatedMemoryDepthCfg()).validate()
    return ConsolidatedMemoryDepthStack(cfg).estimate_storage_capacity()


__all__ = [
    "CMS_MANIFOLD_STORAGE_FACTORS",
    "CMS_SUPER_PRODUCT_MANIFOLD",
    "ConsolidatedMemoryDepthCfg",
    "ConsolidatedMemoryDepthStack",
    "DEFAULT_CMS_DEPTH_MANIFOLDS",
    "DEFAULT_CMS_VISIBLE_MANIFOLD_STACK",
    "estimate_consolidated_memory_depth_capacity",
]
