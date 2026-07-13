"""
Plain-language summary
----------------------
What this file is for: Optional loop storing parameter bundles across geometry manifolds.
How it fits in the system: Extra structure beyond ordinary weight tensors.
Status: OPT-IN
Important notes for non-coders: Training writes may stay disabled even when present.
"""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ltm.transformer_utils import TransformerBlock, TransformerStack
from .memory_attention import MultiScaleAttention


DEFAULT_PARAMETER_LOOP_MANIFOLDS: Tuple[str, ...] = (
    "hyperbolic",
    "spatial_s3",
    "euclidean_bridge",
    "complex_projective_kahler",
    "spatial_s3",
    "spherical",
    "grassmann_subspace",
    "toroidal",
    "fisher_rao",
    "quaternion_spatial_loop",
)

MANIFOLD_STORAGE_FACTORS: Dict[str, float] = {
    "hyperbolic": 2.50,
    "spatial_s3": 4.00,
    "euclidean_bridge": 1.25,
    "complex_projective_kahler": 6.00,
    "spherical": 2.00,
    "grassmann_subspace": 5.00,
    "toroidal": 3.00,
    "fisher_rao": 2.75,
    "quaternion_spatial_loop": 8.00,
}


@dataclass(frozen=True)
class ParameterBundleRef:
    """Reference to a live parameter represented in a shadow bundle."""

    name: str
    shape: Tuple[int, ...]
    numel: int
    dtype: str
    requires_grad: bool
    group_key: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "numel": int(self.numel),
            "dtype": self.dtype,
            "requires_grad": bool(self.requires_grad),
            "group_key": self.group_key,
        }


@dataclass(frozen=True)
class ParameterBundleRecord:
    """Metadata for a consolidated referenced-shadow parameter bundle."""

    bundle_id: str
    group_key: str
    refs: Tuple[ParameterBundleRef, ...]
    target_layer: int
    target_slot_start: int
    target_slot_count: int
    total_numel: int
    summary_norm: float
    trigger_step: int
    factors: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "group_key": self.group_key,
            "refs": [ref.to_dict() for ref in self.refs],
            "target_layer": int(self.target_layer),
            "target_slot_start": int(self.target_slot_start),
            "target_slot_count": int(self.target_slot_count),
            "total_numel": int(self.total_numel),
            "summary_norm": float(self.summary_norm),
            "trigger_step": int(self.trigger_step),
            "factors": list(self.factors),
        }


@dataclass(frozen=True)
class ParameterStorageLoopConfig:
    """Configuration for the parameter storage loop stack prototype.

    The module is intentionally additive: it does not mutate external memory,
    activate QSPIN routing, or write to shared slots.
    """

    model_dim: int = 128
    visible_layers: int = 10
    hidden_storage_layers: int = 10
    free_hidden_layers: int = 4
    parameter_slots_per_layer: int = 64
    num_heads: int = 4
    dropout: float = 0.0
    ffn_mult: int = 2
    manifold_stack: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_PARAMETER_LOOP_MANIFOLDS)
    storage_dtype_bits: int = 16
    include_loop_product_token: bool = True
    enable_training_slot_updates: bool = False
    training_update_lr: float = 0.01
    lightbulb_threshold: float = 0.35
    explosive_recall_gain: float = 0.65

    def validate(self) -> "ParameterStorageLoopConfig":
        if self.model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if self.visible_layers != 10:
            raise ValueError("parameter storage loop stack expects exactly 10 visible layers")
        if self.hidden_storage_layers != 10:
            raise ValueError("parameter storage loop stack expects exactly 10 hidden storage layers")
        if self.free_hidden_layers < 0:
            raise ValueError("free_hidden_layers must be non-negative")
        if self.parameter_slots_per_layer <= 0:
            raise ValueError("parameter_slots_per_layer must be positive")
        if self.storage_dtype_bits <= 0:
            raise ValueError("storage_dtype_bits must be positive")
        if self.training_update_lr < 0.0:
            raise ValueError("training_update_lr must be non-negative")
        if self.lightbulb_threshold < 0.0:
            raise ValueError("lightbulb_threshold must be non-negative")
        if self.explosive_recall_gain < 0.0:
            raise ValueError("explosive_recall_gain must be non-negative")
        if len(self.manifold_stack) != self.visible_layers:
            raise ValueError("manifold_stack must contain exactly 10 entries")
        missing = [m for m in self.manifold_stack if m not in MANIFOLD_STORAGE_FACTORS]
        if missing:
            raise ValueError(f"unknown manifold names: {missing}")
        return self


class ParameterStorageLoopStack(nn.Module):
    """Curved parameter-store prototype with visible/hidden manifold loops.

    The stack treats parameter slots as addressable manifold tokens. The visible
    stack stores observed parameter structure; the mirrored hidden stack stores
    usage/control structure; free hidden transformer layers process both through
    a special loop attention path.
    """

    def __init__(self, config: Optional[ParameterStorageLoopConfig] = None):
        super().__init__()
        self.config = (config or ParameterStorageLoopConfig()).validate()
        dim = int(self.config.model_dim)
        heads = self._resolve_heads(dim, int(self.config.num_heads))
        self.num_heads = heads

        self.visible_parameter_slots = nn.Parameter(
            torch.randn(self.config.visible_layers, self.config.parameter_slots_per_layer, dim) * 0.02
        )
        self.hidden_parameter_slots = nn.Parameter(
            torch.randn(self.config.hidden_storage_layers, self.config.parameter_slots_per_layer, dim) * 0.02
        )
        self.loop_link_tokens = nn.Parameter(torch.randn(self.config.visible_layers, dim) * 0.02)
        self.product_manifold_token = nn.Parameter(torch.randn(1, dim) * 0.02)

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
                max_len=max(64, 2 * self.config.parameter_slots_per_layer + 32),
            )
            if self.config.free_hidden_layers > 0
            else None
        )

        self.loop_attention = MultiScaleAttention(dim, heads, scales=(1, 2, 5))
        self.parameter_observer_attention = MultiScaleAttention(dim, heads, scales=(1, 4))
        self.loop_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1), nn.Sigmoid())
        self.output_norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, dim)
        self.register_buffer("last_lightbulb_intensity", torch.tensor(0.0))
        self.last_context_tokens: Optional[torch.Tensor] = None
        self.last_slot_update_trace: Dict[str, object] = {}
        self.parameter_bundle_registry: Dict[str, ParameterBundleRecord] = {}
        self.last_parameter_consolidation_trace: Dict[str, Any] = {}

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
    def manifold_stack(self) -> Tuple[str, ...]:
        return tuple(self.config.manifold_stack)

    def forward(
        self,
        x: torch.Tensor,
        *,
        fire_mask: Optional[torch.Tensor] = None,
        recall_boost: float = 0.0,
        allow_slot_update: bool = False,
        slot_update_scale: float = 1.0,
        return_trace: bool = False,
    ):
        self._validate_input(x)
        bsz = x.size(0)
        visible_contexts: List[torch.Tensor] = []
        hidden_contexts: List[torch.Tensor] = []
        visible_trace: List[Dict[str, object]] = []
        hidden_trace: List[Dict[str, object]] = []

        state = x
        for idx, (block, manifold) in enumerate(zip(self.visible_layers, self.manifold_stack)):
            slots = self._expand_slots(self.visible_parameter_slots[idx], bsz, x)
            link = self.loop_link_tokens[idx].to(device=x.device, dtype=x.dtype).view(1, 1, -1).expand(bsz, 1, -1)
            tokens = torch.cat([state, slots, link], dim=1)
            tokens, _ = block(tokens)
            state = self._project_to_manifold(tokens[:, : state.size(1), :], manifold, idx)
            slot_view = self._project_to_manifold(tokens[:, state.size(1) : state.size(1) + slots.size(1), :], manifold, idx)
            visible_contexts.append(slot_view.mean(dim=1))
            visible_trace.append(
                {
                    "layer": idx + 1,
                    "manifold": manifold,
                    "storage_factor": MANIFOLD_STORAGE_FACTORS[manifold],
                    "slot_tokens": int(slots.size(1)),
                }
            )

        hidden_state = state
        for idx, (block, manifold) in enumerate(zip(self.hidden_storage_layers, self.manifold_stack)):
            slots = self._expand_slots(self.hidden_parameter_slots[idx], bsz, x)
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
                    "storage_factor": MANIFOLD_STORAGE_FACTORS[manifold],
                    "mirrors_visible_layer": idx + 1,
                }
            )

        memory_tokens = torch.stack(visible_contexts + hidden_contexts, dim=1)
        link_tokens = self.loop_link_tokens.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(bsz, -1, -1)
        product_token = self._build_product_token(memory_tokens)
        attention_tokens = torch.cat([memory_tokens, link_tokens, product_token], dim=1)
        self.last_context_tokens = attention_tokens.detach()

        loop_out, _ = self.loop_attention(hidden_state, attention_tokens, attention_tokens, need_weights=False)
        observed, _ = self.parameter_observer_attention(loop_out, attention_tokens, attention_tokens, need_weights=False)
        auto_fire, intensity = self.detect_lightbulb_moment(hidden_state, product_token)
        effective_fire = self._resolve_fire_mask(fire_mask, auto_fire)
        boost = self._recall_multiplier(effective_fire, recall_boost)
        gate = self.loop_gate(torch.cat([hidden_state, observed], dim=-1))
        processed = hidden_state + gate * boost * observed

        if self.free_processor is not None:
            processed, _ = self.free_processor(processed, need_weights=False)
        out = self.output_proj(self.output_norm(processed))
        slot_update_trace = self.update_parameter_slots(
            x,
            allow_update=bool(allow_slot_update),
            write_scale=float(slot_update_scale),
        )

        if not return_trace:
            return out
        return out, {
            "trace_type": "parameter_storage_loop_stack",
            "visible_layers": visible_trace,
            "hidden_storage_layers": hidden_trace,
            "free_hidden_layers": int(self.config.free_hidden_layers),
            "loop_attention_tokens": int(attention_tokens.size(1)),
            "parameter_observer_attention": True,
            "product_manifold_token": bool(self.config.include_loop_product_token),
            "lightbulb": {
                "triggered": bool(effective_fire.any().item()),
                "auto_triggered": bool(auto_fire.any().item()),
                "intensity_mean": float(intensity.mean().detach().item()),
                "recall_boost": float(recall_boost),
                "explosive_recall_gain": float(self.config.explosive_recall_gain),
            },
            "slot_update": slot_update_trace,
            "capacity_estimate": self.estimate_storage_capacity(),
            "safety": {
                "prototype": True,
                "external_memory_write": False,
                "shared_slot_write": False,
                "qspin_runtime_activation": False,
                "training_slot_update_requires_opt_in": True,
            },
        }

    @torch.no_grad()
    def build_ltm_context_tokens(
        self,
        query: Optional[torch.Tensor] = None,
        *,
        max_tokens: int = 32,
    ) -> torch.Tensor:
        """Build bounded read-only context tokens for LTM attention."""
        ref = query if query is not None else self.visible_parameter_slots
        device = ref.device
        dtype = ref.dtype if torch.is_floating_point(ref) else self.visible_parameter_slots.dtype
        bsz = int(query.size(0)) if query is not None and query.dim() >= 2 else 1
        visible = self.visible_parameter_slots.to(device=device, dtype=dtype).mean(dim=1)
        hidden = self.hidden_parameter_slots.to(device=device, dtype=dtype).mean(dim=1)
        links = self.loop_link_tokens.to(device=device, dtype=dtype)
        seed = self.product_manifold_token.to(device=device, dtype=dtype)
        product = F.normalize(seed + (visible.mean(dim=0, keepdim=True) * hidden.mean(dim=0, keepdim=True)), dim=-1, eps=1e-8)
        tokens = torch.cat([visible, hidden, links, product], dim=0)
        tokens = tokens[: max(1, int(max_tokens))]
        return tokens.unsqueeze(0).expand(bsz, -1, -1).detach()

    @torch.no_grad()
    def read_parameter_summary(self, query: torch.Tensor, *, top_k: int = 8, return_trace: bool = False):
        self._validate_input(query)
        ctx = self.build_ltm_context_tokens(query, max_tokens=max(1, int(top_k)))
        q = F.normalize(query.mean(dim=1), dim=-1, eps=1e-8)
        k = F.normalize(ctx, dim=-1, eps=1e-8)
        scores = torch.einsum("bd,btd->bt", q, k)
        weights = torch.softmax(scores, dim=-1)
        out = torch.sum(weights.unsqueeze(-1) * ctx, dim=1)
        trace = {
            "trace_type": "parameter_loop_read_only",
            "top_k": int(ctx.size(1)),
            "scores_mean": float(scores.mean().detach().item()),
            "read_only": True,
            "no_memory_store_mutation": True,
        }
        if return_trace:
            return out, trace
        return out

    @torch.no_grad()
    def consolidate_parameter_bundles(
        self,
        named_parameters: Iterable[Tuple[str, nn.Parameter]],
        *,
        trigger_state: Optional[Dict[str, Any]] = None,
        max_bundles: int = 4,
        min_params_per_bundle: int = 1,
        min_total_numel: int = 1,
        include_filters: Optional[Sequence[str]] = None,
        exclude_filters: Optional[Sequence[str]] = None,
        write_scale: float = 1.0,
    ) -> Dict[str, Any]:
        """Copy grouped parameter summaries into loop slots and retain refs.

        This is a referenced-shadow transplant: original nn.Parameter objects
        remain owned by their modules and optimizers, while bounded summaries are
        stored in the loop's manifold slot space with stable provenance refs.
        """

        groups = self._group_named_parameters(
            named_parameters,
            include_filters=include_filters,
            exclude_filters=exclude_filters,
        )
        trigger_state = dict(trigger_state or {})
        trigger_step = int(trigger_state.get("step", 0))
        candidates = []
        for group_key, items in groups.items():
            total_numel = int(sum(param.numel() for _name, param in items))
            if len(items) < int(max(1, min_params_per_bundle)):
                continue
            if total_numel < int(max(1, min_total_numel)):
                continue
            candidates.append((group_key, items, total_numel))
        candidates.sort(key=lambda item: (-item[2], item[0]))
        candidates = candidates[: int(max(1, max_bundles))]

        lr = max(0.0, min(1.0, float(write_scale))) * float(self.config.training_update_lr or 0.01)
        lr = max(0.0, min(0.25, lr))
        created: List[ParameterBundleRecord] = []
        for offset, (group_key, items, total_numel) in enumerate(candidates):
            summary = self._bundle_summary_vector(items).to(
                device=self.visible_parameter_slots.device,
                dtype=self.visible_parameter_slots.dtype,
            )
            layer_idx = (len(self.parameter_bundle_registry) + offset) % int(self.config.visible_layers)
            slot_idx = (len(self.parameter_bundle_registry) + offset) % int(self.config.parameter_slots_per_layer)
            visible_target = summary.view(-1)
            hidden_target = torch.tanh(summary).view(-1)
            self.visible_parameter_slots[layer_idx, slot_idx].mul_(1.0 - lr).add_(visible_target, alpha=lr)
            self.hidden_parameter_slots[layer_idx, slot_idx].mul_(1.0 - lr).add_(hidden_target, alpha=lr)
            refs = tuple(
                ParameterBundleRef(
                    name=name,
                    shape=tuple(int(v) for v in param.shape),
                    numel=int(param.numel()),
                    dtype=str(param.dtype).replace("torch.", ""),
                    requires_grad=bool(param.requires_grad),
                    group_key=group_key,
                )
                for name, param in items
            )
            bundle_id = self._bundle_id(group_key, refs, trigger_step)
            record = ParameterBundleRecord(
                bundle_id=bundle_id,
                group_key=group_key,
                refs=refs,
                target_layer=int(layer_idx),
                target_slot_start=int(slot_idx),
                target_slot_count=1,
                total_numel=int(total_numel),
                summary_norm=float(summary.norm().detach().item()),
                trigger_step=int(trigger_step),
                factors=tuple(str(v) for v in trigger_state.get("factors", ("training_threshold",))),
            )
            self.parameter_bundle_registry[bundle_id] = record
            created.append(record)

        trace = {
            "trace_type": "parameter_bundle_consolidation",
            "mode": "referenced_shadow",
            "trigger_state": trigger_state,
            "candidate_groups": int(len(groups)),
            "created_bundle_count": int(len(created)),
            "created_bundles": [record.to_dict() for record in created],
            "registry_size": int(len(self.parameter_bundle_registry)),
            "optimizer_safe": True,
            "original_parameters_replaced": False,
            "original_parameters_frozen": False,
            "write_scale": float(write_scale),
            "slot_lr": float(lr),
        }
        self.last_parameter_consolidation_trace = trace
        return trace

    def parameter_bundle_registry_state(self) -> Dict[str, Any]:
        return {
            "registry_size": int(len(self.parameter_bundle_registry)),
            "bundles": {
                bundle_id: record.to_dict()
                for bundle_id, record in sorted(self.parameter_bundle_registry.items())
            },
        }

    def load_parameter_bundle_registry_state(self, state: Optional[Dict[str, Any]]) -> None:
        self.parameter_bundle_registry = {}
        if not state:
            return
        for bundle_id, payload in (state.get("bundles") or {}).items():
            refs = tuple(
                ParameterBundleRef(
                    name=str(ref.get("name", "")),
                    shape=tuple(int(v) for v in ref.get("shape", [])),
                    numel=int(ref.get("numel", 0)),
                    dtype=str(ref.get("dtype", "unknown")),
                    requires_grad=bool(ref.get("requires_grad", True)),
                    group_key=str(ref.get("group_key", payload.get("group_key", "unknown"))),
                )
                for ref in payload.get("refs", [])
            )
            self.parameter_bundle_registry[str(bundle_id)] = ParameterBundleRecord(
                bundle_id=str(payload.get("bundle_id", bundle_id)),
                group_key=str(payload.get("group_key", "unknown")),
                refs=refs,
                target_layer=int(payload.get("target_layer", 0)),
                target_slot_start=int(payload.get("target_slot_start", 0)),
                target_slot_count=int(payload.get("target_slot_count", 1)),
                total_numel=int(payload.get("total_numel", 0)),
                summary_norm=float(payload.get("summary_norm", 0.0)),
                trigger_step=int(payload.get("trigger_step", 0)),
                factors=tuple(str(v) for v in payload.get("factors", [])),
            )

    def detect_lightbulb_moment(self, hidden_state: torch.Tensor, product_token: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = F.normalize(hidden_state.mean(dim=1), dim=-1, eps=1e-8)
        product = F.normalize(product_token.squeeze(1), dim=-1, eps=1e-8)
        novelty = (1.0 - (pooled * product).sum(dim=-1)).clamp(0.0, 2.0) * 0.5
        energy = torch.tanh(hidden_state.std(dim=1).mean(dim=-1)).clamp_min(0.0)
        intensity = (0.65 * novelty + 0.35 * energy).clamp(0.0, 1.0)
        fire = intensity > float(self.config.lightbulb_threshold)
        self.last_lightbulb_intensity.copy_(intensity.mean().detach())
        return fire, intensity

    @torch.no_grad()
    def update_parameter_slots(
        self,
        observations: torch.Tensor,
        *,
        allow_update: bool = False,
        write_scale: float = 1.0,
    ) -> Dict[str, object]:
        if not (self.training and bool(self.config.enable_training_slot_updates) and bool(allow_update)):
            self.last_slot_update_trace = {
                "updated": False,
                "reason": "disabled_or_not_training",
                "requires_training": True,
                "requires_enable_training_slot_updates": True,
                "requires_allow_update": True,
            }
            return dict(self.last_slot_update_trace)
        self._validate_input(observations)
        obs = observations.mean(dim=(0, 1)).to(device=self.visible_parameter_slots.device, dtype=self.visible_parameter_slots.dtype)
        lr = float(self.config.training_update_lr) * float(max(0.0, write_scale))
        lr = max(0.0, min(0.25, lr))
        if lr <= 0.0:
            self.last_slot_update_trace = {"updated": False, "reason": "zero_lr"}
            return dict(self.last_slot_update_trace)
        target_visible = obs.view(1, 1, -1).expand_as(self.visible_parameter_slots)
        target_hidden = torch.tanh(obs).view(1, 1, -1).expand_as(self.hidden_parameter_slots)
        self.visible_parameter_slots.mul_(1.0 - lr).add_(target_visible, alpha=lr)
        self.hidden_parameter_slots.mul_(1.0 - lr).add_(target_hidden, alpha=lr)
        self.last_slot_update_trace = {
            "updated": True,
            "write_scale": float(write_scale),
            "lr": float(lr),
            "updated_tensors": ["visible_parameter_slots", "hidden_parameter_slots"],
            "external_memory_write": False,
            "shared_slot_write": False,
        }
        return dict(self.last_slot_update_trace)

    def estimate_storage_capacity(self) -> Dict[str, object]:
        """Estimate storage using loop-stack/product-manifold accounting.

        This is deliberately not a normal trainable-parameter count. It treats
        each visible and mirrored hidden storage layer as a manifold-addressed
        parameter field, then adds pairwise loop links and a product manifold
        term created by the final quaternion spatial loop.
        """

        cfg = self.config
        per_layer_scalars = int(cfg.parameter_slots_per_layer * cfg.model_dim)
        factors = [MANIFOLD_STORAGE_FACTORS[m] for m in cfg.manifold_stack]
        factor_sum = float(sum(factors))
        pair_factor_sum = float(sum(factors[i] * factors[j] for i in range(len(factors)) for j in range(i + 1, len(factors))))
        product_factor = float(math.prod(factors))

        visible_units = float(per_layer_scalars * factor_sum)
        hidden_units = float(per_layer_scalars * factor_sum)
        pair_loop_units = float(per_layer_scalars * pair_factor_sum)
        product_units = float(per_layer_scalars * product_factor) if cfg.include_loop_product_token else 0.0
        effective_units = visible_units + hidden_units + pair_loop_units + product_units

        physical_slot_scalars = int((cfg.visible_layers + cfg.hidden_storage_layers) * per_layer_scalars)
        physical_slot_bytes = float(physical_slot_scalars * cfg.storage_dtype_bits / 8.0)
        effective_bytes = float(effective_units * cfg.storage_dtype_bits / 8.0)
        compression_ratio = float(effective_units / max(1, physical_slot_scalars))

        return {
            "accounting": "loop_stack_product_manifold_effective_storage",
            "normal_physical_slot_scalars": physical_slot_scalars,
            "normal_physical_slot_bytes": physical_slot_bytes,
            "normal_physical_slot_mebibytes": physical_slot_bytes / (1024.0 * 1024.0),
            "effective_parameter_storage_units": effective_units,
            "effective_storage_bytes": effective_bytes,
            "effective_storage_mebibytes": effective_bytes / (1024.0 * 1024.0),
            "effective_storage_gibibytes": effective_bytes / (1024.0**3),
            "effective_to_physical_ratio": compression_ratio,
            "per_layer_scalar_slots": per_layer_scalars,
            "visible_effective_units": visible_units,
            "hidden_effective_units": hidden_units,
            "pair_loop_effective_units": pair_loop_units,
            "product_manifold_effective_units": product_units,
            "product_manifold_factor": product_factor,
            "pair_loop_factor_sum": pair_factor_sum,
            "manifold_stack": list(cfg.manifold_stack),
            "manifold_storage_factors": {m: MANIFOLD_STORAGE_FACTORS[m] for m in cfg.manifold_stack},
            "note": "Estimate is an addressable/product-manifold scalar-equivalent, not a literal trainable parameter count.",
        }

    def _build_product_token(self, memory_tokens: torch.Tensor) -> torch.Tensor:
        if not bool(self.config.include_loop_product_token):
            return memory_tokens.mean(dim=1, keepdim=True)
        seed = self.product_manifold_token.to(device=memory_tokens.device, dtype=memory_tokens.dtype)
        seed = seed.view(1, 1, -1).expand(memory_tokens.size(0), 1, -1)
        visible = memory_tokens[:, : self.config.visible_layers, :].mean(dim=1, keepdim=True)
        hidden = memory_tokens[:, self.config.visible_layers :, :].mean(dim=1, keepdim=True)
        return F.normalize(seed + visible * hidden, dim=-1, eps=1e-8)

    def _resolve_fire_mask(self, fire_mask: Optional[torch.Tensor], auto_fire: torch.Tensor) -> torch.Tensor:
        if fire_mask is None:
            return auto_fire
        mask = torch.as_tensor(fire_mask, device=auto_fire.device)
        if mask.dim() > 1:
            mask = mask.reshape(mask.size(0), -1).any(dim=-1)
        return mask.to(dtype=torch.bool) | auto_fire

    def _recall_multiplier(self, fire_mask: torch.Tensor, recall_boost: float) -> torch.Tensor:
        boost = float(max(0.0, recall_boost)) * float(self.config.explosive_recall_gain)
        if boost <= 0.0:
            return torch.ones(fire_mask.size(0), 1, 1, device=fire_mask.device)
        return 1.0 + fire_mask.to(dtype=self.visible_parameter_slots.dtype).view(-1, 1, 1) * boost

    def _group_named_parameters(
        self,
        named_parameters: Iterable[Tuple[str, nn.Parameter]],
        *,
        include_filters: Optional[Sequence[str]] = None,
        exclude_filters: Optional[Sequence[str]] = None,
    ) -> Dict[str, List[Tuple[str, nn.Parameter]]]:
        include = tuple(str(v) for v in (include_filters or ()))
        exclude = tuple(str(v) for v in (exclude_filters or ()))
        groups: Dict[str, List[Tuple[str, nn.Parameter]]] = {}
        for name, param in named_parameters:
            if not isinstance(param, nn.Parameter):
                continue
            if not bool(param.requires_grad):
                continue
            if "parameter_storage_loop_stack" in str(name):
                continue
            if include and not any(token in name for token in include):
                continue
            if exclude and any(token in name for token in exclude):
                continue
            key = self._parameter_group_key(name, param)
            groups.setdefault(key, []).append((name, param))
        return groups

    @staticmethod
    def _parameter_group_key(name: str, param: nn.Parameter) -> str:
        parts = str(name).split(".")
        family = parts[0] if parts else "root"
        role = parts[-1] if parts else "param"
        if "weight" in role:
            role = "weight"
        elif "bias" in role:
            role = "bias"
        elif "norm" in str(name).lower():
            role = "norm"
        rank = len(tuple(param.shape))
        numel = int(param.numel())
        if numel >= 1_000_000:
            bucket = "xl"
        elif numel >= 100_000:
            bucket = "large"
        elif numel >= 10_000:
            bucket = "medium"
        else:
            bucket = "small"
        return f"{family}:{role}:rank{rank}:{bucket}"

    def _bundle_summary_vector(self, items: Sequence[Tuple[str, nn.Parameter]]) -> torch.Tensor:
        features: List[float] = []
        for name, param in items:
            data = param.detach().float().reshape(-1)
            if data.numel() == 0:
                continue
            grad = param.grad.detach().float().reshape(-1) if param.grad is not None else None
            features.extend(
                [
                    float(data.mean().item()),
                    float(data.std(unbiased=False).item()),
                    float(data.norm().item() / max(1, data.numel())),
                    float(data.abs().max().item()),
                    float(data.numel()),
                    0.0 if grad is None else float(grad.norm().item() / max(1, grad.numel())),
                    float((int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16) % 1000) / 1000.0),
                ]
            )
        if not features:
            features = [0.0]
        vec = torch.tensor(features, dtype=self.visible_parameter_slots.dtype, device=self.visible_parameter_slots.device)
        dim = int(self.config.model_dim)
        if vec.numel() < dim:
            repeat = int(math.ceil(dim / max(1, vec.numel())))
            vec = vec.repeat(repeat)
        vec = vec[:dim]
        return torch.tanh(vec)

    @staticmethod
    def _bundle_id(group_key: str, refs: Sequence[ParameterBundleRef], trigger_step: int) -> str:
        payload = "|".join([group_key, str(trigger_step)] + [ref.name for ref in refs])
        return "pbl-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _expand_slots(slots: torch.Tensor, batch_size: int, ref: torch.Tensor) -> torch.Tensor:
        return slots.to(device=ref.device, dtype=ref.dtype).unsqueeze(0).expand(batch_size, -1, -1)

    def _project_to_manifold(self, x: torch.Tensor, manifold: str, layer_index: int) -> torch.Tensor:
        if manifold == "hyperbolic":
            scale = 0.55 + 0.03 * float(layer_index)
            return torch.tanh(x * scale)
        if manifold == "spatial_s3":
            if x.size(-1) < 4:
                return F.normalize(x, dim=-1, eps=1e-8)
            quat = F.normalize(x[..., :4], dim=-1, eps=1e-8)
            return torch.cat([quat, x[..., 4:]], dim=-1)
        if manifold == "euclidean_bridge":
            return x
        if manifold == "complex_projective_kahler":
            even = x[..., 0::2]
            odd = x[..., 1::2]
            if even.size(-1) != odd.size(-1):
                odd = F.pad(odd, (0, even.size(-1) - odd.size(-1)))
            radius = torch.sqrt(even * even + odd * odd).clamp_min(1e-8)
            even = even / radius
            odd = odd / radius
            out = torch.empty_like(x)
            out[..., 0::2] = even[..., : out[..., 0::2].size(-1)]
            out[..., 1::2] = odd[..., : out[..., 1::2].size(-1)]
            return out
        if manifold == "spherical":
            return F.normalize(x, dim=-1, eps=1e-8)
        if manifold == "grassmann_subspace":
            centered = x - x.mean(dim=-1, keepdim=True)
            return F.normalize(centered, dim=-1, eps=1e-8)
        if manifold == "toroidal":
            return torch.atan2(torch.sin(x), torch.cos(x))
        if manifold == "fisher_rao":
            probs = torch.softmax(x, dim=-1)
            return torch.sqrt(probs.clamp_min(1e-8))
        if manifold == "quaternion_spatial_loop":
            if x.size(-1) < 4:
                return F.normalize(x, dim=-1, eps=1e-8)
            quat = F.normalize(x[..., :4], dim=-1, eps=1e-8)
            residual = torch.tanh(x[..., 4:])
            return torch.cat([quat, residual], dim=-1)
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


def estimate_parameter_storage_loop_capacity(
    config: Optional[ParameterStorageLoopConfig] = None,
) -> Dict[str, object]:
    """Convenience estimator without constructing a full model graph."""

    cfg = (config or ParameterStorageLoopConfig()).validate()
    return ParameterStorageLoopStack(cfg).estimate_storage_capacity()


__all__ = [
    "DEFAULT_PARAMETER_LOOP_MANIFOLDS",
    "MANIFOLD_STORAGE_FACTORS",
    "ParameterStorageLoopConfig",
    "ParameterStorageLoopStack",
    "estimate_parameter_storage_loop_capacity",
]
