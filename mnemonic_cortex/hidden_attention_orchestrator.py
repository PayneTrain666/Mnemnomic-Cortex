"""
Plain-language summary
----------------------
What this file is for: Collects and routes hidden activations across many modules.
How it fits in the system: Gives a global view of internal signals for attention / diagnostics.
Status: ACTIVE in full stacks
Important notes for non-coders: Important for hidden-attention training tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .memory_attention import MultiScaleAttention


@dataclass
class HiddenAttentionConfig:
    model_dim: int
    num_heads: int = 8
    attention_type: str = "multiscale"
    transformer_layers: int = 2
    max_captured_layers: int = 192
    capture_every_n: int = 1
    detach_captured: bool = True
    include_parameter_tokens: bool = True
    max_parameter_tokens: int = 48
    enable_context_cross_attention: bool = True


class HiddenAttentionOrchestrator(nn.Module):
    """
    Global hidden-layer attention interface.

    Captures hidden activations from registered submodules, converts them into
    tokens, and lets caller query them with attention for sequence refinement.
    """

    _DEFAULT_CAPTURE_TYPES = (
        nn.Linear,
        nn.MultiheadAttention,
        nn.TransformerEncoderLayer,
        nn.TransformerDecoderLayer,
        nn.Conv1d,
        nn.Conv2d,
        nn.GRU,
        nn.LSTM,
    )

    def __init__(self, config: HiddenAttentionConfig):
        super().__init__()
        self.config = config
        self.model_dim = int(config.model_dim)
        self.num_heads = self._resolve_heads(self.model_dim, int(max(1, config.num_heads)))
        self.self_attn = self._make_attention()
        self.hidden_attn = self._make_attention()
        self.context_attn = self._make_attention() if bool(config.enable_context_cross_attention) else None
        self.norm = nn.LayerNorm(self.model_dim)
        self.output_norm = nn.LayerNorm(self.model_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.model_dim,
                nhead=self.num_heads,
                dim_feedforward=max(128, self.model_dim * 4),
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=max(1, int(config.transformer_layers)),
        )
        self.merge_gate = nn.Parameter(torch.tensor(0.5))
        self.query_hidden_mix = nn.Parameter(torch.tensor([0.45, 0.55]))

        self.param_stat_proj = nn.Sequential(
            nn.Linear(6, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.param_token_seed = nn.Parameter(
            torch.randn(max(1, int(config.max_parameter_tokens)), self.model_dim) * 0.02
        )
        self.param_token_norm = nn.LayerNorm(self.model_dim)

        self._dim_projectors = nn.ModuleDict()
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._tracked_modules: List[Tuple[str, nn.Module]] = []
        self._registered_layer_names = set()
        self._captured: List[Tuple[str, torch.Tensor]] = []
        self._capture_step = 0
        self._capture_depth = 0
        self.last_stats: Dict[str, float] = {}

    @staticmethod
    def _resolve_heads(dim: int, heads: int) -> int:
        if dim % heads == 0:
            return heads
        for h in (8, 4, 2, 1):
            if dim % h == 0:
                return h
        return 1

    def _make_attention(self) -> nn.Module:
        if str(self.config.attention_type).strip().lower() == "multiscale":
            return MultiScaleAttention(self.model_dim, self.num_heads)
        return nn.MultiheadAttention(self.model_dim, num_heads=self.num_heads, batch_first=True)

    @staticmethod
    def _extract_tensor(payload) -> Optional[torch.Tensor]:
        if isinstance(payload, torch.Tensor):
            return payload
        if isinstance(payload, (list, tuple)):
            for item in payload:
                t = HiddenAttentionOrchestrator._extract_tensor(item)
                if t is not None:
                    return t
        if isinstance(payload, dict):
            for item in payload.values():
                t = HiddenAttentionOrchestrator._extract_tensor(item)
                if t is not None:
                    return t
        return None

    @staticmethod
    def _pool_hidden_tensor(t: torch.Tensor) -> Optional[torch.Tensor]:
        if t.numel() == 0:
            return None
        if t.dim() == 1:
            return t.unsqueeze(0)
        if t.dim() == 2:
            return t
        if t.dim() == 3:
            return t.mean(dim=1)
        bsz = t.size(0)
        return t.reshape(bsz, -1, t.size(-1)).mean(dim=1)

    def _project_to_model_dim(self, t: torch.Tensor) -> torch.Tensor:
        if t.size(-1) == self.model_dim:
            return t
        key = str(int(t.size(-1)))
        if key not in self._dim_projectors:
            self._dim_projectors[key] = nn.Linear(int(t.size(-1)), self.model_dim)
        proj = self._dim_projectors[key].to(device=t.device, dtype=t.dtype)
        return proj(t)

    def _make_hook(self, layer_name: str):
        def _hook(_module, _inputs, output):
            if self._capture_depth <= 0:
                return
            every_n = max(1, int(self.config.capture_every_n))
            if every_n > 1 and (self._capture_step % every_n) != 0:
                return
            t = self._extract_tensor(output)
            if t is None:
                return
            pooled = self._pool_hidden_tensor(t)
            if pooled is None:
                return
            pooled = self._project_to_model_dim(pooled)
            if bool(self.config.detach_captured):
                pooled = pooled.detach()
            self._captured.append((layer_name, pooled))
            limit = max(1, int(self.config.max_captured_layers))
            if len(self._captured) > limit:
                self._captured = self._captured[-limit:]

        return _hook

    def clear_registered_sources(self) -> None:
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()
        self._tracked_modules.clear()
        self._registered_layer_names.clear()

    def register_source(
        self,
        module: nn.Module,
        *,
        source_name: str = "",
        capture_types: Optional[Sequence[type]] = None,
    ) -> int:
        if module is None:
            return 0
        capture_t = tuple(capture_types) if capture_types is not None else self._DEFAULT_CAPTURE_TYPES
        added = 0
        for name, sub in module.named_modules():
            if sub is module:
                continue
            if not isinstance(sub, capture_t):
                continue
            fq_name = f"{source_name}.{name}" if source_name else name
            if fq_name in self._registered_layer_names:
                continue
            handle = sub.register_forward_hook(self._make_hook(fq_name))
            self._hooks.append(handle)
            self._tracked_modules.append((fq_name, sub))
            self._registered_layer_names.add(fq_name)
            added += 1
        return added

    def begin_capture(self) -> None:
        if self._capture_depth == 0:
            self._captured.clear()
            self._capture_step += 1
        self._capture_depth += 1

    def end_capture(self) -> None:
        self._capture_depth = max(0, self._capture_depth - 1)

    @staticmethod
    def _module_parameter_signature(module: nn.Module, device, dtype) -> torch.Tensor:
        means = []
        abs_means = []
        sq_means = []
        n_params = 0.0
        n_tensors = 0.0
        with torch.no_grad():
            for p in module.parameters():
                if p.numel() == 0:
                    continue
                t = p.detach().to(device=device)
                if torch.is_complex(t):
                    t_real = t.real.to(dtype=dtype)
                    t_abs = t.abs().to(dtype=dtype)
                else:
                    t_real = t.to(dtype=dtype)
                    t_abs = t_real.abs()
                means.append(t_real.mean())
                abs_means.append(t_abs.mean())
                sq_means.append((t_abs * t_abs).mean())
                n_params += float(t.numel())
                n_tensors += 1.0
        if not means:
            return torch.zeros(6, device=device, dtype=dtype)
        mean = torch.stack(means).mean()
        abs_mean = torch.stack(abs_means).mean()
        sq_mean = torch.stack(sq_means).mean()
        std_like = (sq_mean - mean * mean).clamp_min(0.0).sqrt()
        l2_like = sq_mean.clamp_min(0.0).sqrt()
        log_params = torch.log(torch.tensor(n_params + 1.0, device=device, dtype=dtype))
        tensor_count = torch.tensor(n_tensors, device=device, dtype=dtype)
        return torch.stack([mean, abs_mean, std_like, l2_like, log_params, tensor_count], dim=0)

    def _build_parameter_tokens(self, batch_size: int, ref: torch.Tensor) -> Optional[torch.Tensor]:
        if not bool(self.config.include_parameter_tokens):
            return None
        if not self._tracked_modules:
            return None
        cap = max(1, int(self.config.max_parameter_tokens))
        stats = []
        names = []
        for name, module in self._tracked_modules[:cap]:
            stats.append(self._module_parameter_signature(module, device=ref.device, dtype=ref.dtype))
            names.append(name)
        if not stats:
            return None
        stat_t = torch.stack(stats, dim=0)
        stat_embed = self.param_stat_proj(stat_t)
        seed = self.param_token_seed[: stat_t.size(0)].to(device=ref.device, dtype=ref.dtype)
        tokens = self.param_token_norm(seed + stat_embed).unsqueeze(0).expand(batch_size, -1, -1)
        self.last_stats["parameter_token_count"] = float(stat_t.size(0))
        self.last_stats["parameter_signature_scale"] = float(stat_t.abs().mean().detach().item())
        self.last_stats["tracked_layer_count"] = float(len(names))
        return tokens

    def _align_to_batch(self, t: torch.Tensor, batch_size: int, ref: torch.Tensor) -> torch.Tensor:
        out = t.to(device=ref.device, dtype=ref.dtype)
        if out.size(0) == batch_size:
            return out
        if out.size(0) == 1 and batch_size > 1:
            return out.expand(batch_size, -1)
        return out.mean(dim=0, keepdim=True).expand(batch_size, -1)

    def _build_hidden_tokens(self, query: torch.Tensor) -> Optional[torch.Tensor]:
        if not self._captured:
            return None
        batch_size = query.size(0)
        hidden = []
        for _, h in self._captured:
            aligned = self._align_to_batch(h, batch_size=batch_size, ref=query)
            hidden.append(aligned.unsqueeze(1))
        if not hidden:
            return None
        tokens = torch.cat(hidden, dim=1)
        return tokens

    @staticmethod
    def _mean_weight(w) -> float:
        if w is None:
            return 0.0
        return float(w.detach().mean().item())

    def integrate(self, query: torch.Tensor, *, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if query.dim() != 3:
            raise ValueError("query must be [B,S,D]")
        if query.size(-1) != self.model_dim:
            raise ValueError(f"query dim must be {self.model_dim}")

        hidden_tokens = self._build_hidden_tokens(query)
        param_tokens = self._build_parameter_tokens(batch_size=query.size(0), ref=query)
        if hidden_tokens is None and param_tokens is None:
            self.last_stats = {
                "captured_hidden_count": 0.0,
                "parameter_token_count": 0.0,
                "self_attn_weight_mean": 0.0,
                "hidden_attn_weight_mean": 0.0,
                "context_attn_weight_mean": 0.0,
                "merge_gate": float(torch.sigmoid(self.merge_gate).detach().item()),
            }
            return query

        if hidden_tokens is None:
            memory_tokens = param_tokens
        elif param_tokens is None:
            memory_tokens = hidden_tokens
        else:
            memory_tokens = torch.cat([hidden_tokens, param_tokens], dim=1)

        self_view, self_w = self.self_attn(query, query, query, need_weights=True)
        hidden_view, hidden_w = self.hidden_attn(query, memory_tokens, memory_tokens, need_weights=True)
        mix = torch.softmax(self.query_hidden_mix, dim=0)
        out = query + mix[0] * self_view + mix[1] * hidden_view

        context_w = None
        if context is not None and self.context_attn is not None:
            ctx = context.to(device=query.device, dtype=query.dtype)
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(1).expand(-1, query.size(1), -1)
            elif ctx.dim() == 3 and ctx.size(0) == 1 and query.size(0) > 1:
                ctx = ctx.expand(query.size(0), -1, -1)
            elif ctx.dim() == 3 and ctx.size(0) != query.size(0):
                ctx = ctx.mean(dim=0, keepdim=True).expand(query.size(0), -1, -1)
            if ctx.size(-1) == self.model_dim:
                ctx_view, context_w = self.context_attn(out, ctx, ctx, need_weights=True)
                out = out + 0.25 * ctx_view

        out = self.norm(out)
        refined = self.encoder(out)
        gate = torch.sigmoid(self.merge_gate)
        merged = self.output_norm((1.0 - gate) * query + gate * (out + refined))
        self.last_stats = {
            "captured_hidden_count": float(0 if hidden_tokens is None else hidden_tokens.size(1)),
            "parameter_token_count": float(0 if param_tokens is None else param_tokens.size(1)),
            "memory_token_count": float(memory_tokens.size(1)),
            "self_attn_weight_mean": self._mean_weight(self_w),
            "hidden_attn_weight_mean": self._mean_weight(hidden_w),
            "context_attn_weight_mean": self._mean_weight(context_w),
            "merge_gate": float(gate.detach().item()),
        }
        return merged
