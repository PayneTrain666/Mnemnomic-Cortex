"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: inter-manifold attention.
How it fits in the system: Watches communications among geometry-map manifolds
across WM, LTM banks, MANN, SPCP, and optional parameter-loop manifolds, then
mixes useful messages back into the token stream.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: This does not write long-term stores, shared
slots, QH records, or activate QSPIN live routing.
"""

from __future__ import annotations

from .wm_attention_guards import (
    attention_contract_trace,
    attention_trace,
    ensure_attention_query,
)
from .wm_foundation_guards import ensure_finite_tensor

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .context_geometry_maps import GEOMETRY_SET, build_default_context_geometry_maps
from .wm_geometry_linker import WMGeometryLinker


MANIFOLD_ALIASES: Dict[str, str] = {
    "euclid": "euclidean",
    "sphere": "spherical",
    "cp": "complex_projective",
    "spatial_s3": "spherical",
    "euclidean_bridge": "euclidean",
    "complex_projective_kahler": "cp_kahler",
    "grassmann_subspace": "grassmann",
    "toroidal": "torus",
    "fisher_rao": "product",
    "quaternion_spatial_loop": "quaternion",
    "poincare": "poincare",
}

SYSTEM_NAMES: Tuple[str, ...] = (
    "wm",
    "hg",
    "cgmn",
    "curved",
    "spatial",
    "mann",
    "spcp",
    "psls",
    "ltm",
    "bridge",
    "unknown",
)

MANIFOLD_NAMES: Tuple[str, ...] = tuple(sorted(GEOMETRY_SET)) + (
    "spatial_s3",
    "euclidean_bridge",
    "complex_projective_kahler",
    "grassmann_subspace",
    "toroidal",
    "fisher_rao",
    "quaternion_spatial_loop",
    "unknown",
)


def canonicalize_manifold(name: Optional[str]) -> str:
    raw = str(name or "unknown").strip().lower()
    if not raw:
        return "unknown"
    aliased = MANIFOLD_ALIASES.get(raw, raw)
    if aliased in GEOMETRY_SET or aliased in MANIFOLD_NAMES:
        return aliased
    return "unknown"


def canonicalize_system(name: Optional[str]) -> str:
    raw = str(name or "unknown").strip().lower()
    if raw in SYSTEM_NAMES:
        return raw
    if ":" in raw:
        return canonicalize_system(raw.split(":", 1)[0])
    return "unknown"


def _scalar_weight(value: Any) -> float:
    if value is None:
        return 0.0
    if torch.is_tensor(value):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().float().mean().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class WMInterManifoldAttentionConfig:
    """Cross-manifold communication monitor and residual mixer.

    Tokens stay [B,T,D]. Manifold communications are pooled into a compact
    token set, attended, then mixed back. residual_mix=0 is identity.
    """

    dim: int
    num_heads: int = 0
    residual_mix: float = 0.15
    max_manifold_tokens: int = 48
    eps: float = 1e-8
    top_edges: int = 6

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.num_heads <= 0:
            for heads in (8, 4, 2):
                if self.dim % heads == 0:
                    self.num_heads = heads
                    break
            else:
                self.num_heads = 1
        if self.dim % int(self.num_heads) != 0:
            raise ValueError("dim must be divisible by num_heads")
        if not 0.0 <= self.residual_mix <= 1.0:
            raise ValueError("residual_mix must be in [0,1]")
        if self.max_manifold_tokens <= 1:
            raise ValueError("max_manifold_tokens must be > 1")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        self.top_edges = int(max(1, self.top_edges))


@dataclass
class WMInterManifoldAttentionOutput:
    output: torch.Tensor
    manifold_tokens: torch.Tensor
    attended_tokens: torch.Tensor
    attention_weights: torch.Tensor
    labels: List[str]
    trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_shape": list(self.output.shape),
            "manifold_token_shape": list(self.manifold_tokens.shape),
            "attended_token_shape": list(self.attended_tokens.shape),
            "attention_weight_shape": list(self.attention_weights.shape),
            "labels": list(self.labels),
            "trace": self.trace,
        }


class WMInterManifoldAttention(nn.Module):
    """Attend over geometry-map / memory-system / MANN manifold communications.

    This module:
    - builds one token per (system, manifold) view
    - cross-attends those views so inter-manifold messages are visible
    - mixes the attended communication back into the sequence
    - never writes LTM, MANN, shared slots, or QH storage
    """

    def __init__(
        self,
        config: WMInterManifoldAttentionConfig,
        geometry_linker: Optional[WMGeometryLinker] = None,
    ):
        super().__init__()
        config.validate()
        self.config = config
        self.geometry_links = list(getattr(geometry_linker, "links", []) or [])
        self.context_maps = build_default_context_geometry_maps(num_depths=8)
        self.manifold_index = {name: i for i, name in enumerate(MANIFOLD_NAMES)}
        self.system_index = {name: i for i, name in enumerate(SYSTEM_NAMES)}
        self.manifold_embed = nn.Embedding(len(MANIFOLD_NAMES), config.dim)
        self.system_embed = nn.Embedding(len(SYSTEM_NAMES), config.dim)
        self.view_norm = nn.LayerNorm(config.dim)
        self.comm_attn = nn.MultiheadAttention(
            config.dim,
            num_heads=config.num_heads,
            batch_first=True,
        )
        self.comm_norm = nn.LayerNorm(config.dim)
        self.mix_proj = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, config.dim))
        self.last_output: Optional[WMInterManifoldAttentionOutput] = None
        self.last_stats: Dict[str, Any] = {}

    def _index(self, table: Mapping[str, int], name: str, unknown: str) -> int:
        return int(table.get(name, table[unknown]))

    def _pool_view(self, value: torch.Tensor) -> torch.Tensor:
        value = ensure_finite_tensor("inter_manifold_view", value)
        if value.size(-1) != self.config.dim:
            raise ValueError(
                f"inter_manifold_view last dim must be {self.config.dim}, got {value.size(-1)}"
            )
        if value.dim() == 2:
            return value.unsqueeze(1)
        if value.dim() == 3:
            # Keep compact hop/slot views; pool long sequences.
            if int(value.size(1)) <= 8:
                return value
            return value.mean(dim=1, keepdim=True)
        if value.dim() == 4:
            return value.mean(dim=2)
        if value.dim() == 5:
            return value.mean(dim=(2, 3))
        pooled = value.reshape(value.size(0), -1, value.size(-1))
        return pooled.mean(dim=1, keepdim=True)

    def _append_token(
        self,
        tokens: List[torch.Tensor],
        labels: List[str],
        pairs: List[Tuple[str, str]],
        tensor: torch.Tensor,
        *,
        system: str,
        manifold: str,
    ) -> None:
        if tensor.numel() == 0:
            return
        view = self._pool_view(tensor)
        if view.dim() == 2:
            view = view.unsqueeze(1)
        system = canonicalize_system(system)
        for idx in range(int(view.size(1))):
            if len(tokens) >= int(self.config.max_manifold_tokens):
                return
            slice_tok = view[:, idx : idx + 1, :]
            if not torch.isfinite(slice_tok).all():
                slice_tok = torch.nan_to_num(slice_tok, nan=0.0, posinf=0.0, neginf=0.0)
            manifold_name = canonicalize_manifold(manifold)
            tokens.append(slice_tok)
            labels.append(f"{system}:{manifold_name}")
            pairs.append((system, manifold_name))

    def build_manifold_tokens(
        self,
        tokens: torch.Tensor,
        *,
        depth_state: Optional[torch.Tensor] = None,
        geometry_by_depth: Optional[Sequence[str]] = None,
        context_map_name: Optional[str] = None,
        system_views: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[str], List[Tuple[str, str]]]:
        ensure_attention_query("inter_manifold_tokens", tokens, expected_dim=self.config.dim)
        views: List[torch.Tensor] = []
        labels: List[str] = []
        pairs: List[Tuple[str, str]] = []

        chart = list(geometry_by_depth or ())
        if not chart and context_map_name:
            spec = self.context_maps.get(str(context_map_name))
            if spec is not None:
                chart = list(spec.geometry_by_depth)

        if depth_state is not None:
            depth_tokens = self._pool_view(depth_state)
            n_depth = int(depth_tokens.size(1))
            if len(chart) < n_depth:
                chart = list(chart) + ["euclidean"] * (n_depth - len(chart))
            for idx in range(n_depth):
                self._append_token(
                    views,
                    labels,
                    pairs,
                    depth_tokens[:, idx, :],
                    system="wm",
                    manifold=chart[idx],
                )
        else:
            self._append_token(views, labels, pairs, tokens, system="wm", manifold=chart[0] if chart else "euclidean")

        if system_views:
            for raw_name, tensor in system_views.items():
                if tensor is None or not torch.is_tensor(tensor):
                    continue
                name = str(raw_name)
                if ":" in name:
                    system, manifold = name.split(":", 1)
                else:
                    system, manifold = name, "euclidean"
                    if canonicalize_system(name) == "mann":
                        manifold = "quaternion"
                    elif canonicalize_system(name) == "spcp":
                        manifold = "spcp"
                    elif canonicalize_system(name) == "psls":
                        manifold = "product"
                    elif canonicalize_system(name) in {"hg"}:
                        manifold = "hyperbolic"
                    elif canonicalize_system(name) in {"cgmn"}:
                        manifold = "complex_projective"
                    elif canonicalize_system(name) in {"curved"}:
                        manifold = "spherical"
                    elif canonicalize_system(name) in {"spatial"}:
                        manifold = "spatial_se3"
                self._append_token(views, labels, pairs, tensor, system=system, manifold=manifold)

        if not views:
            self._append_token(views, labels, pairs, tokens, system="wm", manifold="euclidean")
        stacked = torch.cat(views, dim=1)
        return stacked, labels, pairs

    def _link_bias(self, pairs: Sequence[Tuple[str, str]], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        n = len(pairs)
        bias = torch.zeros(n, n, device=device, dtype=dtype)
        for link in self.geometry_links:
            src = canonicalize_manifold(link.source_geometry)
            dst = canonicalize_manifold(link.target_geometry)
            gain = float(link.weight)
            for i, (_, mi) in enumerate(pairs):
                for j, (_, mj) in enumerate(pairs):
                    if i == j:
                        continue
                    if mi == src and mj == dst:
                        bias[i, j] = bias[i, j] + gain
                    elif mi == dst and mj == src:
                        bias[i, j] = bias[i, j] + 0.5 * gain
        return bias

    def _geometry_key_bias(
        self,
        pairs: Sequence[Tuple[str, str]],
        geometry_weights: Optional[Mapping[str, Mapping[str, Any]]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        n = len(pairs)
        bias = torch.zeros(n, device=device, dtype=dtype)
        if not geometry_weights:
            return bias
        for i, (system, manifold) in enumerate(pairs):
            system_w = geometry_weights.get(system) or geometry_weights.get(canonicalize_system(system))
            if not isinstance(system_w, Mapping):
                continue
            w = system_w.get(manifold)
            if w is None:
                w = system_w.get(MANIFOLD_ALIASES.get(manifold, manifold))
            bias[i] = torch.log(torch.tensor(_scalar_weight(w) + self.config.eps, device=device, dtype=dtype))
        return bias

    def _monitor_stats(
        self,
        weights: torch.Tensor,
        labels: Sequence[str],
        residual_mix: float,
    ) -> Dict[str, Any]:
        # weights: [B, M, M] averaged over heads if needed
        mean_w = weights.mean(dim=0)
        entropy = -(weights.clamp_min(self.config.eps) * weights.clamp_min(self.config.eps).log()).sum(dim=-1).mean()
        offdiag = mean_w.clone()
        offdiag.fill_diagonal_(0.0)
        flat = offdiag.reshape(-1)
        k = min(int(self.config.top_edges), int(flat.numel()))
        top_vals, top_idx = torch.topk(flat, k=k)
        n = int(mean_w.size(0))
        edges = []
        for score, idx in zip(top_vals.detach().cpu().tolist(), top_idx.detach().cpu().tolist()):
            src = int(idx) // n
            dst = int(idx) % n
            edges.append(
                {
                    "source": labels[src] if src < len(labels) else str(src),
                    "target": labels[dst] if dst < len(labels) else str(dst),
                    "score": float(score),
                }
            )
        pairwise = {
            labels[i]: {
                labels[j]: float(mean_w[i, j].detach().cpu())
                for j in range(n)
            }
            for i in range(n)
        }
        return {
            "gate": float(residual_mix),
            "entropy": float(entropy.detach().cpu()),
            "token_count": int(n),
            "labels": list(labels),
            "top_edges": edges,
            "pairwise_mean": pairwise,
            "global_mean": float(mean_w.detach().mean().cpu()),
        }

    def mix_manifold_communications(
        self,
        tokens: torch.Tensor,
        manifold_tokens: torch.Tensor,
        attended: torch.Tensor,
        residual_mix: Optional[float] = None,
    ) -> torch.Tensor:
        mix = self.config.residual_mix if residual_mix is None else float(residual_mix)
        pooled = attended.mean(dim=1)
        delta = self.mix_proj(pooled).unsqueeze(1)
        return tokens + mix * delta

    def forward(
        self,
        tokens: torch.Tensor,
        depth_state: Optional[torch.Tensor] = None,
        geometry_by_depth: Optional[Sequence[str]] = None,
        context_map_name: Optional[str] = None,
        system_views: Optional[Mapping[str, torch.Tensor]] = None,
        geometry_weights: Optional[Mapping[str, Mapping[str, Any]]] = None,
        residual_mix: Optional[float] = None,
        return_trace: bool = False,
    ):
        if tokens.dim() != 3 or tokens.size(-1) != self.config.dim:
            raise ValueError(f"Expected tokens [B,T,{self.config.dim}], got {tuple(tokens.shape)}")
        if not torch.isfinite(tokens).all():
            tokens = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)

        mix = self.config.residual_mix if residual_mix is None else float(residual_mix)
        mix = max(0.0, min(1.0, mix))

        manifold_tokens, labels, pairs = self.build_manifold_tokens(
            tokens,
            depth_state=depth_state,
            geometry_by_depth=geometry_by_depth,
            context_map_name=context_map_name,
            system_views=system_views,
        )
        sys_idx = torch.tensor(
            [self._index(self.system_index, s, "unknown") for s, _ in pairs],
            device=tokens.device,
            dtype=torch.long,
        )
        man_idx = torch.tensor(
            [self._index(self.manifold_index, m, "unknown") for _, m in pairs],
            device=tokens.device,
            dtype=torch.long,
        )
        tagged = manifold_tokens + self.system_embed(sys_idx).unsqueeze(0) + self.manifold_embed(man_idx).unsqueeze(0)
        tagged = self.view_norm(tagged)

        n = int(tagged.size(1))
        if n <= 1 or mix <= 0.0:
            attn_w = torch.ones(tokens.size(0), n, n, device=tokens.device, dtype=tokens.dtype) / float(max(1, n))
            attended = tagged
            output = tokens if mix <= 0.0 else self.mix_manifold_communications(tokens, manifold_tokens, attended, mix)
        else:
            link_bias = self._link_bias(pairs, tokens.device, tokens.dtype)
            key_bias = self._geometry_key_bias(pairs, geometry_weights, tokens.device, tokens.dtype)
            attn_mask = link_bias + key_bias.unsqueeze(0)
            attended, attn_w = self.comm_attn(
                tagged,
                tagged,
                tagged,
                attn_mask=attn_mask,
                need_weights=True,
                average_attn_weights=True,
            )
            attended = self.comm_norm(tagged + attended)
            output = self.mix_manifold_communications(tokens, manifold_tokens, attended, mix)

        if not torch.isfinite(output).all():
            output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
        finite = bool(torch.isfinite(output).all().item())
        monitor = self._monitor_stats(attn_w, labels, mix)
        trace = attention_trace(
            module=__name__,
            message="inter_manifold_attention_applied",
            lane="geometry_map",
            payload={
                "labels": list(labels),
                "token_count": int(n),
                "residual_mix": float(mix),
                "top_edges": monitor["top_edges"],
                "entropy": monitor["entropy"],
                "global_mean": monitor["global_mean"],
                "qspin_live_routing": False,
                "shared_slot_writes": False,
                "ltm_writes": False,
                "mann_writes": False,
                "qh_writes": False,
                "monitor_only_writes": True,
            },
            confidence=1.0 if finite else 0.0,
            write_permission_required=False,
        )
        trace["trace_type"] = "wm_inter_manifold_attention"
        stats = {
            "enabled": 1.0,
            "gate": float(mix),
            "entropy": float(monitor["entropy"]),
            "token_count": float(n),
            "global_mean": float(monitor["global_mean"]),
            "finite": 1.0 if finite else 0.0,
        }
        if monitor["top_edges"]:
            stats["top_edge_score"] = float(monitor["top_edges"][0]["score"])
        self.last_stats = stats
        out = WMInterManifoldAttentionOutput(
            output=output,
            manifold_tokens=manifold_tokens,
            attended_tokens=attended,
            attention_weights=attn_w,
            labels=list(labels),
            trace=trace,
        )
        self.last_output = out
        if return_trace:
            return output, out.to_dict()
        return output

    def stability_report(self, tokens: torch.Tensor) -> Dict[str, Any]:
        out, packed = self.forward(tokens, return_trace=True)
        finite = bool(torch.isfinite(out).all().item())
        return {
            "ok": bool(finite and tuple(out.shape) == tuple(tokens.shape)),
            "finite": finite,
            "shape_ok": tuple(out.shape) == tuple(tokens.shape),
            "token_count": packed["trace"]["payload"]["token_count"],
            "qspin_live_routing": packed["trace"]["payload"]["qspin_live_routing"],
            "trace": packed["trace"],
        }


def wm_qd3a_attention_contract() -> dict:
    """Return serialization-safe quality metadata for this attention module."""
    return attention_contract_trace(
        module=__name__,
        message="inter-manifold attention monitors geometry-map communications without store writes",
        payload={
            "query_shapes": ["[B,T,D]"],
            "candidate_shapes": ["[B,M,D]", "[B,Z,T,3,D]", "[B,D]"],
            "score_shapes": ["[B,M,M]"],
            "inter_manifold_monitor_required": True,
            "candidate_schema_required": True,
            "qspin_live_routing": False,
            "shared_slot_writes": False,
            "stable_softmax_required": True,
        },
    )
