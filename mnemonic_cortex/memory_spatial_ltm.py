"""Spatial atlas LTM bank adapter for triple-hybrid integration."""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .ltm.config import default_config
from .ltm.transformer_policy import DualTransformerPolicy
from .ltm.geometry_keys import DepthRouter
from .ltm.ltm_system import LTMSubsystem
from .ltm.shared_memory import SharedValueStore
from .ltm.transformer_utils import TransformerStack
from .memory_attention import MultiScaleAttention


class EnhancedSpatialLTMMemory(nn.Module):
    """Geometry-aware spatial atlas memory with triple-hybrid compatible surface."""

    def __init__(
        self,
        input_dim: int,
        *,
        value_dim: int = 0,
        key_dim: int = 64,
        slots: int = 256,
        depth_slices: int = 8,
        ltm_topk: int = 8,
        conformal_b: float = 0.08,
        transformer_layers: int = 0,
        fixed_transformer_layers: int = 3,
        transformer_heads: int = 0,
        fusion_transformer_layers: int = 0,
        decoder_transformer_layers: int = 0,
        inherited_bank_layers: int = 3,
        inherited_fusion_layers: int | None = None,
        cross_model_attention_layers: int = 4,
        attention_type: str = "multiscale",
        transformer_dropout: float = 0.1,
        wm_lattice_mirror: Any = None,
        wm_shared_slot_store_mirror: Any = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.value_dim = int(value_dim) if int(value_dim) > 0 else min(256, self.input_dim)
        self.key_dim = int(key_dim)
        self.slots = int(slots)
        self.attention_type = str(attention_type).strip().lower()
        self.transformer_policy = DualTransformerPolicy.resolve(
            bank_layers_cfg=int(transformer_layers),
            fixed_layers_cfg=int(fixed_transformer_layers),
            fusion_layers_cfg=int(fusion_transformer_layers),
            decoder_layers_cfg=int(decoder_transformer_layers),
            inherited_bank_layers=int(inherited_bank_layers),
            inherited_fusion_layers=inherited_fusion_layers,
        )
        self.transformer_layers = int(self.transformer_policy.bank_layers)
        self.fixed_transformer_layers = int(self.transformer_policy.fixed_layers)
        self.fusion_transformer_layers = int(self.transformer_policy.fusion_layers)
        self.decoder_transformer_layers = int(self.transformer_policy.decoder_layers)
        self.cross_model_attention_layers = int(max(0, cross_model_attention_layers))
        self.transformer_dropout = float(transformer_dropout)
        self.transformer_heads = int(transformer_heads) if int(transformer_heads) > 0 else self._pick_num_heads(self.input_dim)

        if wm_lattice_mirror is not None:
            depth_slices = int(getattr(wm_lattice_mirror, "num_depths", depth_slices))
            slots = int(getattr(wm_lattice_mirror, "slot_count", slots))
            self.value_dim = int(getattr(wm_lattice_mirror, "value_dim", self.value_dim))
            ltm_topk = int(getattr(wm_lattice_mirror, "read_top_k_slots", ltm_topk))

        self.cfg = default_config(
            input_dim=self.input_dim,
            model_dim=self.input_dim,
            value_dim=self.value_dim,
            key_dim=self.key_dim,
            shared_slots=self.slots,
            depth_slices=depth_slices,
            ltm_topk=ltm_topk,
            conformal_b=conformal_b,
            bank_transformer_layers=int(transformer_layers),
            fixed_transformer_layers=int(fixed_transformer_layers),
            fusion_transformer_layers=int(fusion_transformer_layers),
            decoder_transformer_layers=int(decoder_transformer_layers),
            wm_tf_depth=int(self.transformer_policy.bank_layers),
        )
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.value_dim),
            nn.LayerNorm(self.value_dim),
            nn.GELU(),
        )
        self.decoder = nn.Linear(self.value_dim, self.input_dim)
        self.shared = SharedValueStore(self.slots, self.value_dim)
        self.depth_router = DepthRouter(
            self.value_dim,
            self.cfg.phase_bins,
            self.cfg.scale_bins,
            self.cfg.spin_bins,
            self.cfg.depth_slices,
        )
        self.spatial = LTMSubsystem(
            "spatial_atlas",
            self.shared,
            self.key_dim,
            "spatial",
            self.cfg,
            self.cfg.spatial_depth_chart,
        )
        self.memory_curvature = nn.Parameter(torch.zeros(self.slots))
        self.temperature = 1.0
        self.energy_efficient_mode = False
        self.external_attention_context: Optional[torch.Tensor] = None
        self.last_router_features = None
        self.last_read_trace = None
        self.wm_lattice_mirror = wm_lattice_mirror
        self.wm_shared_slot_store_mirror = wm_shared_slot_store_mirror

        self._build_transformer_stacks(
            bank_layers=self.transformer_layers,
            fixed_layers=self.fixed_transformer_layers,
            fusion_layers=self.fusion_transformer_layers,
            decoder_layers=self.decoder_transformer_layers,
            n_heads=self.transformer_heads,
            attention_type=self.attention_type,
        )
        from .memory_transformer_v2 import _make_attention

        self.cross_model_attn = _make_attention(self.input_dim, self.transformer_heads, self.attention_type)
        self.cross_model_norm = nn.LayerNorm(self.input_dim)

    def _build_transformer_stacks(
        self,
        *,
        bank_layers: int,
        fixed_layers: int,
        fusion_layers: int,
        decoder_layers: int,
        n_heads: int,
        attention_type: str,
    ) -> None:
        from .memory_transformer_v2 import _build_transformer_stack

        self.transformer_layers = int(max(0, bank_layers))
        self.fixed_transformer_layers = int(max(0, fixed_layers))
        self.fusion_transformer_layers = int(max(0, fusion_layers))
        self.decoder_transformer_layers = int(max(0, decoder_layers))
        self.transformer_heads = int(n_heads)
        self.attention_type = str(attention_type)

        if self.transformer_layers > 0:
            self.sequence_transformer = _build_transformer_stack(
                self.input_dim,
                self.transformer_layers,
                self.transformer_heads,
                self.attention_type,
            )
            self.sequence_norm = nn.LayerNorm(self.input_dim)
        else:
            self.sequence_transformer = None
            self.sequence_norm = None

        if self.fixed_transformer_layers > 0:
            self.aux_transformer = _build_transformer_stack(
                self.input_dim,
                self.fixed_transformer_layers,
                self.transformer_heads,
                self.attention_type,
            )
            self.aux_norm = nn.LayerNorm(self.input_dim)
        else:
            self.aux_transformer = None
            self.aux_norm = None

        if self.fusion_transformer_layers > 0:
            self.fusion_refiner = _build_transformer_stack(
                self.input_dim,
                self.fusion_transformer_layers,
                self.transformer_heads,
                self.attention_type,
            )
            self.fusion_norm = nn.LayerNorm(self.input_dim)
        else:
            self.fusion_refiner = None
            self.fusion_norm = None

        if self.decoder_transformer_layers > 0:
            self.decoder_stack = TransformerStack(
                self.input_dim,
                depth=self.decoder_transformer_layers,
                heads=self.transformer_heads,
                dropout=self.transformer_dropout,
                ffn_mult=2,
                max_len=64,
            )
        else:
            self.decoder_stack = None

    def rebuild_transformer_stacks(
        self,
        *,
        bank_layers: int,
        fixed_layers: int = 3,
        fusion_layers: int,
        decoder_layers: int,
        n_heads: int,
        attention_type: str,
        inherited_bank_layers: int | None = None,
        inherited_fusion_layers: int | None = None,
    ) -> None:
        inherited_bank = (
            int(inherited_bank_layers)
            if inherited_bank_layers is not None
            else int(getattr(self.transformer_policy, "inherited_bank_layers", bank_layers))
        )
        self.transformer_policy = DualTransformerPolicy.resolve(
            bank_layers_cfg=int(bank_layers),
            fixed_layers_cfg=int(fixed_layers),
            fusion_layers_cfg=int(fusion_layers),
            decoder_layers_cfg=int(decoder_layers),
            inherited_bank_layers=inherited_bank,
            inherited_fusion_layers=inherited_fusion_layers,
        )
        self._build_transformer_stacks(
            bank_layers=self.transformer_policy.bank_layers,
            fixed_layers=self.transformer_policy.fixed_layers,
            fusion_layers=self.transformer_policy.fusion_layers,
            decoder_layers=self.transformer_policy.decoder_layers,
            n_heads=n_heads,
            attention_type=attention_type,
        )

    def attach_wm_lattice_mirror(self, mirror: Any, store: Any = None) -> None:
        self.wm_lattice_mirror = mirror
        self.wm_shared_slot_store_mirror = store
        if mirror is not None:
            self.slots = int(getattr(mirror, "slot_count", self.slots))
            self.cfg = default_config(
                input_dim=self.input_dim,
                model_dim=self.input_dim,
                value_dim=int(getattr(mirror, "value_dim", self.value_dim)),
                key_dim=self.key_dim,
                shared_slots=self.slots,
                depth_slices=int(getattr(mirror, "num_depths", 8)),
                ltm_topk=int(getattr(mirror, "read_top_k_slots", self.cfg.ltm_topk)),
                conformal_b=self.cfg.conformal_b,
            )

    @staticmethod
    def _pick_num_heads(dim: int) -> int:
        for h in (8, 4, 2):
            if dim % h == 0:
                return h
        return 1

    def set_temperature(self, t: torch.Tensor) -> None:
        self.temperature = float(t.detach().mean().item()) if torch.is_tensor(t) else float(t)

    def enable_energy_efficient_mode(self, enable: bool = True) -> None:
        self.energy_efficient_mode = bool(enable)

    def consolidate_unused(self, threshold: float = 0.1) -> None:
        with torch.no_grad():
            self.memory_curvature.mul_(1.0 - 0.05 * float(threshold))

    def step_topology(self, loss_value: float) -> None:
        with torch.no_grad():
            fit = 1.0 / (1.0 + float(loss_value))
            self.memory_curvature.add_(0.01 * (fit - 0.5) * torch.randn_like(self.memory_curvature))

    def set_external_attention_context(self, context: torch.Tensor) -> None:
        if context is None:
            self.external_attention_context = None
            return
        ctx = torch.as_tensor(context, device=self.encoder[0].weight.device, dtype=self.encoder[0].weight.dtype)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0)
        self.external_attention_context = ctx.detach()

    def clear_external_attention_context(self) -> None:
        self.external_attention_context = None

    def ingest_external_vectors(self, vectors: torch.Tensor, *, write_scale: float = 1.0) -> None:
        x = torch.as_tensor(vectors, device=self.encoder[0].weight.device, dtype=self.encoder[0].weight.dtype)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.size(-1) != self.input_dim:
            raise ValueError(f"vectors last dim must be {self.input_dim}")
        z = self.encoder(x).mean(dim=1)
        with torch.no_grad():
            self.spatial.write(z, importance=float(max(0.0, min(1.0, write_scale))))
        self._mirror_write_to_wm_store(z)

    def _mirror_write_to_wm_store(self, pooled: torch.Tensor) -> None:
        store = self.wm_shared_slot_store_mirror
        if store is None:
            return
        for b in range(min(pooled.size(0), 4)):
            vec = pooled[b]
            if vec.numel() != store.config.dim:
                continue
            try:
                store.write_slot(
                    memory_type="spatial_ltm",
                    local_slot_id=f"spatial.bank.slot{b % self.slots}",
                    content=vec.detach().reshape(-1),
                    owner="spatial_ltm",
                    geometry_map="spatial",
                    depth_index=0,
                    write_permission=False,
                    metadata={"mirror_source": "spatial_ltm_bank"},
                )
            except Exception:
                continue

    def _encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.encoder(x)

    def _collapse(self, z: torch.Tensor) -> torch.Tensor:
        return z.mean(dim=1) if z.dim() == 3 else z

    def _apply_external_context(self, out: torch.Tensor) -> torch.Tensor:
        if self.external_attention_context is None:
            return out
        ctx = self.external_attention_context.to(device=out.device, dtype=out.dtype)
        if ctx.size(0) == 1 and out.size(0) > 1:
            ctx = ctx.expand(out.size(0), -1, -1)
        elif ctx.size(0) != out.size(0):
            ctx = ctx.mean(dim=0, keepdim=True).expand(out.size(0), -1, -1)
        if isinstance(self.cross_model_attn, MultiScaleAttention):
            ext, _ = self.cross_model_attn(out, ctx, ctx, need_weights=False)
        else:
            ext, _ = self.cross_model_attn(out, ctx, ctx, need_weights=False)
        return self.cross_model_norm(out + ext)

    def _decode_to_sequence(self, pooled: torch.Tensor, seq_len: int) -> torch.Tensor:
        base = self.decoder(pooled)
        out = base.unsqueeze(1).expand(-1, seq_len, -1)
        if self.decoder_stack is not None:
            out, _ = self.decoder_stack(out, need_weights=False)
        return out

    def _apply_transformer_stack(self, out: torch.Tensor, stack: nn.Module, norm: nn.LayerNorm) -> torch.Tensor:
        if stack is None or isinstance(stack, nn.Identity):
            return out
        if isinstance(stack, nn.TransformerEncoder):
            return norm(out + stack(out))
        for block in stack:
            attn, block_norm = block[0], block[1]
            attn_out, _ = attn(out, out, out, need_weights=False)
            out = block_norm(out + attn_out)
        return out

    def _postprocess_sequence(self, out: torch.Tensor) -> torch.Tensor:
        out = self._apply_transformer_stack(out, self.sequence_transformer, self.sequence_norm)
        out = self._apply_transformer_stack(out, self.aux_transformer, self.aux_norm)
        if self.fusion_refiner is not None and not isinstance(self.fusion_refiner, nn.Identity):
            if isinstance(self.fusion_refiner, nn.TransformerEncoder):
                out = self.fusion_norm(out + self.fusion_refiner(out))
            else:
                for block in self.fusion_refiner:
                    attn, norm = block[0], block[1]
                    attn_out, _ = attn(out, out, out, need_weights=False)
                    out = norm(out + attn_out)
        return out

    def forward(
        self,
        x: torch.Tensor,
        operation: str = "read",
        importance=None,
        fire_mask=None,
        recall_boost: float = 0.3,
    ) -> torch.Tensor:
        del fire_mask, recall_boost
        if x.dim() == 2:
            x = x.unsqueeze(1)
        seq_len = x.size(1)
        z = self._encode_sequence(x)
        pooled = self._collapse(z)
        *_bins, depth_slice = self.depth_router(pooled)

        if operation == "write":
            imp = 1.0
            if importance is not None and torch.is_tensor(importance):
                imp = float(importance.detach().mean().clamp(0.0, 1.0).item())
            with torch.no_grad():
                self.spatial.write(pooled, depth_slice=depth_slice, importance=imp)
            self._mirror_write_to_wm_store(pooled)
            return x

        read = self.spatial.read(pooled, depth_slice=depth_slice, b_override=self.cfg.conformal_b)
        self.last_read_trace = read.traces
        self.last_router_features = {
            "omega_mean": read.confidence.detach().reshape(-1),
            "curv_mean": self.memory_curvature.mean().expand(pooled.size(0)),
            "dist_mean": read.traces["bank"].distances.mean(dim=-1).detach(),
            "entropy": -(read.traces["bank"].weights * read.traces["bank"].weights.clamp_min(1e-9).log()).sum(dim=-1).detach(),
        }
        out = self._decode_to_sequence(read.output, seq_len)
        out = self._apply_external_context(out)
        if self.energy_efficient_mode:
            out = out * 0.98
        return self._postprocess_sequence(out)

    def detect_spatial_insight(self, activation: torch.Tensor):
        spread = activation.float().std(dim=-1)
        trig = bool((torch.sigmoid(spread * 4.0).mean() > 0.55).item())
        return trig, spread

    def get_metrics(self):
        metrics = {
            "spatial_slots": float(self.slots),
            "spatial_value_dim": float(self.value_dim),
            "scratch_alive": float(self.shared.snapshot().get("scratch_alive", 0)),
            "bank_transformer_layers": float(self.transformer_layers),
            "fixed_transformer_layers": float(self.fixed_transformer_layers),
            "fusion_transformer_layers": float(self.fusion_transformer_layers),
            "decoder_transformer_layers": float(self.decoder_transformer_layers),
            "dual_stack_active": float(self.fixed_transformer_layers > 0),
        }
        if self.wm_lattice_mirror is not None and hasattr(self.wm_lattice_mirror, "to_dict"):
            metrics["wm_lattice_mirror_slots"] = float(self.wm_lattice_mirror.slot_count)
            metrics["wm_lattice_mirror_depths"] = float(self.wm_lattice_mirror.num_depths)
        return metrics
