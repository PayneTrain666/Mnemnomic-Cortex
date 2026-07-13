"""
Plain-language summary
----------------------
What this file is for: Transformer wrappers around the individual memory banks.
How it fits in the system: Lets each bank refine its own representation before fusion.
Status: ACTIVE
Important notes for non-coders: Prefer this v2 stack over older transformer helpers.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory_hg import EnhancedHyperGeometricMemory
from .memory_cgmn import EnhancedCGMNMemory
from .memory_curved import EnhancedCurvedMemory
from .memory_spatial_ltm import EnhancedSpatialLTMMemory
from .memory_attention import MultiScaleAttention


def _make_attention(
    input_dim: int,
    n_heads: int,
    attention_type: str,
) -> nn.Module:
    attn_type = str(attention_type).strip().lower()
    if attn_type == "multiscale":
        return MultiScaleAttention(input_dim, n_heads)
    return nn.MultiheadAttention(input_dim, num_heads=n_heads, batch_first=True)


def _build_transformer_stack(
    input_dim: int,
    n_layers: int,
    n_heads: int,
    attention_type: str,
) -> nn.Module:
    if int(n_layers) <= 0:
        return nn.Identity()
    if str(attention_type).strip().lower() == "multiscale":
        return nn.Sequential(
            *[
                nn.Sequential(
                    _make_attention(input_dim, n_heads, attention_type),
                    nn.LayerNorm(input_dim),
                )
                for _ in range(int(n_layers))
            ]
        )
    return nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=max(128, input_dim * 2),
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        ),
        num_layers=int(n_layers),
    )


class _MemoryTransformerV2Base(nn.Module):
    """V4.1 transformer wrapper over a core memory bank."""

    def __init__(
        self,
        core: nn.Module,
        input_dim: int,
        n_transformer_layers: int,
        n_heads: int,
        attention_type: str,
    ):
        super().__init__()
        self.memory_core = core
        self.input_dim = int(input_dim)
        self.n_transformer_layers = int(max(0, n_transformer_layers))
        self.n_heads = int(n_heads)
        self.attention_type = str(attention_type).strip().lower()
        self.output_norm = nn.LayerNorm(self.input_dim)
        self.output_transformer = _build_transformer_stack(
            self.input_dim,
            self.n_transformer_layers,
            self.n_heads,
            self.attention_type,
        )

    def _apply_v2_stack(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_transformer_layers <= 0:
            return x
        if isinstance(self.output_transformer, nn.Identity):
            return x
        if isinstance(self.output_transformer, nn.TransformerEncoder):
            return self.output_norm(x + self.output_transformer(x))
        out = x
        for block in self.output_transformer:
            attn, norm = block[0], block[1]
            attn_out, _ = attn(out, out, out, need_weights=False)
            out = norm(out + attn_out)
        return out

    @property
    def core(self) -> nn.Module:
        return self.memory_core

    def set_temperature(self, t: torch.Tensor) -> None:
        self.memory_core.set_temperature(t)

    def enable_energy_efficient_mode(self, enable: bool = True) -> None:
        self.memory_core.enable_energy_efficient_mode(enable)

    def consolidate_unused(self, threshold: float = 0.1) -> None:
        self.memory_core.consolidate_unused(threshold)

    def step_topology(self, loss_value: float) -> None:
        if hasattr(self.memory_core, "step_topology"):
            self.memory_core.step_topology(float(loss_value))

    def get_metrics(self):
        if hasattr(self.memory_core, "get_metrics"):
            return self.memory_core.get_metrics()
        return {}

    def set_external_attention_context(self, context: torch.Tensor) -> None:
        self.memory_core.set_external_attention_context(context)

    def clear_external_attention_context(self) -> None:
        self.memory_core.clear_external_attention_context()

    def ingest_external_vectors(self, vectors: torch.Tensor, *, write_scale: float = 1.0) -> None:
        self.memory_core.ingest_external_vectors(vectors, write_scale=write_scale)

    def __getattr__(self, name: str):
        for container in ("_parameters", "_buffers", "_modules"):
            store = self.__dict__.get(container)
            if store is not None and name in store:
                return store[name]
        core = self.__dict__.get("_modules", {}).get("memory_core")
        if core is not None and hasattr(core, name):
            return getattr(core, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")


def _sinusoidal_pos_enc(seq_len: int, dim: int, device, dtype) -> torch.Tensor:
    pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    i = torch.arange(0, dim, 2, device=device, dtype=dtype)
    div = torch.exp(i * (-math.log(10000.0) / max(1, dim)))
    pe = torch.zeros(seq_len, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:, : pe[:, 1::2].size(1)])
    return pe


class EnhancedHyperGeometricMemoryWithTransformerV2(_MemoryTransformerV2Base):
    def __init__(
        self,
        input_dim: int,
        hg_dim: int,
        hg_slots: int,
        hg_qubits: int,
        n_transformer_layers: int,
        n_heads: int,
        attention_type: str,
        *,
        enable_bank: bool = True,
        bank_size: int = 1024,
        use_entanglement: bool = True,
        ann_centroids: int = 256,
        ann_top: int = 8,
    ):
        core = EnhancedHyperGeometricMemory(
            input_dim,
            manifold_dim=int(hg_dim),
            mem_slots=int(hg_slots),
            quantum_qubits=int(hg_qubits),
            ann_centroids=int(ann_centroids),
            ann_top_centroids=int(ann_top),
            transformer_layers=int(max(1, n_transformer_layers)),
            transformer_heads=int(n_heads),
        )
        super().__init__(core, input_dim, n_transformer_layers, n_heads, attention_type)
        self.enable_bank = bool(enable_bank)
        self.bank_size = int(max(0, bank_size))
        self.use_entanglement = bool(use_entanglement)
        if self.enable_bank and self.bank_size > 0:
            self.aux_bank = nn.Parameter(torch.randn(self.bank_size, self.input_dim) * 0.02)
            self.bank_attn = _make_attention(self.input_dim, self.n_heads, self.attention_type)
            self.bank_norm = nn.LayerNorm(self.input_dim)
        else:
            self.aux_bank = None
            self.bank_attn = None
            self.bank_norm = None
        if self.use_entanglement:
            self.entangle_gate = nn.Parameter(torch.tensor(0.20))
            self.entangle_proj = nn.Linear(self.input_dim * 2, self.input_dim)
            self.entangle_norm = nn.LayerNorm(self.input_dim)
        else:
            self.entangle_gate = None
            self.entangle_proj = None
            self.entangle_norm = None

    def _apply_bank_context(self, x: torch.Tensor) -> torch.Tensor:
        if self.aux_bank is None or self.bank_attn is None:
            return x
        bank = self.aux_bank.unsqueeze(0).expand(x.size(0), -1, -1)
        bank_out, _ = self.bank_attn(x, bank, bank, need_weights=False)
        return self.bank_norm(x + bank_out)

    def _apply_entanglement(self, x: torch.Tensor) -> torch.Tensor:
        if self.entangle_proj is None or self.entangle_gate is None:
            return x
        pooled = x.mean(dim=1, keepdim=True).expand_as(x)
        gate = torch.sigmoid(self.entangle_gate)
        mixed = self.entangle_proj(torch.cat([x, pooled], dim=-1))
        return self.entangle_norm(x + gate * mixed)

    def _postprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = self._apply_v2_stack(x)
        x = self._apply_bank_context(x)
        x = self._apply_entanglement(x)
        return x

    def forward(self, x, operation: str = "read", fire_mask=None, recall_boost: float = 0.3):
        out = self.memory_core(x, operation=operation, fire_mask=fire_mask, recall_boost=recall_boost)
        if operation == "read":
            return self._postprocess(out)
        return out

    def read_batch(self, x, *, fire_mask=None, recall_boost: float = 0.3):
        return self._postprocess(self.memory_core.read_batch(x, fire_mask=fire_mask, recall_boost=recall_boost))

    def pos_enc(self, x: torch.Tensor) -> torch.Tensor:
        return _sinusoidal_pos_enc(x.size(1), self.input_dim, x.device, x.dtype)

    def detect_lightbulb_moment(self, hg_query: torch.Tensor) -> Tuple[bool, torch.Tensor]:
        nov = hg_query.float().std(dim=-1).mean(dim=-1)
        trig = bool((torch.sigmoid(nov * 5).mean() > 0.6).item())
        return trig, nov


class EnhancedCGMNMemoryWithTransformerV2(_MemoryTransformerV2Base):
    def __init__(
        self,
        input_dim: int,
        cgmn_dim: int,
        cgmn_slots: int,
        cgmn_slot_dim: int,
        n_transformer_layers: int,
        n_heads: int,
        attention_type: str,
    ):
        core = EnhancedCGMNMemory(
            input_dim,
            manifold_dim=int(cgmn_dim),
            mem_slots=int(cgmn_slots),
            slot_dim=int(cgmn_slot_dim),
            transformer_layers=int(max(1, n_transformer_layers)),
            transformer_heads=int(n_heads),
        )
        super().__init__(core, input_dim, n_transformer_layers, n_heads, attention_type)

    def forward(self, x, operation: str = "read", fire_mask=None, recall_boost: float = 0.3):
        out = self.memory_core(x, operation=operation, fire_mask=fire_mask, recall_boost=recall_boost)
        if operation == "read":
            return self._apply_v2_stack(out)
        return out

    def read_batch(self, x, *, fire_mask=None, recall_boost: float = 0.3):
        return self._apply_v2_stack(self.memory_core.read_batch(x, fire_mask=fire_mask, recall_boost=recall_boost))

    def detect_geometric_insight(self, cgmn_positions: torch.Tensor) -> Tuple[bool, torch.Tensor]:
        sim = cgmn_positions.float().std(dim=-1).mean(dim=-1)
        trig = bool((torch.sigmoid(sim * 5).mean() > 0.6).item())
        return trig, sim


class SequenceTCN(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        pad = max(0, int(kernel_size) // 2)
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=pad)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + y)


class EnhancedCurvedMemoryWithTransformerV2(_MemoryTransformerV2Base):
    def __init__(
        self,
        input_dim: int,
        curved_hidden: int,
        curved_curvature: int,
        curved_slots: int,
        n_transformer_layers: int,
        n_heads: int,
        attention_type: str,
        *,
        use_tcn: bool = True,
    ):
        core = EnhancedCurvedMemory(
            input_dim,
            hidden_dim=int(curved_hidden),
            curvature_dim=int(curved_curvature),
            mem_slots=int(curved_slots),
            transformer_layers=int(max(1, n_transformer_layers)),
            transformer_heads=int(n_heads),
        )
        super().__init__(core, input_dim, n_transformer_layers, n_heads, attention_type)
        self.use_tcn = bool(use_tcn)
        self.tcn = SequenceTCN(self.input_dim) if self.use_tcn else None

    def _postprocess(self, x: torch.Tensor) -> torch.Tensor:
        if self.tcn is not None:
            x = self.tcn(x)
        return self._apply_v2_stack(x)

    def forward(self, x, operation: str = "read", importance=None):
        out = self.memory_core(x, operation=operation, importance=importance)
        if operation == "read":
            return self._postprocess(out)
        return out

    def read_batch(self, x):
        return self._postprocess(self.memory_core.read_batch(x))

    def detect_associative_chain(self, curved_activation: torch.Tensor) -> Tuple[bool, torch.Tensor]:
        spread = curved_activation.float().std(dim=-1)
        trig = bool((torch.sigmoid(spread * 5).mean() > 0.6).item())
        return trig, spread


class EnhancedSpatialLTMMemoryWithTransformerV2(_MemoryTransformerV2Base):
    def __init__(
        self,
        input_dim: int,
        spatial_value_dim: int,
        spatial_slots: int,
        spatial_key_dim: int,
        n_transformer_layers: int,
        n_heads: int,
        attention_type: str,
        *,
        ltm_topk: int = 8,
        conformal_b: float = 0.08,
        fusion_transformer_layers: int = 0,
        decoder_transformer_layers: int = 0,
        cross_model_attention_layers: int = 4,
        fixed_transformer_layers: int = 3,
        transformer_layers_cfg: int | None = None,
        inherited_bank_layers: int | None = None,
        inherited_fusion_layers: int | None = None,
        wm_lattice_mirror: Any = None,
        wm_shared_slot_store_mirror: Any = None,
    ):
        bank_cfg = int(n_transformer_layers) if transformer_layers_cfg is None else int(transformer_layers_cfg)
        inherited_bank = int(inherited_bank_layers) if inherited_bank_layers is not None else int(max(1, n_transformer_layers))
        core = EnhancedSpatialLTMMemory(
            input_dim,
            value_dim=int(spatial_value_dim),
            key_dim=int(spatial_key_dim),
            slots=int(spatial_slots),
            ltm_topk=int(ltm_topk),
            conformal_b=float(conformal_b),
            transformer_layers=bank_cfg,
            fixed_transformer_layers=int(fixed_transformer_layers),
            transformer_heads=int(n_heads),
            fusion_transformer_layers=int(fusion_transformer_layers),
            decoder_transformer_layers=int(decoder_transformer_layers),
            inherited_bank_layers=inherited_bank,
            inherited_fusion_layers=inherited_fusion_layers,
            cross_model_attention_layers=int(cross_model_attention_layers),
            attention_type=str(attention_type),
            wm_lattice_mirror=wm_lattice_mirror,
            wm_shared_slot_store_mirror=wm_shared_slot_store_mirror,
        )
        super().__init__(core, input_dim, n_transformer_layers, n_heads, attention_type)

    def forward(self, x, operation: str = "read", importance=None, fire_mask=None, recall_boost: float = 0.3):
        out = self.memory_core(x, operation=operation, importance=importance, fire_mask=fire_mask, recall_boost=recall_boost)
        if operation == "read":
            return self._apply_v2_stack(out)
        return out

    def detect_spatial_insight(self, activation: torch.Tensor) -> Tuple[bool, torch.Tensor]:
        return self.memory_core.detect_spatial_insight(activation)
