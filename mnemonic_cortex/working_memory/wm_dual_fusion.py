from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .wm_ltm_cross_attention import WMLTMCrossAttention, WMLTMCrossAttentionConfig
from .wm_mann_cross_attention import WMMANNCrossAttention, WMMANNCrossAttentionConfig
from .wm_spcp_cross_attention import WMSPCPCrossAttention, WMSPCPCrossAttentionConfig


@dataclass
class WMDualFusionConfig:
    """Fusion controller for WM + LTM + MANN + SPCP.

    'Dual fusion' here preserves the historical LTM/MANN dual-fusion doctrine
    while also accepting SPCP as a procedural side-branch.
    """

    dim: int
    top_k: int = 4
    wm_weight: float = 0.55
    ltm_weight: float = 0.18
    mann_weight: float = 0.18
    spcp_weight: float = 0.09
    residual_mix: float = 0.35
    eps: float = 1e-8

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        for name, value in {
            "wm_weight": self.wm_weight,
            "ltm_weight": self.ltm_weight,
            "mann_weight": self.mann_weight,
            "spcp_weight": self.spcp_weight,
            "residual_mix": self.residual_mix,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass
class WMDualFusionOutput:
    output: torch.Tensor
    fused_context: torch.Tensor
    ltm_trace: Dict[str, Any]
    mann_trace: Dict[str, Any]
    spcp_trace: Dict[str, Any]
    trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_shape": list(self.output.shape),
            "fused_context_shape": list(self.fused_context.shape),
            "ltm_trace": self.ltm_trace,
            "mann_trace": self.mann_trace,
            "spcp_trace": self.spcp_trace,
            "trace": self.trace,
        }


class WMDualFusionController(nn.Module):
    """Cross-memory fusion controller.

    Runs LTM, MANN, and SPCP cross-attention modules, then fuses the resulting
    contexts with the current WM token state. MANN trace visibility is preserved
    in the output trace.
    """

    def __init__(self, config: WMDualFusionConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.ltm = WMLTMCrossAttention(WMLTMCrossAttentionConfig(dim=config.dim, top_k=config.top_k))
        self.mann = WMMANNCrossAttention(WMMANNCrossAttentionConfig(dim=config.dim, top_k=config.top_k))
        self.spcp = WMSPCPCrossAttention(WMSPCPCrossAttentionConfig(dim=config.dim, top_k=config.top_k))
        self.fusion_proj = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, config.dim))
        self.last_output: Optional[WMDualFusionOutput] = None

    def _weights(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        raw = torch.tensor(
            [self.config.wm_weight, self.config.ltm_weight, self.config.mann_weight, self.config.spcp_weight],
            device=device,
            dtype=dtype,
        )
        return raw / raw.sum().clamp_min(self.config.eps)

    def forward(
        self,
        tokens: torch.Tensor,
        depth_state: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ):
        if tokens.dim() != 3 or tokens.size(-1) != self.config.dim:
            raise ValueError(f"Expected tokens [B,T,{self.config.dim}], got {tuple(tokens.shape)}")
        if not torch.isfinite(tokens).all():
            raise ValueError("tokens contains NaN or Inf")

        ltm_out, ltm_trace = self.ltm(tokens, depth_state=depth_state, context=context, return_trace=True)
        mann_out, mann_trace = self.mann(tokens, depth_state=depth_state, context=context, return_trace=True)
        spcp_out, spcp_trace = self.spcp(tokens, depth_state=depth_state, context=context, return_trace=True)

        wm_context = tokens.mean(dim=1)
        ltm_context = self.ltm.last_output.memory_context
        mann_context = self.mann.last_output.memory_context
        spcp_context = self.spcp.last_output.memory_context

        w = self._weights(tokens.device, tokens.dtype)
        fused_context = (
            w[0] * wm_context
            + w[1] * ltm_context
            + w[2] * mann_context
            + w[3] * spcp_context
        )
        delta = self.fusion_proj(fused_context).unsqueeze(1)
        output = tokens + self.config.residual_mix * delta

        # Confidence/disagreement
        confidence_parts = torch.stack([
            self.ltm.last_output.response.confidence,
            self.mann.last_output.response.confidence,
            self.spcp.last_output.response.confidence,
        ], dim=0)
        confidence = confidence_parts.mean(dim=0)
        disagreement = torch.stack([ltm_context, mann_context, spcp_context], dim=1).var(dim=1).mean(dim=-1)
        finite = bool(torch.isfinite(output).all().item())

        trace = {
            "trace_type": "wm_dual_fusion_controller",
            "fusion_weights": w.detach().cpu().tolist(),
            "pre_fusion_outputs": {
                "ltm": ltm_trace["output_shape"],
                "mann": mann_trace["output_shape"],
                "spcp": spcp_trace["output_shape"],
            },
            "mann_trace_visibility": mann_trace["visibility"],
            "confidence": confidence.detach().cpu().tolist(),
            "disagreement": disagreement.detach().cpu().tolist(),
            "finite": finite,
            "paamax_metadata": {
                "trace_type": "wm_dual_fusion_controller",
                "confidence": float(confidence.mean().detach().cpu()) if finite else 0.0,
                "disagreement": float(disagreement.mean().detach().cpu()),
                "mann_trace_visible": True,
                "shared_slot_doctrine_deferred_to": "WM-4B",
                "qh_storage_deferred_to": "WM-4C",
            },
        }
        out = WMDualFusionOutput(
            output=output,
            fused_context=fused_context,
            ltm_trace=ltm_trace,
            mann_trace=mann_trace,
            spcp_trace=spcp_trace,
            trace=trace,
        )
        self.last_output = out
        if return_trace:
            return output, out.to_dict()
        return output

    def stability_report(self, tokens: torch.Tensor) -> Dict[str, Any]:
        out, trace = self.forward(tokens, return_trace=True)
        finite = bool(torch.isfinite(out).all().item())
        return {
            "ok": bool(finite and tuple(out.shape) == tuple(tokens.shape)),
            "finite": finite,
            "shape_ok": tuple(out.shape) == tuple(tokens.shape),
            "mann_trace_visible": trace["trace"]["paamax_metadata"]["mann_trace_visible"],
        }
