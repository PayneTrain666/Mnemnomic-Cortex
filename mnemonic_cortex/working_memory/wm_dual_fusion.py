"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: wm dual fusion.
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.
"""

from __future__ import annotations

from .wm_external_memory_guards import ensure_external_memory_response, ensure_mann_trace_visibility, ensure_fusion_inputs, ensure_shared_slot_id, ensure_shared_slot_record, ensure_qh_code_schema, ensure_qh_storage_record, interference_score, external_memory_contract_trace, external_memory_trace

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .wm_ltm_cross_attention import WMLTMCrossAttention, WMLTMCrossAttentionConfig
from .wm_mann_cross_attention import WMMANNCrossAttention, WMMANNCrossAttentionConfig
from .wm_spcp_cross_attention import WMSPCPCrossAttention, WMSPCPCrossAttentionConfig
from .wm_chart_fusion_policy import (
    WMChartFusionPolicy,
    WMChartFusionPolicyConfig,
)
from .wm_prefusion_handoff import (
    FUSION_SPACE,
    charts_from_handoffs,
    tangent_from_output,
    wm_token_handoff,
)


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
    native_chart_mix: float = 0.0
    enable_native_chart_attention: bool = True
    native_chart_attention_mix: float = 1.0
    native_chart_score_temperature: float = 0.35
    enable_chart_fusion_policy: bool = False
    chart_fusion_gate_init: float = 0.0
    chart_fusion_condition_mix: float = 0.15
    enable_prefusion_handoff: bool = False
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
            "native_chart_mix": self.native_chart_mix,
            "native_chart_attention_mix": self.native_chart_attention_mix,
            "chart_fusion_gate_init": self.chart_fusion_gate_init,
            "chart_fusion_condition_mix": self.chart_fusion_condition_mix,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= float(self.native_chart_mix) <= 1.0:
            raise ValueError("native_chart_mix must be in [0,1]")
        if not 0.0 <= float(self.native_chart_attention_mix) <= 1.0:
            raise ValueError("native_chart_attention_mix must be in [0,1]")
        if float(self.native_chart_score_temperature) <= 0:
            raise ValueError("native_chart_score_temperature must be positive")
        if not 0.0 <= float(self.chart_fusion_gate_init) <= 1.0:
            raise ValueError("chart_fusion_gate_init must be in [0,1]")
        if not 0.0 <= float(self.chart_fusion_condition_mix) <= 1.0:
            raise ValueError("chart_fusion_condition_mix must be in [0,1]")


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
        handoff_on = bool(config.enable_prefusion_handoff or config.enable_chart_fusion_policy)
        native_on = bool(config.enable_native_chart_attention or handoff_on)
        native_attn = {
            "enable_native_chart_attention": native_on,
            "native_chart_attention_mix": float(config.native_chart_attention_mix),
            "native_chart_score_temperature": float(config.native_chart_score_temperature),
            "enable_prefusion_handoff": handoff_on,
        }
        self.ltm = WMLTMCrossAttention(
            WMLTMCrossAttentionConfig(dim=config.dim, top_k=config.top_k, **native_attn)
        )
        self.mann = WMMANNCrossAttention(
            WMMANNCrossAttentionConfig(dim=config.dim, top_k=config.top_k, **native_attn)
        )
        self.spcp = WMSPCPCrossAttention(
            WMSPCPCrossAttentionConfig(dim=config.dim, top_k=config.top_k, **native_attn)
        )
        self.fusion_proj = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, config.dim))
        self.fusion_policy: Optional[WMChartFusionPolicy] = None
        if bool(config.enable_chart_fusion_policy):
            self.fusion_policy = WMChartFusionPolicy(
                WMChartFusionPolicyConfig(
                    enable=True,
                    gate_init=float(config.chart_fusion_gate_init),
                    condition_mix=float(config.chart_fusion_condition_mix),
                    legacy_weights=(
                        float(config.wm_weight),
                        float(config.ltm_weight),
                        float(config.mann_weight),
                        float(config.spcp_weight),
                    ),
                    eps=float(config.eps),
                )
            )
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
        context_map_name: Optional[str] = None,
        native_chart_mix: Optional[float] = None,
    ):
        if tokens.dim() != 3 or tokens.size(-1) != self.config.dim:
            raise ValueError(f"Expected tokens [B,T,{self.config.dim}], got {tuple(tokens.shape)}")
        if not torch.isfinite(tokens).all():
            tokens = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)

        chart_mix = self.config.native_chart_mix if native_chart_mix is None else float(native_chart_mix)
        ltm_out, ltm_trace = self.ltm(
            tokens,
            depth_state=depth_state,
            context=context,
            context_map_name=context_map_name,
            native_chart_mix=chart_mix,
            return_trace=True,
        )
        mann_out, mann_trace = self.mann(
            tokens,
            depth_state=depth_state,
            context=context,
            context_map_name=context_map_name,
            native_chart_mix=chart_mix,
            return_trace=True,
        )
        spcp_out, spcp_trace = self.spcp(
            tokens,
            depth_state=depth_state,
            context=context,
            context_map_name=context_map_name,
            native_chart_mix=chart_mix,
            return_trace=True,
        )

        wm_handoff = None
        handoff_on = self.fusion_policy is not None or bool(self.config.enable_prefusion_handoff)
        if handoff_on:
            wm_handoff = wm_token_handoff(tokens, map_name=context_map_name)
            wm_context = wm_handoff.tangent
            ltm_context, ltm_handoff = tangent_from_output(self.ltm.last_output, self.ltm.last_output.memory_context)
            mann_context, mann_handoff = tangent_from_output(self.mann.last_output, self.mann.last_output.memory_context)
            spcp_context, spcp_handoff = tangent_from_output(self.spcp.last_output, self.spcp.last_output.memory_context)
        else:
            wm_context = tokens.mean(dim=1)
            ltm_context = self.ltm.last_output.memory_context
            mann_context = self.mann.last_output.memory_context
            spcp_context = self.spcp.last_output.memory_context
            ltm_handoff = mann_handoff = spcp_handoff = None

        confidence_parts = torch.stack([
            self.ltm.last_output.response.confidence,
            self.mann.last_output.response.confidence,
            self.spcp.last_output.response.confidence,
        ], dim=0)
        confidence = confidence_parts.mean(dim=0)
        disagreement = torch.stack([ltm_context, mann_context, spcp_context], dim=1).var(dim=1).mean(dim=-1)

        policy_trace: Dict[str, Any] = {"enabled": False, "testbed": False}
        if self.fusion_policy is not None:
            charts = charts_from_handoffs([wm_handoff, ltm_handoff, mann_handoff, spcp_handoff])
            if not charts:
                charts = []
                for packed in (ltm_trace, mann_trace, spcp_trace):
                    inner = packed.get("trace") if isinstance(packed, dict) else None
                    inner = inner or packed
                    query_chart = inner.get("query_chart") if isinstance(inner, dict) else None
                    if query_chart:
                        charts.append(query_chart)
                    key_charts = inner.get("key_charts") if isinstance(inner, dict) else None
                    if key_charts:
                        charts.extend(list(key_charts))
            policy_out = self.fusion_policy(
                batch=int(tokens.size(0)),
                device=tokens.device,
                dtype=tokens.dtype,
                map_name=context_map_name,
                charts=charts,
                confidence=confidence,
                disagreement=disagreement,
            )
            w = policy_out.weights
            fused_context = self.fusion_policy.mix_sources(
                w, wm_context, ltm_context, mann_context, spcp_context
            )
            policy_trace = dict(policy_out.trace)
            weight_list = w.mean(dim=0).detach().cpu().tolist() if w.dim() == 2 else w.detach().cpu().tolist()
        else:
            w = self._weights(tokens.device, tokens.dtype)
            fused_context = (
                w[0] * wm_context
                + w[1] * ltm_context
                + w[2] * mann_context
                + w[3] * spcp_context
            )
            weight_list = w.detach().cpu().tolist()
        delta = self.fusion_proj(fused_context).unsqueeze(1)
        output = tokens + self.config.residual_mix * delta

        finite = bool(torch.isfinite(output).all().item())

        trace = {
            "trace_type": "wm_dual_fusion_controller",
            "fusion_weights": weight_list,
            "chart_fusion_policy": policy_trace,
            "prefusion_handoff": {
                "enabled": bool(handoff_on),
                "space": FUSION_SPACE if handoff_on else None,
                "complete": bool(
                    handoff_on
                    and ltm_handoff is not None
                    and mann_handoff is not None
                    and spcp_handoff is not None
                ),
                "systems": {
                    "wm": None if wm_handoff is None else wm_handoff.to_dict(),
                    "ltm": None if ltm_handoff is None else ltm_handoff.to_dict(),
                    "mann": None if mann_handoff is None else mann_handoff.to_dict(),
                    "spcp": None if spcp_handoff is None else spcp_handoff.to_dict(),
                },
            },
            "pre_fusion_outputs": {
                "ltm": ltm_trace["output_shape"],
                "mann": mann_trace["output_shape"],
                "spcp": spcp_trace["output_shape"],
            },
            "mann_trace_visibility": mann_trace["visibility"],
            "confidence": confidence.detach().cpu().tolist(),
            "disagreement": disagreement.detach().cpu().tolist(),
            "finite": finite,
            "context_map_name": context_map_name,
            "native_chart_mix": float(chart_mix),
            "native_chart_attention": bool(self.config.enable_native_chart_attention),
            "native_chart_attention_mix": float(self.config.native_chart_attention_mix),
            "paamax_metadata": {
                "trace_type": "wm_dual_fusion_controller",
                "confidence": float(confidence.mean().detach().cpu()) if finite else 0.0,
                "disagreement": float(disagreement.mean().detach().cpu()),
                "mann_trace_visible": True,
                "native_chart_score_and_mix": bool(self.config.enable_native_chart_attention),
                "chart_fusion_policy": bool(self.fusion_policy is not None),
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


# ---------------------------------------------------------------------------
# WM-QD-4A external-memory/shared-slot/QH quality contract
# ---------------------------------------------------------------------------

def wm_qd4a_external_memory_contract() -> dict:
    """Return serialization-safe quality metadata for this external-memory layer.

    This no-mutation contract declares external memory response schemas, MANN
    trace visibility, fusion shape checks, shared-slot ownership/conflict
    metadata, QH code schema validation, interference checks, trace
    serialization, PAAMA-X write-permission metadata, fallback behavior, and
    compatibility with QDTWorkingMemory.
    """
    return external_memory_contract_trace(module=__name__)
