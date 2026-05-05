from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .wm_config import QDTWorkingMemoryConfig
from .wm_trace import WMTraceEmitter
from .wm_triplet_state import WMTripletProjector
from .wm_depth_adapters import WMDepthAdapters, WMDepthAdaptersConfig
from .wm_depth_fusion import WMDepthFusion, WMDepthFusionConfig
from .wm_quaternion_depth import QuaternionDepthReplicator
from .wm_intra_depth_transformer import WMIntraDepthTransformer, WMIntraDepthTransformerConfig
from .wm_cross_depth_transformer import WMCrossDepthTransformer, WMCrossDepthTransformerConfig
from .curved_slot_state import CurvedSlotStateBank, CurvedSlotStateConfig
from .curvature_metric_policy import CurvatureMetricPolicy, CurvatureMetricPolicyConfig
from .depth_specific_addressing import DepthSpecificAddressing, DepthSpecificAddressingConfig
from .curved_resonant_wm_core import CurvedResonanceConfig, CurvedResonantWMCore
from .curved_shadow_write import CurvedShadowWriteBuffer, CurvedShadowWriteConfig
from .wm_memory_augmented_attention import WMMemoryAugmentedAttention, WMMemoryAugmentedAttentionConfig
from .wm_dual_fusion import WMDualFusionController, WMDualFusionConfig
from .wm_shared_slot_store import SharedSlotStore, SharedSlotStoreConfig
from .wm_quantum_holographic_storage import QuantumHolographicStorage, QuantumHolographicStorageConfig
from .wm_system_commit_gate import SystemCommitGate, SystemWriteProposal


class QDTWorkingMemory(nn.Module):
    """Assembled Quaternion Depth Transformer Working Memory.

    WM-2C wires the usable stack:
    - CurvedResonantWMCore preserves and upgrades the curved WM foundation.
    - QuaternionDepthReplicator creates [B,Z,T,3,D].
    - WMIntraDepthTransformer processes temporal streams.
    - WMCrossDepthTransformer exchanges across depth slices.
    - WMDepthAdapters applies per-depth/triplet adapters.
    - DepthSpecificAddressing reads slot bank per depth.
    - WMDepthFusion fuses back to [B,T,D].
    - CurvedShadowWriteBuffer is available for write staging.
    - WMTraceEmitter emits assembly trace.
    """

    def __init__(self, config: QDTWorkingMemoryConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.trace_emitter = WMTraceEmitter()
        self.shared_slot_store = SharedSlotStore(SharedSlotStoreConfig(namespace="qdt_wm", dim=config.input_dim))
        self.qh_storage = QuantumHolographicStorage(QuantumHolographicStorageConfig(dim=config.input_dim, num_depths=config.num_depths), shared_slot_store=self.shared_slot_store)
        self.slot_bank = CurvedSlotStateBank(CurvedSlotStateConfig(num_slots=config.num_slots, dim=config.input_dim))
        self.curvature_policy = CurvatureMetricPolicy(
            CurvatureMetricPolicyConfig(
                num_slots=config.num_slots,
                num_depths=config.num_depths,
                context_dim=config.input_dim,
            )
        )
        self.shadow_write_buffer = (
            CurvedShadowWriteBuffer(
                CurvedShadowWriteConfig(dim=config.input_dim, require_paamax_permission=True, interference_threshold=1.0),
                slot_bank=self.slot_bank,
            )
            if config.use_shadow_writes
            else None
        )

        self.system_commit_gate = SystemCommitGate(
            dim=config.input_dim,
            shared_slot_store=self.shared_slot_store,
            qh_storage=self.qh_storage,
            shadow_buffer=self.shadow_write_buffer,
            require_write_permission=True,
        )

        self.curved_core = CurvedResonantWMCore(
            CurvedResonanceConfig(
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                resonance_slots=config.num_slots,
            ),
            shadow_write_buffer=self.shadow_write_buffer,
        )

        self.triplet_projector = WMTripletProjector(config.input_dim)
        self.quaternion_depth = QuaternionDepthReplicator(dim=config.input_dim, num_depths=config.num_depths)
        self.intra_depth = WMIntraDepthTransformer(
            WMIntraDepthTransformerConfig(
                dim=config.input_dim,
                num_depths=config.num_depths,
                num_heads=config.num_heads,
                num_layers=config.transformer_layers,
            )
        )
        self.cross_depth = WMCrossDepthTransformer(
            WMCrossDepthTransformerConfig(
                dim=config.input_dim,
                num_depths=config.num_depths,
                num_heads=config.num_heads,
                num_layers=config.transformer_layers,
            )
        )
        self.depth_adapters = WMDepthAdapters(WMDepthAdaptersConfig(dim=config.input_dim, num_depths=config.num_depths))
        self.depth_addressing = DepthSpecificAddressing(
            DepthSpecificAddressingConfig(
                dim=config.input_dim,
                num_slots=config.num_slots,
                num_depths=config.num_depths,
            ),
            slot_bank=self.slot_bank,
            curvature_policy=self.curvature_policy,
        )
        self.memory_augmented_attention = WMMemoryAugmentedAttention(
            WMMemoryAugmentedAttentionConfig(dim=config.input_dim, top_k=min(4, config.num_slots)),
            slot_bank=self.slot_bank,
        )

        self.dual_fusion = WMDualFusionController(
            WMDualFusionConfig(dim=config.input_dim, top_k=min(4, config.num_slots))
        )
        self.dual_fusion.ltm.external_bank.shared_slot_store = self.shared_slot_store
        self.dual_fusion.mann.external_bank.shared_slot_store = self.shared_slot_store
        self.dual_fusion.spcp.external_bank.shared_slot_store = self.shared_slot_store

        self.depth_fusion = WMDepthFusion(
            WMDepthFusionConfig(
                dim=config.input_dim,
                num_depths=config.num_depths,
                residual_weight=config.residual_fusion_weight,
            )
        )

        self.last_trace = None

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.dim() != 3 or x.size(-1) != self.config.input_dim:
            raise ValueError(f"Expected x [B,T,{self.config.input_dim}], got {tuple(x.shape)}")
        if not torch.isfinite(x).all():
            raise ValueError("input contains NaN or Inf")

    def forward(
        self,
        x: torch.Tensor,
        operation: str = "read",
        context: Optional[torch.Tensor] = None,
        context_map_name: Optional[str] = None,
        importance: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ):
        self._validate_input(x)
        trace = self.trace_emitter.start(
            operation,
            stage="WM-2C",
            write_permission_required=operation == "write",
            qdt_config=self.config.to_dict(),
        )

        if operation == "write":
            out, curved_trace = self.curved_core(x, operation="write", importance=importance, return_trace=True)
            trace.merge_dict("curved_core", curved_trace, message="write_path")

            # WM-5A systemwide simultaneous read/write gate.
            write_content = out.detach().mean(dim=(0, 1))
            proposal = SystemWriteProposal.create(
                content=write_content,
                memory_type="wm",
                local_slot_id="qdt_system_write",
                geometry_map=context_map_name or "quantum_holographic",
                depth_index=0,
                triplet_index=0,
                bank_name="qdt_working_memory",
                task_mode=context_map_name or "quantum_holographic",
                confidence=1.0,
                write_permission=True,
                metadata={"source": "QDTWorkingMemory.write_path"},
            )
            stage_trace = self.system_commit_gate.stage(proposal)
            evaluation = self.system_commit_gate.evaluate(proposal.proposal_id)
            decision = self.system_commit_gate.commit(proposal.proposal_id)
            trace.add("system_commit_gate", "write_proposal_staged", stage_trace=stage_trace)
            trace.add("system_commit_gate", "write_proposal_evaluated", evaluation=evaluation.to_dict())
            trace.add("system_commit_gate", "write_decision", decision=decision.to_dict(), gate_summary=self.system_commit_gate.trace_summary())

            confidence = 1.0 if decision.decision in {"commit", "quarantine"} else 0.0
            disagreement = 1.0 if decision.decision in {"reject", "quarantine"} else 0.0
            trace = self.trace_emitter.finish(trace, confidence=confidence, disagreement=disagreement)
            self.last_trace = trace
            if return_trace:
                return out, trace.to_dict()
            return out

        if operation not in {"read", "process"}:
            raise ValueError(f"Unsupported operation: {operation}")

        curved_out, curved_trace = self.curved_core(x, operation="read", return_trace=True)
        trace.merge_dict("curved_core", curved_trace)

        triplet_fused, triplet_state = self.triplet_projector(curved_out, return_state=True)
        trace.add("triplet_state", "triplet_projected", state=triplet_state.to_dict())

        depth_state, q_trace = self.quaternion_depth(triplet_fused, return_trace=True)
        trace.merge_dict("quaternion_depth", q_trace)

        depth_state, intra_trace = self.intra_depth(depth_state, return_trace=True)
        trace.merge_dict("intra_depth", intra_trace)

        depth_state, cross_trace = self.cross_depth(depth_state, return_trace=True)
        trace.merge_dict("cross_depth", cross_trace)

        depth_state, adapter_trace = self.depth_adapters(depth_state, return_trace=True)
        trace.merge_dict("depth_adapters", adapter_trace)

        addressing_output, addressing_trace = self.depth_addressing(
            depth_state,
            context=context,
            context_map_name=context_map_name,
            return_trace=True,
        )
        trace.merge_dict("depth_specific_addressing", addressing_trace)

        maae_tokens, maae_trace = self.memory_augmented_attention(
            curved_out,
            context=context,
            require_write_permission=False,
            prior_trace=trace.to_dict(),
            return_trace=True,
        )
        trace.merge_dict("memory_augmented_attention", maae_trace)

        dual_tokens, dual_fusion_trace = self.dual_fusion(
            maae_tokens,
            depth_state=depth_state,
            context=context,
            return_trace=True,
        )
        trace.merge_dict("dual_fusion", dual_fusion_trace)
        trace.add("shared_slot_store", "shared_slot_registry_updated", registry=self.shared_slot_store.registry.trace_summary())

        # WM-4C: create a lightweight QH-compatible storage record for the read anchor.
        anchor_content = dual_tokens.detach().mean(dim=(0, 1))
        qh_write = self.shared_slot_store.write_slot(
            memory_type="wm",
            local_slot_id="qdt_read_anchor",
            content=anchor_content,
            owner="wm",
            geometry_map=context_map_name or "quantum_holographic",
            depth_index=0,
            confidence=1.0,
            write_permission=True,
            metadata={"source": "QDTWorkingMemory.read_anchor"},
        )
        qh_record = self.qh_storage.create_record(
            canonical_slot_id=qh_write.canonical_id,
            vector=anchor_content,
            depth_index=0,
            bank_name="qdt_working_memory",
            geometry_name=context_map_name or "holographic_phase",
            triplet_index=0,
            memory_type="wm",
            task_mode=context_map_name or "quantum_holographic",
            confidence=1.0,
            write_permission=True,
            metadata={"source": "QDTWorkingMemory.read_path"},
        )
        trace.add("quantum_holographic_storage", "qh_record_created", record=qh_record.to_dict(), qh_summary=self.qh_storage.trace_summary())
        trace.add("system_commit_gate", "read_path_gate_summary", gate_summary=self.system_commit_gate.trace_summary())

        fused, fusion_trace = self.depth_fusion(depth_state, residual=dual_tokens, return_trace=True)
        trace.merge_dict("depth_fusion", fusion_trace)

        # Small depth-addressed content residual, averaged over depth.
        slot_residual = addressing_output.read_content.mean(dim=1).unsqueeze(1).expand_as(fused)
        out = 0.85 * fused + 0.15 * slot_residual

        if operation == "process":
            out = 0.50 * x + 0.50 * out

        disagreement = float(fusion_trace.get("disagreement", 0.0))
        confidence = float(1.0 / (1.0 + disagreement))
        trace = self.trace_emitter.finish(trace, confidence=confidence, disagreement=disagreement)
        self.last_trace = trace

        if return_trace:
            return out, trace.to_dict()
        return out

    def stability_report(self, x: torch.Tensor) -> Dict[str, Any]:
        out, trace = self.forward(x, operation="read", return_trace=True)
        finite = bool(torch.isfinite(out).all().item())
        shape_ok = tuple(out.shape) == tuple(x.shape)
        return {
            "ok": bool(finite and shape_ok),
            "finite": finite,
            "shape_ok": shape_ok,
            "output_shape": list(out.shape),
            "trace_summary": {
                "item_count": len(trace.get("items", [])),
                "confidence": trace.get("confidence"),
                "disagreement": trace.get("disagreement"),
            },
        }
