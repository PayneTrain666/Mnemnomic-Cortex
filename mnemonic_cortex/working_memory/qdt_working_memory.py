"""
Plain-language summary
----------------------
What this file is for: Main modern working-memory assembly (QDT-WM): depths, fusion, and commit-gated writes.
How it fits in the system: Active scratchpad between sensory input and long-term memory when QDT is enabled.
Status: ACTIVE when QDT-WM enabled
Important notes for non-coders: This is the center of the working_memory package.
"""

from __future__ import annotations

from .wm_commit_cortex_guards import ensure_commit_proposal_like, ensure_commit_decision_like, ensure_rollback_trace, ensure_compatibility_input, ensure_migration_template_safety, ensure_no_fake_real_source_patch_claim, commit_cortex_contract_trace, commit_cortex_trace

from .wm_depth_guards import ensure_depth_state, ensure_token_state, ensure_triplet_axis, normalize_quaternion, ensure_quaternion_pack, depth_contract_trace, assert_depth_compatible_tokens

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
from .wm_inter_manifold_attention import WMInterManifoldAttention, WMInterManifoldAttentionConfig
from .wm_shared_slot_store import SharedSlotStore, SharedSlotStoreConfig
from .wm_quantum_holographic_storage import QuantumHolographicStorage, QuantumHolographicStorageConfig
from .wm_system_commit_gate import SystemCommitGate, SystemWriteProposal
from .qspin_runtime_shadow_activation import QSpinRuntimeFeatureFlagEvaluator, QSpinRuntimeFeatureFlagSnapshot
from .qspin_experimental_live_activation import (
    QSpinExperimentalLiveActivationController,
    QSpinExperimentalLiveConfig,
    QSpinExperimentalLiveMode,
    QSpinExperimentalLiveRequest,
)


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
        self.maae_stack = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.input_dim,
                nhead=config.num_heads,
                dim_feedforward=max(128, config.input_dim * 2),
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=max(1, int(config.maae_transformer_layers)),
        )
        self.maae_stack_norm = nn.LayerNorm(config.input_dim)

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

        self.inter_manifold_attention = None
        if bool(getattr(config, "enable_inter_manifold_attention", True)):
            self.inter_manifold_attention = WMInterManifoldAttention(
                WMInterManifoldAttentionConfig(
                    dim=config.input_dim,
                    num_heads=config.num_heads,
                    residual_mix=float(getattr(config, "inter_manifold_residual_mix", 0.15)),
                ),
                geometry_linker=getattr(self.memory_augmented_attention, "geometry_linker", None),
            )
        self.last_inter_manifold_stats = {}
        self.last_depth_state = None
        self.last_geometry_by_depth = None
        self.last_trace = None
        self.external_attention_context = None
        self.cross_model_attn = nn.MultiheadAttention(config.input_dim, num_heads=config.num_heads, batch_first=True)
        self.cross_model_stack = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.input_dim,
                nhead=config.num_heads,
                dim_feedforward=max(128, config.input_dim * 2),
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=max(1, int(config.cross_model_attention_layers)),
        )
        self.cross_model_norm = nn.LayerNorm(config.input_dim)
        self.last_attention_stack_tokens = None
        self._ltm_adapter = None
        self.last_qspin_shadow_trace = None
        self.last_qspin_live_trace = None

    def attach_ltm_adapter(self, triple_hybrid, shared_slot_store=None):
        """Replace synthetic LTM bank with live triple-hybrid adapter."""
        from .wm_triple_hybrid_ltm_adapter import TripleHybridLTMExternalMemoryBank

        bank = TripleHybridLTMExternalMemoryBank(
            dim=self.config.input_dim,
            triple_hybrid=triple_hybrid,
            shared_slot_store=shared_slot_store or self.shared_slot_store,
        )
        self.dual_fusion.ltm.external_bank = bank
        self._ltm_adapter = bank
        return bank

    def set_external_attention_context(self, context: Optional[torch.Tensor]) -> None:
        if context is None:
            self.external_attention_context = None
            return
        ref = next(self.parameters())
        ctx = torch.as_tensor(context, device=ref.device, dtype=ref.dtype)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0)
        elif ctx.dim() != 3:
            raise ValueError("external attention context must be [T,D], [B,D], or [B,S,D]")
        if ctx.size(-1) != self.config.input_dim:
            raise ValueError(f"external attention context dim must be {self.config.input_dim}")
        self.external_attention_context = ctx.detach()

    def clear_external_attention_context(self) -> None:
        self.external_attention_context = None

    def get_attention_stack_output(self) -> Optional[torch.Tensor]:
        return self.last_attention_stack_tokens

    def _build_qspin_guarded_shadow_trace(self, operation: str, context_map_name: Optional[str]) -> Dict[str, Any]:
        if not bool(getattr(self.config, "qspin_guarded_shadow", False)):
            trace = {
                "enabled": False,
                "mode": "disabled",
                "reason": "qspin_guarded_shadow_disabled",
                "live_routing": False,
                "payload_transfer": False,
                "writes": False,
                "production_activation": False,
            }
            self.last_qspin_shadow_trace = trace
            return trace
        snapshot = QSpinRuntimeFeatureFlagSnapshot()
        block_reasons = [reason.value for reason in QSpinRuntimeFeatureFlagEvaluator().evaluate(snapshot)]
        trace = {
            "enabled": True,
            "mode": "guarded_experimental_shadow",
            "operation": str(operation),
            "context_map_name": context_map_name or "quantum_holographic",
            "source_matrix_complete": bool(getattr(self.config, "qspin_source_matrix_complete", True)),
            "unsafe_feature_flags": [],
            "block_reasons": block_reasons,
            "allowed_shadow_only": bool(not block_reasons and getattr(self.config, "qspin_source_matrix_complete", True)),
            "live_routing": False,
            "payload_transfer": False,
            "writes": False,
            "production_activation": False,
        }
        self.last_qspin_shadow_trace = trace
        return trace

    def _qspin_live_config(self) -> QSpinExperimentalLiveConfig:
        mode = (
            QSpinExperimentalLiveMode.EXPERIMENTAL_LIVE
            if str(getattr(self.config, "qspin_live_mode", "disabled")).strip().lower() == "experimental_live"
            else QSpinExperimentalLiveMode.DISABLED
        )
        return QSpinExperimentalLiveConfig(
            enabled=bool(getattr(self.config, "qspin_live_activation", False)),
            mode=mode,
            allow_live_routing=bool(getattr(self.config, "qspin_live_allow_routing", False)),
            allow_payload_transfer=bool(getattr(self.config, "qspin_live_allow_payload_transfer", False)),
            allow_shared_slot_write=bool(getattr(self.config, "qspin_live_allow_shared_slot_write", False)),
            allow_qh_storage_write=bool(getattr(self.config, "qspin_live_allow_qh_storage_write", False)),
            allow_commit_execution=bool(getattr(self.config, "qspin_live_allow_commit_execution", False)),
            max_payload_tokens=int(getattr(self.config, "qspin_live_max_payload_tokens", 8)),
            payload_scale=float(getattr(self.config, "qspin_live_payload_scale", 0.05)),
            routing_scale=float(getattr(self.config, "qspin_live_routing_scale", 0.10)),
        ).validate()

    def _evaluate_qspin_live(self, operation: str, context_map_name: Optional[str], *, write_permission_present: bool) -> Dict[str, Any]:
        cfg = self._qspin_live_config()
        requested_writes = operation == "write"
        mode = cfg.mode if cfg.enabled else QSpinExperimentalLiveMode.DISABLED
        req = QSpinExperimentalLiveRequest(
            request_id=f"qspin_live_{operation}",
            operation=operation,
            mode=mode,
            source_matrix_complete=bool(getattr(self.config, "qspin_source_matrix_complete", True)),
            rollback_evidence_present=bool(getattr(self.config, "qspin_rollback_evidence_present", True)),
            kill_switch_enabled=bool(getattr(self.config, "qspin_live_kill_switch_enabled", True)),
            write_permission_present=bool(write_permission_present),
            live_routing_requested=True,
            payload_transfer_requested=True,
            shared_slot_write_requested=requested_writes,
            qh_storage_write_requested=requested_writes,
            commit_execution_requested=requested_writes,
            metadata={"context_map_name": context_map_name or "quantum_holographic"},
        )
        result = QSpinExperimentalLiveActivationController(cfg).evaluate(req).to_dict()
        self.last_qspin_live_trace = result
        return result

    def _qspin_live_decision(self, live_trace: Dict[str, Any]) -> Dict[str, Any]:
        decision = live_trace.get("decision", {})
        return decision if isinstance(decision, dict) else {}

    def _apply_qspin_live_depth_routing(self, depth_state: torch.Tensor, live_trace: Dict[str, Any]):
        decision = self._qspin_live_decision(live_trace)
        if not decision.get("live_routing", False):
            return depth_state, None
        scale = float(getattr(self.config, "qspin_live_routing_scale", 0.10))
        routed = (1.0 - scale) * depth_state + scale * depth_state.roll(shifts=1, dims=1)
        return routed, {
            "mode": "experimental_live",
            "effect": "depth_phase_roll_blend",
            "routing_scale": scale,
            "input_shape": list(depth_state.shape),
            "output_shape": list(routed.shape),
            "raw_payload_free": True,
        }

    def _build_qspin_live_payload(self, depth_state: torch.Tensor, live_trace: Dict[str, Any]) -> Optional[torch.Tensor]:
        decision = self._qspin_live_decision(live_trace)
        if not decision.get("payload_transfer", False):
            return None
        max_tokens = int(getattr(self.config, "qspin_live_max_payload_tokens", 8))
        payload = torch.tanh(depth_state.mean(dim=(1, 3)))[:, :max_tokens, :].contiguous()
        return payload.detach()

    def _validate_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(-1) != self.config.input_dim:
            raise ValueError(f"Expected x [B,T,{self.config.input_dim}], got {tuple(x.shape)}")
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x

    def forward(
        self,
        x: torch.Tensor,
        operation: str = "read",
        context: Optional[torch.Tensor] = None,
        context_map_name: Optional[str] = None,
        importance: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ):
        x = self._validate_input(x)
        trace = self.trace_emitter.start(
            operation,
            stage="WM-2C",
            write_permission_required=operation == "write",
            qdt_config=self.config.to_dict(),
        )
        trace.add(
            "qspin_guarded_shadow",
            "guarded_qspin_shadow_metadata",
            qspin=self._build_qspin_guarded_shadow_trace(operation, context_map_name),
        )
        qspin_live_trace = self._evaluate_qspin_live(
            operation,
            context_map_name,
            write_permission_present=True,
        )
        trace.add(
            "qspin_experimental_live",
            "experimental_live_activation_evaluated",
            qspin=qspin_live_trace,
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
                metadata={
                    "source": "QDTWorkingMemory.write_path",
                    "qspin_experimental_live": self._qspin_live_decision(qspin_live_trace),
                },
            )
            stage_trace = self.system_commit_gate.stage(proposal)
            evaluation = self.system_commit_gate.evaluate(proposal.proposal_id)
            decision = self.system_commit_gate.commit(proposal.proposal_id)
            trace.add("system_commit_gate", "write_proposal_staged", stage_trace=stage_trace)
            trace.add("system_commit_gate", "write_proposal_evaluated", evaluation=evaluation.to_dict())
            trace.add("system_commit_gate", "write_decision", decision=decision.to_dict(), gate_summary=self.system_commit_gate.trace_summary())
            trace.add(
                "qspin_experimental_live",
                "live_write_path_gated",
                qspin_decision=self._qspin_live_decision(qspin_live_trace),
                shared_slot_write_gate=True,
                qh_storage_write_gate=True,
                commit_gate=True,
            )

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
        depth_state, routing_trace = self._apply_qspin_live_depth_routing(depth_state, qspin_live_trace)
        if routing_trace is not None:
            trace.add("qspin_experimental_live", "live_depth_phase_routing_applied", routing=routing_trace)

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
        if context is not None:
            trace.add(
                "context_buffer",
                "context_mounted",
                context_map=context_map_name or "default",
            )

        qspin_payload = self._build_qspin_live_payload(depth_state, qspin_live_trace)
        effective_context = context
        maae_input = curved_out
        if qspin_payload is not None:
            scale = float(getattr(self.config, "qspin_live_payload_scale", 0.05))
            if qspin_payload.size(1) < maae_input.size(1):
                pad = qspin_payload[:, -1:, :].expand(-1, maae_input.size(1) - qspin_payload.size(1), -1)
                payload_for_tokens = torch.cat([qspin_payload, pad], dim=1)
            else:
                payload_for_tokens = qspin_payload[:, : maae_input.size(1), :]
            maae_input = maae_input + scale * payload_for_tokens
            effective_context = qspin_payload if effective_context is None else effective_context
            trace.add(
                "qspin_experimental_live",
                "bounded_payload_transferred_to_attention",
                payload_shape=list(qspin_payload.shape),
                payload_scale=scale,
                raw_payload_free=True,
            )

        maae_tokens, maae_trace = self.memory_augmented_attention(
            maae_input,
            context=effective_context,
            require_write_permission=False,
            prior_trace=trace.to_dict(),
            return_trace=True,
        )
        maae_tokens = self.maae_stack_norm(maae_tokens + self.maae_stack(maae_tokens))
        trace.merge_dict("memory_augmented_attention", maae_trace)
        trace.add(
            "memory_augmented_attention_stack",
            "maae_stack_applied",
            layers=int(self.config.maae_transformer_layers),
        )

        dual_tokens, dual_fusion_trace = self.dual_fusion(
            maae_tokens,
            depth_state=depth_state,
            context=effective_context,
            return_trace=True,
        )
        if self.external_attention_context is not None:
            ctx = self.external_attention_context.to(device=dual_tokens.device, dtype=dual_tokens.dtype)
            if ctx.size(0) == 1 and dual_tokens.size(0) > 1:
                ctx = ctx.expand(dual_tokens.size(0), -1, -1)
            elif ctx.size(0) != dual_tokens.size(0):
                ctx = ctx.mean(dim=0, keepdim=True).expand(dual_tokens.size(0), -1, -1)
            dx, _ = self.cross_model_attn(dual_tokens, ctx, ctx, need_weights=False)
            dual_tokens = self.cross_model_norm(dual_tokens + dx)
            dual_tokens = self.cross_model_norm(dual_tokens + self.cross_model_stack(dual_tokens))
            trace.add(
                "dual_fusion",
                "cross_model_attention_stack_applied",
                layers=int(self.config.cross_model_attention_layers),
            )
        self.last_attention_stack_tokens = dual_tokens.detach()
        trace.merge_dict("dual_fusion", dual_fusion_trace)
        self.last_depth_state = depth_state
        self.last_geometry_by_depth = list(addressing_trace.get("geometry_by_depth") or [])
        if self.inter_manifold_attention is not None:
            system_views = self._dual_fusion_manifold_views()
            dual_tokens, ima_trace = self.inter_manifold_attention(
                dual_tokens,
                depth_state=depth_state,
                geometry_by_depth=self.last_geometry_by_depth,
                context_map_name=context_map_name,
                system_views=system_views,
                return_trace=True,
            )
            self.last_inter_manifold_stats = dict(self.inter_manifold_attention.last_stats)
            self.last_attention_stack_tokens = dual_tokens.detach()
            trace.merge_dict("inter_manifold_attention", ima_trace)
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

    def _dual_fusion_manifold_views(self) -> Dict[str, torch.Tensor]:
        views: Dict[str, torch.Tensor] = {}
        dual = getattr(self.dual_fusion, "last_output", None)
        ltm_out = getattr(getattr(self.dual_fusion, "ltm", None), "last_output", None)
        mann_out = getattr(getattr(self.dual_fusion, "mann", None), "last_output", None)
        spcp_out = getattr(getattr(self.dual_fusion, "spcp", None), "last_output", None)
        if ltm_out is not None and torch.is_tensor(getattr(ltm_out, "memory_context", None)):
            views["ltm"] = ltm_out.memory_context
        if mann_out is not None and torch.is_tensor(getattr(mann_out, "memory_context", None)):
            views["mann"] = mann_out.memory_context
            vis = getattr(mann_out, "visibility", None)
            hops = getattr(vis, "scratchpad_tokens", None)
            if torch.is_tensor(hops):
                views["mann:quaternion"] = hops
        if spcp_out is not None and torch.is_tensor(getattr(spcp_out, "memory_context", None)):
            views["spcp"] = spcp_out.memory_context
        if dual is not None and torch.is_tensor(getattr(dual, "fused_context", None)):
            views["bridge"] = dual.fused_context
        return views

    def get_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "ima_enabled": 1.0 if self.inter_manifold_attention is not None else 0.0,
        }
        for key, value in self.last_inter_manifold_stats.items():
            if isinstance(value, (int, float)):
                metrics[f"ima_{key}"] = float(value)
        return metrics

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


# ---------------------------------------------------------------------------
# WM-QD-2A quaternion-depth quality contract
# ---------------------------------------------------------------------------

def wm_qd2a_depth_contract() -> dict:
    """Return serialization-safe quality metadata for this depth/assembly module.

    This is a no-mutation contract used by the quality-deepening tooling. It
    declares the expected [B,Z,T,3,D] depth-state invariants, [B,T,D] token-state
    compatibility, quaternion normalization requirement, trace serialization,
    PAAMA-X metadata, fallback behavior, and boundedness expectations.
    """
    return depth_contract_trace(module=__name__)


# ---------------------------------------------------------------------------
# WM-QD-5A system commit / cortex integration quality contract
# ---------------------------------------------------------------------------

def wm_qd5a_commit_cortex_contract() -> dict:
    """Return serialization-safe quality metadata for this commit/cortex layer.

    This no-mutation contract declares system write proposal validation,
    commit/reject/rollback/quarantine decision schemas, PAAMA-X write-permission
    enforcement, rollback trace safety, compatibility wrapper shape/finite
    checks, cortex migration template safety, no-fake-real-source-patch
    guarantees, and QDTWorkingMemory write/read path compatibility.
    """
    return commit_cortex_contract_trace(module=__name__)
