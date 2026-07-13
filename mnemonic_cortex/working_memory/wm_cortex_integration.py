"""
Plain-language summary
----------------------
What this file is for: Hooks that attach QDT working memory into EnhancedMnemonicCortex.
How it fits in the system: Integration glue between cortex and WM.
Status: ACTIVE when QDT-WM enabled
Important notes for non-coders: Not a standalone memory algorithm.
"""

from __future__ import annotations

from .wm_commit_cortex_guards import ensure_commit_proposal_like, ensure_commit_decision_like, ensure_rollback_trace, ensure_compatibility_input, ensure_migration_template_safety, ensure_no_fake_real_source_patch_claim, commit_cortex_contract_trace, commit_cortex_trace

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .wm_config import QDTWorkingMemoryConfig
from .qdt_working_memory import QDTWorkingMemory
from .wm_compatibility_wrapper import QDTWMCompatibilityConfig, QDTWMCompatibilityWrapper


@dataclass
class CortexWorkingMemoryIntegrationConfig:
    input_dim: int
    hidden_dim: int = 128
    num_depths: int = 8
    num_slots: int = 8
    num_heads: int = 0
    transformer_layers: int = 2
    maae_transformer_layers: int = 2
    cross_model_attention_layers: int = 4
    context_tokens: int = 0
    hardware_profile: str = "custom"
    qspin_guarded_shadow: bool = False
    qspin_source_matrix_complete: bool = True
    qspin_rollback_evidence_present: bool = True
    qspin_live_activation: bool = False
    qspin_live_mode: str = "disabled"
    qspin_live_kill_switch_enabled: bool = True
    qspin_live_allow_routing: bool = False
    qspin_live_allow_payload_transfer: bool = False
    qspin_live_allow_shared_slot_write: bool = False
    qspin_live_allow_qh_storage_write: bool = False
    qspin_live_allow_commit_execution: bool = False
    qspin_live_max_payload_tokens: int = 8
    qspin_live_payload_scale: float = 0.05
    qspin_live_routing_scale: float = 0.10
    default_operation: str = "process"
    return_trace_by_default: bool = False
    use_compatibility_wrapper: bool = True
    preserve_old_reference: bool = True
    old_reference_attr: str = "legacy_working_memory"

    def validate(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_depths <= 0:
            raise ValueError("num_depths must be positive")
        if self.num_slots <= 0:
            raise ValueError("num_slots must be positive")
        if self.num_heads <= 0:
            self.num_heads = QDTWorkingMemoryConfig._pick_num_heads(self.input_dim)
        if self.input_dim % self.num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads")
        if not self.old_reference_attr:
            raise ValueError("old_reference_attr must be non-empty")
        if self.context_tokens < 0:
            raise ValueError("context_tokens must be non-negative")
        if self.default_operation not in {"read", "process", "write"}:
            raise ValueError("default_operation must be read/process/write")

    @classmethod
    def from_hardware_profile(
        cls,
        profile_name: str,
        *,
        input_dim: int | None = None,
        use_compatibility_wrapper: bool = True,
        preserve_old_reference: bool = True,
    ) -> "CortexWorkingMemoryIntegrationConfig":
        qdt_cfg = QDTWorkingMemoryConfig.from_hardware_profile(profile_name, input_dim=input_dim)
        return cls(
            input_dim=qdt_cfg.input_dim,
            hidden_dim=qdt_cfg.hidden_dim,
            num_depths=qdt_cfg.num_depths,
            num_slots=qdt_cfg.num_slots,
            num_heads=qdt_cfg.num_heads,
            transformer_layers=qdt_cfg.transformer_layers,
            maae_transformer_layers=qdt_cfg.maae_transformer_layers,
            cross_model_attention_layers=qdt_cfg.cross_model_attention_layers,
            context_tokens=qdt_cfg.context_tokens,
            hardware_profile=qdt_cfg.hardware_profile,
            qspin_guarded_shadow=qdt_cfg.qspin_guarded_shadow,
            qspin_source_matrix_complete=qdt_cfg.qspin_source_matrix_complete,
            qspin_rollback_evidence_present=qdt_cfg.qspin_rollback_evidence_present,
            qspin_live_activation=qdt_cfg.qspin_live_activation,
            qspin_live_mode=qdt_cfg.qspin_live_mode,
            qspin_live_kill_switch_enabled=qdt_cfg.qspin_live_kill_switch_enabled,
            qspin_live_allow_routing=qdt_cfg.qspin_live_allow_routing,
            qspin_live_allow_payload_transfer=qdt_cfg.qspin_live_allow_payload_transfer,
            qspin_live_allow_shared_slot_write=qdt_cfg.qspin_live_allow_shared_slot_write,
            qspin_live_allow_qh_storage_write=qdt_cfg.qspin_live_allow_qh_storage_write,
            qspin_live_allow_commit_execution=qdt_cfg.qspin_live_allow_commit_execution,
            qspin_live_max_payload_tokens=qdt_cfg.qspin_live_max_payload_tokens,
            qspin_live_payload_scale=qdt_cfg.qspin_live_payload_scale,
            qspin_live_routing_scale=qdt_cfg.qspin_live_routing_scale,
            use_compatibility_wrapper=use_compatibility_wrapper,
            preserve_old_reference=preserve_old_reference,
        )

    def qdt_config(self) -> QDTWorkingMemoryConfig:
        return QDTWorkingMemoryConfig(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_depths=self.num_depths,
            num_slots=self.num_slots,
            num_heads=self.num_heads,
            transformer_layers=self.transformer_layers,
            maae_transformer_layers=self.maae_transformer_layers,
            cross_model_attention_layers=self.cross_model_attention_layers,
            context_tokens=self.context_tokens,
            hardware_profile=self.hardware_profile,
            qspin_guarded_shadow=self.qspin_guarded_shadow,
            qspin_source_matrix_complete=self.qspin_source_matrix_complete,
            qspin_rollback_evidence_present=self.qspin_rollback_evidence_present,
            qspin_live_activation=self.qspin_live_activation,
            qspin_live_mode=self.qspin_live_mode,
            qspin_live_kill_switch_enabled=self.qspin_live_kill_switch_enabled,
            qspin_live_allow_routing=self.qspin_live_allow_routing,
            qspin_live_allow_payload_transfer=self.qspin_live_allow_payload_transfer,
            qspin_live_allow_shared_slot_write=self.qspin_live_allow_shared_slot_write,
            qspin_live_allow_qh_storage_write=self.qspin_live_allow_qh_storage_write,
            qspin_live_allow_commit_execution=self.qspin_live_allow_commit_execution,
            qspin_live_max_payload_tokens=self.qspin_live_max_payload_tokens,
            qspin_live_payload_scale=self.qspin_live_payload_scale,
            qspin_live_routing_scale=self.qspin_live_routing_scale,
        )

    def compatibility_config(self) -> QDTWMCompatibilityConfig:
        return QDTWMCompatibilityConfig(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_depths=self.num_depths,
            num_slots=self.num_slots,
            num_heads=self.num_heads,
            transformer_layers=self.transformer_layers,
            maae_transformer_layers=self.maae_transformer_layers,
            cross_model_attention_layers=self.cross_model_attention_layers,
            context_tokens=self.context_tokens,
            hardware_profile=self.hardware_profile,
            qspin_guarded_shadow=self.qspin_guarded_shadow,
            qspin_source_matrix_complete=self.qspin_source_matrix_complete,
            qspin_rollback_evidence_present=self.qspin_rollback_evidence_present,
            qspin_live_activation=self.qspin_live_activation,
            qspin_live_mode=self.qspin_live_mode,
            qspin_live_kill_switch_enabled=self.qspin_live_kill_switch_enabled,
            qspin_live_allow_routing=self.qspin_live_allow_routing,
            qspin_live_allow_payload_transfer=self.qspin_live_allow_payload_transfer,
            qspin_live_allow_shared_slot_write=self.qspin_live_allow_shared_slot_write,
            qspin_live_allow_qh_storage_write=self.qspin_live_allow_qh_storage_write,
            qspin_live_allow_commit_execution=self.qspin_live_allow_commit_execution,
            qspin_live_max_payload_tokens=self.qspin_live_max_payload_tokens,
            qspin_live_payload_scale=self.qspin_live_payload_scale,
            qspin_live_routing_scale=self.qspin_live_routing_scale,
            default_operation=self.default_operation,
            return_trace_by_default=self.return_trace_by_default,
        )


@dataclass
class CortexWorkingMemoryMigrationResult:
    replaced: bool
    target_class: str
    working_memory_class: str
    preserved_old_reference: bool
    old_reference_attr: Optional[str]
    trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EnhancedMnemonicCortexQDTAdapter(nn.Module):
    """Minimal cortex-shell adapter used when full cortex source is unavailable.

    This proves the working-memory replacement contract without pretending to
    patch a real EnhancedMnemonicCortex source file that is not present.
    """

    def __init__(self, config: CortexWorkingMemoryIntegrationConfig):
        super().__init__()
        config.validate()
        self.integration_config = config
        self.working_memory = QDTWMCompatibilityWrapper(config.compatibility_config())
        self.last_trace: Optional[Dict[str, Any]] = None

    def read_working_memory(self, x: torch.Tensor, **kwargs: Any):
        out, trace = self.working_memory.read(x, return_trace=True, **kwargs)
        self.last_trace = trace
        return out

    def process_working_memory(self, x: torch.Tensor, **kwargs: Any):
        out, trace = self.working_memory.process(x, return_trace=True, **kwargs)
        self.last_trace = trace
        return out

    def write_working_memory(self, x: torch.Tensor, **kwargs: Any):
        out, trace = self.working_memory.write(x, return_trace=True, **kwargs)
        self.last_trace = trace
        return out

    def forward(self, x: torch.Tensor, operation: str = "process", **kwargs: Any):
        if operation == "read":
            return self.read_working_memory(x, **kwargs)
        if operation == "write":
            return self.write_working_memory(x, **kwargs)
        if operation == "process":
            return self.process_working_memory(x, **kwargs)
        raise ValueError("operation must be read/process/write")


def build_qdt_working_memory_for_cortex(config: CortexWorkingMemoryIntegrationConfig) -> nn.Module:
    config.validate()
    if config.use_compatibility_wrapper:
        return QDTWMCompatibilityWrapper(config.compatibility_config())
    return QDTWorkingMemory(config.qdt_config())


def _resolve_unified_shared_slot_store(cortex: Any) -> Any:
    """Prefer cortex hg-episodic shared store over QDT-local registry when both exist."""
    subsystem = getattr(cortex, "shared_memory_subsystem", None)
    if subsystem is not None and getattr(subsystem, "store", None) is not None:
        store = subsystem.store
        # TripleHybridLTMExternalMemoryBank expects WM shared-slot API with registry.
        if hasattr(store, "registry") and hasattr(store, "write_slot"):
            return store
    wm = getattr(cortex, "working_memory", None)
    qdt = getattr(wm, "qdt_working_memory", wm)
    return getattr(qdt, "shared_slot_store", None)


def wire_qdt_ltm_adapter(cortex: Any) -> bool:
    """Attach live triple-hybrid LTM to QDT dual-fusion cross-attention."""
    ltm = getattr(cortex, "long_term_memory", None)
    wm = getattr(cortex, "working_memory", None)
    if ltm is None or wm is None:
        return False

    qdt = getattr(wm, "qdt_working_memory", wm)
    shared = _resolve_unified_shared_slot_store(cortex)
    if hasattr(wm, "attach_ltm_adapter"):
        wm.attach_ltm_adapter(ltm, shared_slot_store=shared)
        return True
    if hasattr(qdt, "attach_ltm_adapter"):
        qdt.attach_ltm_adapter(ltm, shared_slot_store=shared)
        return True
    return False


def replace_cortex_working_memory(
    cortex: Any,
    config: CortexWorkingMemoryIntegrationConfig,
) -> CortexWorkingMemoryMigrationResult:
    config.validate()
    had_old = hasattr(cortex, "working_memory")
    old_wm = getattr(cortex, "working_memory", None)
    if had_old and config.preserve_old_reference:
        setattr(cortex, config.old_reference_attr, old_wm)

    replacement = build_qdt_working_memory_for_cortex(config)
    setattr(cortex, "working_memory", replacement)

    return CortexWorkingMemoryMigrationResult(
        replaced=True,
        target_class=type(cortex).__name__,
        working_memory_class=type(replacement).__name__,
        preserved_old_reference=bool(had_old and config.preserve_old_reference),
        old_reference_attr=config.old_reference_attr if had_old and config.preserve_old_reference else None,
        trace={
            "trace_type": "cortex_working_memory_migration",
            "had_old_working_memory": had_old,
            "replacement": type(replacement).__name__,
            "preserve_old_reference": config.preserve_old_reference,
            "dimensional_depth_preserved": {
                "num_depths": config.num_depths,
                "triplets": True,
                "quaternion_depth": True,
                "maae": True,
                "advanced_attention": True,
                "dual_fusion": True,
                "shared_slot_store": True,
                "qh_storage": True,
                "system_commit_gate": True,
            },
            "hardware_profile": config.hardware_profile,
            "capacity_estimate": config.qdt_config().capacity_estimate(batch_size=1, seq_len=1).to_dict(),
            "qspin_guarded_shadow": {
                "enabled": bool(config.qspin_guarded_shadow),
                "source_matrix_complete": bool(config.qspin_source_matrix_complete),
                "live_payload_transfer": bool(config.qspin_live_allow_payload_transfer and config.qspin_live_activation),
                "experimental_live_activation": bool(config.qspin_live_activation),
                "live_mode": str(config.qspin_live_mode),
                "live_kill_switch_enabled": bool(config.qspin_live_kill_switch_enabled),
                "production_activation": False,
            },
        },
    )


def migration_patch_template(config: CortexWorkingMemoryIntegrationConfig) -> str:
    """Return a text patch template for a real EnhancedMnemonicCortex source file."""
    return f"""# QDT-WM-MAAE WM-6A patch template
from mnemonic_cortex.working_memory import (
    CortexWorkingMemoryIntegrationConfig,
    replace_cortex_working_memory,
)

# In EnhancedMnemonicCortex.__init__ after legacy dimensions are known:
replace_cortex_working_memory(
    self,
    CortexWorkingMemoryIntegrationConfig(
        input_dim={config.input_dim},
        hidden_dim={config.hidden_dim},
        num_depths={config.num_depths},
        num_slots={config.num_slots},
        num_heads={config.num_heads},
        transformer_layers={config.transformer_layers},
        maae_transformer_layers={config.maae_transformer_layers},
        cross_model_attention_layers={config.cross_model_attention_layers},
        context_tokens={config.context_tokens},
        hardware_profile="{config.hardware_profile}",
        qspin_guarded_shadow={config.qspin_guarded_shadow},
        qspin_live_activation={config.qspin_live_activation},
        qspin_live_mode="{config.qspin_live_mode}",
        qspin_live_kill_switch_enabled={config.qspin_live_kill_switch_enabled},
        qspin_live_max_payload_tokens={config.qspin_live_max_payload_tokens},
        default_operation="{config.default_operation}",
        return_trace_by_default={config.return_trace_by_default},
        use_compatibility_wrapper={config.use_compatibility_wrapper},
        preserve_old_reference={config.preserve_old_reference},
    ),
)
"""


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
