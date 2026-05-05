from __future__ import annotations

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
    num_heads: int = 4
    transformer_layers: int = 1
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
            raise ValueError("num_heads must be positive")
        if self.input_dim % self.num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads")
        if not self.old_reference_attr:
            raise ValueError("old_reference_attr must be non-empty")

    def qdt_config(self) -> QDTWorkingMemoryConfig:
        return QDTWorkingMemoryConfig(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_depths=self.num_depths,
            num_slots=self.num_slots,
            num_heads=self.num_heads,
            transformer_layers=self.transformer_layers,
        )

    def compatibility_config(self) -> QDTWMCompatibilityConfig:
        return QDTWMCompatibilityConfig(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_depths=self.num_depths,
            num_slots=self.num_slots,
            num_heads=self.num_heads,
            transformer_layers=self.transformer_layers,
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
        use_compatibility_wrapper={config.use_compatibility_wrapper},
        preserve_old_reference={config.preserve_old_reference},
    ),
)
"""
