from __future__ import annotations

from .wm_commit_cortex_guards import ensure_commit_proposal_like, ensure_commit_decision_like, ensure_rollback_trace, ensure_compatibility_input, ensure_migration_template_safety, ensure_no_fake_real_source_patch_claim, commit_cortex_contract_trace, commit_cortex_trace

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .wm_config import QDTWorkingMemoryConfig
from .qdt_working_memory import QDTWorkingMemory


@dataclass
class QDTWMCompatibilityConfig:
    input_dim: int
    hidden_dim: int = 128
    num_depths: int = 8
    num_slots: int = 8
    num_heads: int = 4
    transformer_layers: int = 1
    default_operation: str = "process"
    return_trace_by_default: bool = False

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
        if self.default_operation not in {"read", "process", "write"}:
            raise ValueError("default_operation must be read/process/write")


@dataclass
class QDTWMCompatibilityTrace:
    operation: str
    input_shape: list
    output_shape: list
    routed_to: str
    qdt_trace: Optional[Dict[str, Any]] = None
    legacy_call_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QDTWMCompatibilityWrapper(nn.Module):
    """Compatibility wrapper for replacing old self.working_memory.

    Old code may call:
    - self.working_memory(x)
    - self.working_memory.read(x)
    - self.working_memory.write(x)
    - self.working_memory.process(x)

    This wrapper keeps those shapes and routes into QDTWorkingMemory.
    """

    def __init__(
        self,
        config: QDTWMCompatibilityConfig,
        qdt_working_memory: Optional[QDTWorkingMemory] = None,
    ):
        super().__init__()
        config.validate()
        self.config = config
        self.qdt_working_memory = qdt_working_memory or QDTWorkingMemory(
            QDTWorkingMemoryConfig(
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                num_depths=config.num_depths,
                num_slots=config.num_slots,
                num_heads=config.num_heads,
                transformer_layers=config.transformer_layers,
            )
        )
        self.last_compatibility_trace: Optional[QDTWMCompatibilityTrace] = None

    @property
    def input_dim(self) -> int:
        return self.config.input_dim

    @property
    def hidden_dim(self) -> int:
        return self.config.hidden_dim

    @property
    def memory_importance(self) -> Optional[torch.Tensor]:
        # Legacy cortex paths may touch this for decay.
        return getattr(self.qdt_working_memory, "memory_importance", None)

    def set_temperature(self, temperature: torch.Tensor | float) -> None:
        # Legacy cortex paths expect WM to accept temperature scaling.
        if hasattr(self.qdt_working_memory, "set_temperature"):
            self.qdt_working_memory.set_temperature(temperature)  # type: ignore[misc]
            return
        # Safe no-op fallback to preserve compatibility contract.
        return

    def get_metrics(self) -> Dict[str, Any]:
        if hasattr(self.qdt_working_memory, "get_metrics"):
            try:
                out = self.qdt_working_memory.get_metrics()  # type: ignore[misc]
                if isinstance(out, dict):
                    return out
            except Exception:
                pass
        trace = None if self.last_compatibility_trace is None else self.last_compatibility_trace.to_dict()
        return {
            "qdt_wrapper_enabled": True,
            "input_dim": float(self.config.input_dim),
            "hidden_dim": float(self.config.hidden_dim),
            "num_depths": float(self.config.num_depths),
            "has_memory_importance": 1.0 if self.memory_importance is not None else 0.0,
            "last_trace_present": 1.0 if trace is not None else 0.0,
        }

    def _validate_x(self, x: torch.Tensor) -> None:
        if x.dim() != 3 or x.size(-1) != self.config.input_dim:
            raise ValueError(f"Expected x [B,T,{self.config.input_dim}], got {tuple(x.shape)}")
        if not torch.isfinite(x).all():
            raise ValueError("x contains NaN or Inf")

    def route(
        self,
        x: torch.Tensor,
        *,
        operation: str,
        context: Optional[torch.Tensor] = None,
        context_map_name: Optional[str] = None,
        importance: Optional[torch.Tensor] = None,
        return_trace: Optional[bool] = None,
        **legacy_kwargs: Any,
    ):
        self._validate_x(x)
        if operation not in {"read", "process", "write"}:
            raise ValueError("operation must be read/process/write")
        want_trace = self.config.return_trace_by_default if return_trace is None else bool(return_trace)
        out, qdt_trace = self.qdt_working_memory(
            x,
            operation=operation,
            context=context,
            context_map_name=context_map_name,
            importance=importance,
            return_trace=True,
        )
        compat_trace = QDTWMCompatibilityTrace(
            operation=operation,
            input_shape=list(x.shape),
            output_shape=list(out.shape),
            routed_to="QDTWorkingMemory",
            qdt_trace=qdt_trace,
            legacy_call_metadata={
                "legacy_kwargs": legacy_kwargs,
                "context_map_name": context_map_name,
                "compatibility_layer": "QDTWMCompatibilityWrapper",
            },
        )
        self.last_compatibility_trace = compat_trace
        if want_trace:
            return out, compat_trace.to_dict()
        return out

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        operation = kwargs.pop("operation", self.config.default_operation)
        return self.route(x, operation=operation, **kwargs)

    def read(self, x: torch.Tensor, **kwargs: Any):
        return self.route(x, operation="read", **kwargs)

    def process(self, x: torch.Tensor, **kwargs: Any):
        return self.route(x, operation="process", **kwargs)

    def write(self, x: torch.Tensor, **kwargs: Any):
        return self.route(x, operation="write", **kwargs)

    def stability_report(self, x: torch.Tensor) -> Dict[str, Any]:
        self._validate_x(x)
        qdt_report = self.qdt_working_memory.stability_report(x)
        return {
            "ok": bool(qdt_report.get("ok", False)),
            "wrapper": "QDTWMCompatibilityWrapper",
            "qdt_report": qdt_report,
            "last_compatibility_trace": None if self.last_compatibility_trace is None else self.last_compatibility_trace.to_dict(),
        }


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
