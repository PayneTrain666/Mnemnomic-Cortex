from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from .wm_foundation_guards import (
    WMFoundationValidationError,
    ensure_finite_tensor,
    foundation_trace,
    safe_jsonable,
)
from .wm_depth_guards import ensure_token_state


class WMCommitCortexValidationError(WMFoundationValidationError):
    """Raised when system commit/cortex integration validation fails."""


def _as_mapping(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        return obj
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return vars(obj)
    raise WMCommitCortexValidationError(f"object is not mapping/dataclass/to_dict-compatible: {type(obj).__name__}")


def ensure_commit_proposal_like(name: str, proposal: Any, expected_dim: Optional[int] = None) -> Mapping[str, Any]:
    """Validate a system write proposal-like object.

    Required fields:
    - proposal_id
    - content
    - confidence
    - write_permission

    All validation failures are normalized to WMCommitCortexValidationError
    so commit/cortex callers can catch one exception family.
    """
    try:
        data = _as_mapping(proposal)
        proposal_id = data.get("proposal_id")
        if not isinstance(proposal_id, str) or not proposal_id.startswith("sysprop-"):
            raise WMCommitCortexValidationError(f"{name}.proposal_id must start with sysprop-")
        content = data.get("content")
        if not isinstance(content, torch.Tensor):
            raise WMCommitCortexValidationError(f"{name}.content must be a torch.Tensor")
        ensure_finite_tensor(f"{name}.content", content)
        if content.dim() != 1:
            raise WMCommitCortexValidationError(f"{name}.content must be [D], got {tuple(content.shape)}")
        if expected_dim is not None and content.size(-1) != expected_dim:
            raise WMCommitCortexValidationError(f"{name}.content dim must be {expected_dim}, got {content.size(-1)}")
        confidence = float(data.get("confidence", -1.0))
        if not 0.0 <= confidence <= 1.0:
            raise WMCommitCortexValidationError(f"{name}.confidence must be in [0,1]")
        if "write_permission" not in data:
            raise WMCommitCortexValidationError(f"{name}.write_permission missing")
        return data
    except WMCommitCortexValidationError:
        raise
    except Exception as exc:
        raise WMCommitCortexValidationError(str(exc)) from exc


def ensure_commit_decision_like(name: str, decision: Any) -> Mapping[str, Any]:
    """Validate commit/reject/rollback/quarantine decision-like object."""
    try:
        data = _as_mapping(decision)
        proposal_id = data.get("proposal_id")
        if not isinstance(proposal_id, str):
            raise WMCommitCortexValidationError(f"{name}.proposal_id must be string")
        decision_value = data.get("decision")
        allowed = {"commit", "reject", "rollback", "quarantine"}
        if decision_value not in allowed:
            raise WMCommitCortexValidationError(f"{name}.decision must be one of {sorted(allowed)}")
        if not data.get("reason"):
            raise WMCommitCortexValidationError(f"{name}.reason must be non-empty")
        if "paamax_metadata" not in data:
            raise WMCommitCortexValidationError(f"{name}.paamax_metadata missing")
        return data
    except WMCommitCortexValidationError:
        raise
    except Exception as exc:
        raise WMCommitCortexValidationError(str(exc)) from exc


def ensure_rollback_trace(name: str, trace: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate rollback trace is explicit, safe, and serialization-friendly."""
    if not isinstance(trace, Mapping):
        raise WMCommitCortexValidationError(f"{name} must be mapping")
    decision = trace.get("decision") or trace.get("paamax_metadata", {}).get("decision")
    if decision != "rollback":
        raise WMCommitCortexValidationError(f"{name} must describe a rollback decision")
    if trace.get("automatic_memory_store_mutation", False):
        raise WMCommitCortexValidationError(f"{name} must not claim automatic memory-store mutation")
    safe_jsonable(trace)
    return trace


def ensure_compatibility_input(name: str, tensor: torch.Tensor, expected_dim: Optional[int] = None) -> torch.Tensor:
    """Validate cortex/compatibility wrapper input [B,T,D]."""
    try:
        ensure_token_state(name, tensor, expected_dim=expected_dim)
        return tensor
    except Exception as exc:
        raise WMCommitCortexValidationError(str(exc)) from exc


def ensure_migration_template_safety(name: str, template: str) -> str:
    """Validate migration patch template does not pretend a real source patch occurred."""
    if not isinstance(template, str) or not template.strip():
        raise WMCommitCortexValidationError(f"{name} must be non-empty string")
    required = ["replace_cortex_working_memory", "CortexWorkingMemoryIntegrationConfig", "QDT-WM-MAAE"]
    missing = [item for item in required if item not in template]
    if missing:
        raise WMCommitCortexValidationError(f"{name} missing required migration markers: {missing}")
    forbidden = ["# already applied", "production complete", "real source patched"]
    bad = [item for item in forbidden if item.lower() in template.lower()]
    if bad:
        raise WMCommitCortexValidationError(f"{name} contains unsafe/fake-completion markers: {bad}")
    return template


def ensure_no_fake_real_source_patch_claim(name: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Reject metadata that claims real cortex source was patched when source was absent."""
    if not isinstance(payload, Mapping):
        raise WMCommitCortexValidationError(f"{name} must be mapping")
    real_source_available = bool(payload.get("real_cortex_source_available", False))
    real_source_patched = bool(payload.get("real_cortex_source_patched", False))
    if real_source_patched and not real_source_available:
        raise WMCommitCortexValidationError(f"{name} claims real source patch without source availability")
    return payload


def commit_cortex_trace(
    *,
    module: str,
    message: str,
    payload: Optional[Dict[str, Any]] = None,
    confidence: float = 1.0,
    disagreement: float = 0.0,
    conflict_quarantine: bool = False,
    write_permission_required: bool = True,
) -> Dict[str, Any]:
    """Serialization-safe PAAMA-X-compatible commit/cortex trace."""
    return foundation_trace(
        trace_type="wm_qd5a_commit_cortex_trace",
        module=module,
        message=message,
        payload={
            "commit_proposal_schema_required": True,
            "commit_decision_schema_required": True,
            "rollback_trace_safety_required": True,
            "paamax_write_permission_required": True,
            "compatibility_wrapper_shape_checks_required": True,
            "cortex_migration_template_safety_required": True,
            "no_fake_real_source_patch_claim": True,
            **(payload or {}),
        },
        paamax={
            "trace_governance": True,
            "write_permission_required": write_permission_required,
            "conflict_quarantine": conflict_quarantine,
            "confidence": float(confidence),
            "disagreement": float(disagreement),
            "commit_cortex_contract": True,
        },
    )


def commit_cortex_contract_trace(
    *,
    module: str,
    message: str = "system commit/cortex integration module hardened by WM-QD-5A",
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a no-mutation contract trace for commit/cortex modules."""
    return commit_cortex_trace(
        module=module,
        message=message,
        payload={
            "system_write_proposal_prefix": "sysprop-",
            "allowed_commit_decisions": ["commit", "reject", "rollback", "quarantine"],
            "expected_wrapper_input_shape": "[B,T,D]",
            "migration_template_only_when_real_source_absent": True,
            "paamax_metadata_required": True,
            **(payload or {}),
        },
    )


__all__ = [
    "WMCommitCortexValidationError",
    "ensure_commit_proposal_like",
    "ensure_commit_decision_like",
    "ensure_rollback_trace",
    "ensure_compatibility_input",
    "ensure_migration_template_safety",
    "ensure_no_fake_real_source_patch_claim",
    "commit_cortex_trace",
    "commit_cortex_contract_trace",
]
