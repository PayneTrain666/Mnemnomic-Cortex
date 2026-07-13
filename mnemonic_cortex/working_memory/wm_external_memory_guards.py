"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: wm external memory guards.
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import re

import torch
import torch.nn.functional as F

from .wm_foundation_guards import (
    WMFoundationValidationError,
    ensure_finite_tensor,
    foundation_trace,
    safe_jsonable,
)
from .wm_attention_guards import ensure_candidate_tensor, ensure_attention_scores
from .wm_depth_guards import normalize_quaternion


class WMExternalMemoryValidationError(WMFoundationValidationError):
    """Raised when external-memory/shared-slot/QH validation fails."""


_SHARED_ID_RE = re.compile(r"^css-[A-Za-z0-9_.:-]+|^css-[a-f0-9]{16,}$")
_QH_RECORD_RE = re.compile(r"^qhrec-[A-Za-z0-9_.:-]+|^qhrec-[a-f0-9]{16,}$")


def ensure_external_memory_response(
    name: str,
    response: Mapping[str, Any],
    *,
    expected_dim: Optional[int] = None,
    require_trace: bool = True,
) -> Mapping[str, Any]:
    """Validate external-memory response mapping.

    Flexible enough for synthetic LTM/MANN/SPCP interfaces, but strict about
    finite tensors, optional candidate shapes, confidence/disagreement ranges,
    and trace metadata.
    """
    if not isinstance(response, Mapping):
        raise WMExternalMemoryValidationError(f"{name} must be a mapping")
    for key in ["output", "tokens", "candidate_vectors", "scratchpad_tokens", "pre_fusion_output"]:
        value = response.get(key)
        if isinstance(value, torch.Tensor):
            ensure_candidate_tensor(f"{name}.{key}", value, expected_dim=expected_dim, min_rank=1, max_rank=5)
    for key in ["scores", "attention", "per_hop_attention"]:
        value = response.get(key)
        if isinstance(value, torch.Tensor):
            ensure_attention_scores(f"{name}.{key}", value, min_rank=1, max_rank=5)
    for key in ["confidence", "disagreement"]:
        if key in response and response[key] is not None:
            value = float(response[key])
            if not 0.0 <= value <= 1.0:
                raise WMExternalMemoryValidationError(f"{name}.{key} must be in [0,1]")
    if require_trace and not any(key in response for key in ["trace", "metadata", "paamax_metadata"]):
        raise WMExternalMemoryValidationError(f"{name} must include trace/metadata/paamax_metadata")
    return response


def ensure_mann_trace_visibility(name: str, trace: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate required MANN trace visibility fields."""
    if not isinstance(trace, Mapping):
        raise WMExternalMemoryValidationError(f"{name} must be a mapping")
    required = ["pre_fusion_outputs", "per_hop_attention", "scratchpad_tokens", "confidence", "disagreement"]
    missing = [key for key in required if key not in trace]
    if missing:
        raise WMExternalMemoryValidationError(f"{name} missing MANN trace fields: {missing}")
    for key in ["confidence", "disagreement"]:
        value = float(trace[key])
        if not 0.0 <= value <= 1.0:
            raise WMExternalMemoryValidationError(f"{name}.{key} must be in [0,1]")
    return trace


def ensure_fusion_inputs(
    ltm: torch.Tensor,
    mann: torch.Tensor,
    spcp: Optional[torch.Tensor] = None,
    *,
    expected_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Validate LTM/MANN/SPCP fusion tensors."""
    ensure_candidate_tensor("ltm", ltm, expected_dim=expected_dim, min_rank=2, max_rank=4)
    ensure_candidate_tensor("mann", mann, expected_dim=expected_dim, min_rank=2, max_rank=4)
    if ltm.shape != mann.shape:
        raise WMExternalMemoryValidationError(f"ltm and mann shapes must match, got {tuple(ltm.shape)} vs {tuple(mann.shape)}")
    if spcp is not None:
        ensure_candidate_tensor("spcp", spcp, expected_dim=expected_dim, min_rank=2, max_rank=4)
        if spcp.shape != ltm.shape:
            raise WMExternalMemoryValidationError(f"spcp shape must match ltm/mann, got {tuple(spcp.shape)} vs {tuple(ltm.shape)}")
    return ltm, mann, spcp


def ensure_shared_slot_id(name: str, slot_id: str) -> str:
    """Validate canonical shared slot ID shape."""
    if not isinstance(slot_id, str) or not slot_id.startswith("css-") or len(slot_id) < 8:
        raise WMExternalMemoryValidationError(f"{name} must be a canonical shared slot id starting with css-")
    return slot_id


def ensure_shared_slot_record(name: str, record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate shared-slot ownership/conflict/write metadata."""
    if not isinstance(record, Mapping):
        raise WMExternalMemoryValidationError(f"{name} must be a mapping")
    slot_id = record.get("canonical_id") or record.get("canonical_slot_id") or record.get("slot_id")
    ensure_shared_slot_id(f"{name}.canonical_id", slot_id)
    if not any(k in record for k in ["owner", "owner_memory_type", "source_memory_type", "memory_type"]):
        raise WMExternalMemoryValidationError(f"{name} missing owner/source memory metadata")
    if not any(k in record for k in ["conflict_state", "conflict", "quarantine", "paamax_metadata"]):
        raise WMExternalMemoryValidationError(f"{name} missing conflict/quarantine metadata")
    return record


def ensure_qh_code_schema(name: str, schema: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate QH-compatible code schema fields."""
    if not isinstance(schema, Mapping):
        raise WMExternalMemoryValidationError(f"{name} must be a mapping")
    required = ["depth_code", "bank_code", "geometry_code", "triplet_code", "memory_type_code", "task_mode_code"]
    missing = [key for key in required if key not in schema]
    if missing:
        raise WMExternalMemoryValidationError(f"{name} missing QH code fields: {missing}")
    if not str(schema["depth_code"]).startswith("depth-"):
        raise WMExternalMemoryValidationError(f"{name}.depth_code must start with depth-")
    if not str(schema["triplet_code"]).startswith("triplet-"):
        raise WMExternalMemoryValidationError(f"{name}.triplet_code must start with triplet-")
    return schema


def ensure_qh_storage_record(name: str, record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate QH record linkage to shared slot and PAAMA-X metadata."""
    if not isinstance(record, Mapping):
        raise WMExternalMemoryValidationError(f"{name} must be a mapping")
    record_id = record.get("record_id")
    if not isinstance(record_id, str) or not record_id.startswith("qhrec-"):
        raise WMExternalMemoryValidationError(f"{name}.record_id must start with qhrec-")
    ensure_shared_slot_id(f"{name}.canonical_slot_id", record.get("canonical_slot_id"))
    schema = record.get("code_schema")
    if hasattr(schema, "to_dict"):
        schema = schema.to_dict()
    ensure_qh_code_schema(f"{name}.code_schema", schema)
    if "paamax_metadata" not in record and "metadata" not in record:
        raise WMExternalMemoryValidationError(f"{name} missing PAAMA-X/metadata")
    return record


def interference_score(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    """Absolute cosine-style interference score in [0,1]."""
    ensure_finite_tensor("a", a)
    ensure_finite_tensor("b", b)
    if a.shape != b.shape:
        raise WMExternalMemoryValidationError(f"interference tensors must have same shape, got {tuple(a.shape)} vs {tuple(b.shape)}")
    av = F.normalize(a.detach().float().reshape(-1), dim=0, eps=eps)
    bv = F.normalize(b.detach().float().reshape(-1), dim=0, eps=eps)
    value = float(torch.abs(torch.dot(av, bv)).cpu())
    if not 0.0 <= value <= 1.0 + 1e-5:
        raise WMExternalMemoryValidationError("interference score outside [0,1]")
    return min(max(value, 0.0), 1.0)


def external_memory_trace(
    *,
    module: str,
    message: str,
    memory_type: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    confidence: float = 1.0,
    disagreement: float = 0.0,
    conflict_quarantine: bool = False,
    write_permission_required: bool = False,
) -> Dict[str, Any]:
    """Serialization-safe PAAMA-X-compatible external-memory trace."""
    return foundation_trace(
        trace_type="wm_qd4a_external_memory_trace",
        module=module,
        message=message,
        payload={
            "memory_type": memory_type,
            "external_response_schema_required": True,
            "mann_trace_visibility_required": True,
            "fusion_shape_checks_required": True,
            "shared_slot_metadata_required": True,
            "qh_code_schema_required": True,
            "interference_checks_required": True,
            "trace_serialization_required": True,
            "fallback_behavior_required": True,
            **(payload or {}),
        },
        paamax={
            "trace_governance": True,
            "write_permission_required": write_permission_required,
            "conflict_quarantine": conflict_quarantine,
            "confidence": float(confidence),
            "disagreement": float(disagreement),
            "external_memory_contract": True,
        },
    )


def external_memory_contract_trace(
    *,
    module: str,
    message: str = "external-memory/shared-slot/QH module hardened by WM-QD-4A",
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a no-mutation contract trace for external-memory/QH modules."""
    return external_memory_trace(
        module=module,
        message=message,
        payload={
            "supported_memory_types": ["ltm", "mann", "spcp", "shared", "wm"],
            "canonical_shared_slot_id_prefix": "css-",
            "qh_record_id_prefix": "qhrec-",
            "paamax_metadata_required": True,
            "no_fake_quantum_claim": True,
            **(payload or {}),
        },
    )


__all__ = [
    "WMExternalMemoryValidationError",
    "ensure_external_memory_response",
    "ensure_mann_trace_visibility",
    "ensure_fusion_inputs",
    "ensure_shared_slot_id",
    "ensure_shared_slot_record",
    "ensure_qh_code_schema",
    "ensure_qh_storage_record",
    "interference_score",
    "external_memory_trace",
    "external_memory_contract_trace",
]
