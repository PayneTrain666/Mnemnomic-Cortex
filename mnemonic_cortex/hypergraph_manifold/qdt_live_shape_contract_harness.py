"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: qdt live shape contract harness.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
Live-shape contract harness for HGM/QDT WRITE-PREP-5.

This module validates contract-object preview shape/permission semantics without
calling live QDT/WM writer methods.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from .hgm_qdt_write_prep4_result import HGMQDTWritePrep4Result, HGMQDTWritePrep4Options, RealContractObjectConstructionResult
from .hgm_qdt_write_prep5_result import (
    HGMQDTWritePrep5Options,
    LiveShapeContractCheck,
    LiveShapeContractHarnessResult,
    write_prep5_result_id,
)
from .hgm_qdt_write_prep_result import trace_write_prep
from .validation import ValidationResult


def coerce_write_prep5_options(options: Optional[HGMQDTWritePrep5Options | Mapping[str, Any]] = None) -> HGMQDTWritePrep5Options:
    if options is None:
        return HGMQDTWritePrep5Options()
    if isinstance(options, HGMQDTWritePrep5Options):
        return options
    if isinstance(options, Mapping):
        allowed = {k: v for k, v in dict(options).items() if k in HGMQDTWritePrep5Options.__dataclass_fields__}
        return HGMQDTWritePrep5Options(**allowed)
    return HGMQDTWritePrep5Options()


def _contract_result_from_input(obj: Any) -> RealContractObjectConstructionResult | None:
    if isinstance(obj, HGMQDTWritePrep4Result):
        return obj.contract_object_result
    if isinstance(obj, RealContractObjectConstructionResult):
        return obj
    return None


def _finite_vector(values: Any) -> bool:
    try:
        seq = tuple(float(v) for v in values or tuple())
    except Exception:
        return False
    return bool(seq) and all(math.isfinite(v) for v in seq)


def build_live_shape_contract_harness(
    contract_result_or_prep4: Any,
    config=None,
    options: Optional[HGMQDTWritePrep5Options | Mapping[str, Any]] = None,
) -> LiveShapeContractHarnessResult:
    """Validate live contract shape semantics from dry-run previews.

    The check is deliberately strict about future live-shape requirements while
    still remaining non-mutating and preview-only.
    """
    opts = coerce_write_prep5_options(options)
    validation = ValidationResult()
    traces = []
    contract_result = _contract_result_from_input(contract_result_or_prep4)
    if contract_result is None:
        validation.error("prep5.live_shape.invalid_input", "RealContractObjectConstructionResult or HGMQDTWritePrep4Result is required", "contract_result")
        trace = trace_write_prep("qdt_live_shape_contract_harness.invalid", validation, {"live_write_executed": False})
        return LiveShapeContractHarnessResult(
            harness_id=write_prep5_result_id("live_shape_harness_invalid", type(contract_result_or_prep4).__name__),
            checks=tuple(),
            live_shape_ready=False,
            validation=validation,
            trace_records=(trace,),
            metadata={"invalid": True, "live_write_executed": False},
        )
    validation.merge(contract_result.validation)
    traces.extend(contract_result.trace_records)
    checks = []
    previews = tuple(contract_result.previews)[: opts.max_shape_checks]
    if len(contract_result.previews) > opts.max_shape_checks:
        validation.warning("prep5.live_shape.truncated", "contract previews truncated to max_shape_checks", str(opts.max_shape_checks))
    for preview in previews:
        blockers: list[str] = []
        content_shape = tuple(int(v) for v in preview.content_shape or tuple())
        vector = tuple(float(v) for v in preview.object_trace.get("content_vector", tuple()) or tuple())
        # WRITE-PREP-4 redacts object trace to shape only in most paths; fall back
        # to shape semantics when vector values are intentionally absent.
        content_rank_ok = bool(len(content_shape) == 1 and content_shape[0] > 0)
        shape_matches = bool(content_rank_ok and (not vector or len(vector) == content_shape[0]))
        finite_content = True if not vector else _finite_vector(vector)
        confidence_ok = bool(math.isfinite(float(preview.confidence)) and 0.0 <= float(preview.confidence) <= 1.0)
        triplet_ok = bool(int(preview.triplet_index) in (0, 1, 2))
        write_permission_false = bool(preview.write_permission is False and preview.object_trace.get("write_permission", False) is False)
        if opts.require_contract_constructed and not preview.constructed:
            blockers.append("contract object was not constructed")
        if not content_rank_ok:
            blockers.append("content must be one-dimensional [D]")
        if not shape_matches:
            blockers.append("content shape does not match vector length")
        if not finite_content:
            blockers.append("content contains non-finite values")
        if not confidence_ok:
            blockers.append("confidence must be finite and in [0,1]")
        if not triplet_ok:
            blockers.append("triplet_index must be 0/1/2")
        if opts.require_write_permission_false and not write_permission_false:
            blockers.append("write_permission must remain false in WRITE-PREP-5")
        ready = bool(not blockers and preview.validation_ready)
        if not ready:
            validation.warning("prep5.live_shape.not_ready", "live-shape contract preview is not ready", preview.preview_id)
        trace = trace_write_prep("qdt_live_shape_contract_harness.check", validation, {
            "preview_id": preview.preview_id,
            "proposal_id": preview.proposal_id,
            "live_shape_ready": ready,
            "blockers": tuple(blockers),
            "write_permission_false": write_permission_false,
            "live_write_executed": False,
        })
        traces.append(trace)
        checks.append(LiveShapeContractCheck(
            check_id=write_prep5_result_id("live_shape_check", preview.preview_id, ready),
            preview_id=preview.preview_id,
            proposal_id=preview.proposal_id,
            contract_class_name=preview.contract_class_name,
            constructed=bool(preview.constructed),
            shape_matches=shape_matches,
            content_rank_ok=content_rank_ok,
            finite_content=finite_content,
            confidence_ok=confidence_ok,
            triplet_ok=triplet_ok,
            write_permission_false=write_permission_false,
            live_shape_ready=ready,
            blockers=tuple(blockers),
            trace_id=trace.trace_id,
            metadata={
                "stage": "HGM-QDT-WRITE-PREP-5",
                "dry_run": True,
                "contract_shape_only": True,
                "live_write_executed": False,
            },
        ))
    live_shape_ready = bool(checks and all(c.live_shape_ready for c in checks))
    final_trace = trace_write_prep("qdt_live_shape_contract_harness.build_live_shape_contract_harness", validation, {
        "check_count": len(checks),
        "live_shape_ready": live_shape_ready,
        "live_write_executed": False,
    })
    traces.append(final_trace)
    return LiveShapeContractHarnessResult(
        harness_id=write_prep5_result_id("live_shape_harness", tuple(c.check_id for c in checks)),
        checks=tuple(checks),
        live_shape_ready=live_shape_ready,
        validation=validation,
        trace_records=tuple(traces),
        metadata={
            "stage": "HGM-QDT-WRITE-PREP-5",
            "dry_run": True,
            "live_shape_contract_harness": True,
            "live_write_executed": False,
            "system_commitgate_stage_called": False,
            "system_commitgate_commit_called": False,
        },
    )
