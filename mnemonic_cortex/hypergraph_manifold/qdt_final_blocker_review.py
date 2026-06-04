"""Final production-write blocker review for WRITE-PREP-7."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .hgm_qdt_write_prep5_result import HGMQDTWritePrep5Result, ProductionWriteBlocker
from .hgm_qdt_write_prep6_result import HGMQDTWritePrep6Result
from .hgm_qdt_write_prep7_result import (
    FinalProductionWriteBlocker,
    FinalProductionWriteReadinessReview,
    HGMQDTWritePrep7Options,
    PermissionTokenContractResult,
    ShadowCommitSandboxResult,
    write_prep7_result_id,
)
from .hgm_qdt_write_prep_result import trace_write_prep
from .qdt_permission_token_contract import coerce_write_prep7_options
from .validation import ValidationResult


def _from_prep5(prep5_result: Any) -> list[FinalProductionWriteBlocker]:
    out: list[FinalProductionWriteBlocker] = []
    if not isinstance(prep5_result, HGMQDTWritePrep5Result):
        return out
    for b in prep5_result.blocker_burndown.blockers:
        if not isinstance(b, ProductionWriteBlocker):
            continue
        out.append(FinalProductionWriteBlocker(
            blocker_id=write_prep7_result_id("final_from_prep5", b.blocker_id),
            blocker_code=b.blocker_code,
            severity=b.severity,
            status=b.status,
            description=b.description,
            required_resolution=b.required_resolution,
            evidence=tuple(b.evidence),
            trace_id=b.trace_id,
            metadata={"source_stage": "WRITE-PREP-5", **dict(b.metadata or {})},
        ))
    return out


def build_final_production_write_readiness_review(
    permission_contract: Any,
    shadow_sandbox: Any,
    prep6_result: Any = None,
    prep5_result: Any = None,
    config=None,
    options: Optional[HGMQDTWritePrep7Options | Mapping[str, Any]] = None,
) -> FinalProductionWriteReadinessReview:
    """Build final blocker review without enabling production writes."""
    opts = coerce_write_prep7_options(options)
    validation = ValidationResult()
    traces = []
    blockers: list[FinalProductionWriteBlocker] = _from_prep5(prep5_result)
    token_ready = isinstance(permission_contract, PermissionTokenContractResult) and permission_contract.token_contract_ready
    shadow_ready = isinstance(shadow_sandbox, ShadowCommitSandboxResult) and shadow_sandbox.shadow_success
    if permission_contract is None or not isinstance(permission_contract, PermissionTokenContractResult):
        validation.error("prep7.final_review.invalid_permission_contract", "PermissionTokenContractResult is required", "permission_contract")
        token_ready = False
    else:
        validation.merge(permission_contract.validation)
        traces.extend(permission_contract.trace_records)
    if shadow_sandbox is None or not isinstance(shadow_sandbox, ShadowCommitSandboxResult):
        validation.error("prep7.final_review.invalid_shadow_sandbox", "ShadowCommitSandboxResult is required", "shadow_sandbox")
        shadow_ready = False
    else:
        validation.merge(shadow_sandbox.validation)
        traces.extend(shadow_sandbox.trace_records)
    if isinstance(prep6_result, HGMQDTWritePrep6Result):
        validation.merge(prep6_result.validation)
    elif prep6_result is not None:
        validation.warning("prep7.final_review.invalid_prep6", "prep6_result was supplied but is not HGMQDTWritePrep6Result", "prep6_result")

    def add_blocker(code: str, severity: str, status: str, desc: str, resolution: str, evidence=()):
        trace = trace_write_prep("qdt_final_blocker_review.blocker", validation, {
            "blocker_code": code,
            "severity": severity,
            "status": status,
            "description": desc,
        })
        traces.append(trace)
        blockers.append(FinalProductionWriteBlocker(
            blocker_id=write_prep7_result_id("final_blocker", code, status, tuple(evidence)),
            blocker_code=code,
            severity=severity,
            status=status,
            description=desc,
            required_resolution=resolution,
            evidence=tuple(evidence),
            trace_id=trace.trace_id,
            metadata={"stage": "HGM-QDT-WRITE-PREP-7"},
        ))

    add_blocker(
        "PERMISSION_TOKEN_CONTRACT",
        "HIGH",
        "open" if not token_ready else "resolved",
        "Production write execution requires an explicit permission token and human approval marker.",
        "Implement a signed/bounded permission-token gate in a later explicit write stage.",
        (getattr(permission_contract, "contract_id", "missing_permission_contract"),),
    )
    add_blocker(
        "SHADOW_COMMIT_SANDBOX",
        "MEDIUM",
        "resolved" if shadow_ready else "open",
        "Shadow commit sandbox must simulate stage/commit semantics without live mutation.",
        "Keep passing sandbox parity while adding real adapter boundary review.",
        (getattr(shadow_sandbox, "sandbox_id", "missing_shadow_sandbox"),),
    )
    rollback_ready = isinstance(prep6_result, HGMQDTWritePrep6Result) and prep6_result.rollback_binding_dry_run.binding_ready
    add_blocker(
        "ROLLBACK_SNAPSHOT_LIVE_BINDING",
        "HIGH",
        "open" if not rollback_ready else "resolved",
        "Rollback snapshot binding is still preview-only and not attached to a live rollback_stack.",
        "Bind to real rollback snapshots only in an explicit permissioned stage.",
        (getattr(getattr(prep6_result, "rollback_binding_dry_run", None), "dry_run_id", "missing_rollback_dryrun"),),
    )
    add_blocker(
        "PRODUCTION_WRITE_PERMISSION_STAGE",
        "HIGH",
        "open",
        "No production write stage has been approved or enabled.",
        "Run a separate explicit write-execution authorization stage before any live write.",
        ("WRITE-PREP-7 remains non-mutating",),
    )
    if len(blockers) > opts.max_final_blockers:
        validation.warning("prep7.final_review.bounded_blockers", "blocker count exceeded max_final_blockers; records truncated", "blockers")
        blockers = blockers[: opts.max_final_blockers]
    open_count = sum(1 for b in blockers if str(b.status).lower() != "resolved")
    resolved_count = sum(1 for b in blockers if str(b.status).lower() == "resolved")
    high_open = sum(1 for b in blockers if str(b.status).lower() != "resolved" and str(b.severity).upper() == "HIGH")
    production_ready = bool(token_ready and shadow_ready and open_count == 0 and high_open == 0)
    # This stage must not make production ready even if inputs are artificially marked ready.
    if production_ready:
        validation.warning("prep7.final_review.production_blocked_by_stage_policy", "WRITE-PREP-7 cannot mark production writes ready", "production_write_ready")
    production_ready = False
    trace = trace_write_prep("qdt_final_blocker_review.build_final_production_write_readiness_review", validation, {
        "open_blocker_count": open_count,
        "resolved_blocker_count": resolved_count,
        "high_severity_open_count": high_open,
        "shadow_commit_ready": shadow_ready,
        "permission_token_ready": token_ready,
        "production_write_ready": production_ready,
    })
    traces.append(trace)
    return FinalProductionWriteReadinessReview(
        review_id=write_prep7_result_id("final_write_readiness_review", tuple(b.blocker_id for b in blockers), open_count, high_open),
        blockers=tuple(blockers),
        open_blocker_count=open_count,
        resolved_blocker_count=resolved_count,
        high_severity_open_count=high_open,
        shadow_commit_ready=shadow_ready,
        permission_token_ready=token_ready,
        production_write_ready=production_ready,
        validation=validation,
        trace_records=tuple(traces),
        metadata={
            "stage": "HGM-QDT-WRITE-PREP-7",
            "production_write_ready": False,
            "write_execution_authorized": False,
        },
    )
