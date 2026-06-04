"""Production write blocker burn-down register for HGM/QDT WRITE-PREP-5."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .hgm_qdt_write_prep4_result import HGMQDTWritePrep4Result
from .hgm_qdt_write_prep5_result import (
    HGMQDTWritePrep5Options,
    LiveShapeContractHarnessResult,
    PermissionedCommitBoundaryAuditResult,
    ProductionWriteBlocker,
    ProductionWriteBlockerBurnDownResult,
    write_prep5_result_id,
)
from .hgm_qdt_write_prep_result import trace_write_prep
from .qdt_live_shape_contract_harness import coerce_write_prep5_options
from .validation import ValidationResult


def _blocker(code: str, stage: str, severity: str, status: str, description: str, resolution: str, evidence: tuple[str, ...], validation: ValidationResult) -> ProductionWriteBlocker:
    trace = trace_write_prep("qdt_production_write_blocker_burndown.blocker", validation, {
        "blocker_code": code,
        "severity": severity,
        "status": status,
        "evidence": evidence,
    })
    return ProductionWriteBlocker(
        blocker_id=write_prep5_result_id("production_blocker", code, status, severity),
        source_stage=stage,
        blocker_code=code,
        severity=severity,
        status=status,
        description=description,
        required_resolution=resolution,
        evidence=evidence,
        trace_id=trace.trace_id,
        metadata={"stage": "HGM-QDT-WRITE-PREP-5", "live_write_executed": False},
    )


def build_production_write_blocker_burndown(
    live_shape_harness: LiveShapeContractHarnessResult | None = None,
    permission_boundary_audit: PermissionedCommitBoundaryAuditResult | None = None,
    prep4_result: HGMQDTWritePrep4Result | None = None,
    config=None,
    options: Optional[HGMQDTWritePrep5Options | Mapping[str, Any]] = None,
) -> ProductionWriteBlockerBurnDownResult:
    """Build a conservative production-write blocker register."""
    opts = coerce_write_prep5_options(options)
    validation = ValidationResult()
    blockers: list[ProductionWriteBlocker] = []
    if live_shape_harness is not None:
        validation.merge(live_shape_harness.validation)
    if permission_boundary_audit is not None:
        validation.merge(permission_boundary_audit.validation)

    live_shape_ready = bool(live_shape_harness and live_shape_harness.live_shape_ready)
    boundary_clean = bool(permission_boundary_audit and permission_boundary_audit.permission_boundary_clean)
    rollback_binding_ready = bool(prep4_result and prep4_result.rollback_binding_plan.ready_for_live_commit)
    rollback_bindings_exist = bool(prep4_result and prep4_result.rollback_binding_plan.bindings)

    blockers.append(_blocker(
        "LIVE_SHAPE_CONTRACT_HARNESS",
        "WRITE-PREP-5",
        "MEDIUM",
        "resolved" if live_shape_ready else "open",
        "SystemWriteProposal live-shape contract previews must validate as one-dimensional finite proposal content with safe permission flags.",
        "Keep live-shape harness passing across representative HGM payloads.",
        (f"live_shape_ready={live_shape_ready}",),
        validation,
    ))
    blockers.append(_blocker(
        "PERMISSIONED_COMMIT_BOUNDARY_AUDIT",
        "WRITE-PREP-5",
        "HIGH",
        "resolved" if boundary_clean else "open",
        "The commit boundary must prove stage/commit/store/QH/rollback mutation paths are blocked unless a later explicit write stage grants execution.",
        "Maintain boundary checks and add permission-token verification before any real adapter path.",
        (f"permission_boundary_clean={boundary_clean}",),
        validation,
    ))
    blockers.append(_blocker(
        "ROLLBACK_SNAPSHOT_ACTUAL_BINDING",
        "WRITE-PREP-4",
        "HIGH",
        "open" if not rollback_binding_ready else "resolved",
        "Rollback snapshot bindings are still preview/synthetic and are not bound to actual SystemCommitGate.rollback_stack snapshots.",
        "Bind preview rollback requirements to real rollback_stack snapshot refs inside an isolated adapter stage before live writes.",
        (f"rollback_bindings_exist={rollback_bindings_exist}", f"ready_for_live_commit={rollback_binding_ready}"),
        validation,
    ))
    blockers.append(_blocker(
        "REAL_SHARED_SLOT_STORE_SANDBOX_PARITY",
        "WRITE-PREP-3",
        "MEDIUM",
        "open",
        "Synthetic SharedSlotStore sandbox behavior has not been parity-checked against a permissioned isolated real SharedSlotStore instance.",
        "Create an isolated real-store test harness with no production state and compare traces/fingerprints against synthetic replay.",
        ("synthetic_store_only=True",),
        validation,
    ))
    blockers.append(_blocker(
        "QH_STORAGE_RECORD_LIVE_SCHEMA_PARITY",
        "WRITE-PREP-1",
        "HIGH",
        "open",
        "Q-spin/QH conversion remains a schema preview and has not created isolated QHStorageRecord objects in a non-production sandbox.",
        "Construct isolated QHStorageRecord sandbox objects and validate interference/rollback compatibility before writes.",
        ("qh_conversion_preview_only=True",),
        validation,
    ))
    blockers.append(_blocker(
        "PRODUCTION_WRITE_PERMISSION_TOKEN",
        "WRITE-PREP-2",
        "HIGH",
        "open",
        "No production-grade permission token/capability model exists for converting dry-run preview permission into real SystemWriteProposal.write_permission.",
        "Define explicit signed/scoped write permission token semantics and tests before live execution.",
        ("write_permission_default=False", "simulated_permission_only=True"),
        validation,
    ))
    if len(blockers) > opts.max_blockers:
        validation.warning("prep5.blockers.truncated", "production blockers truncated to max_blockers", str(opts.max_blockers))
        blockers = blockers[: opts.max_blockers]
    open_count = sum(1 for b in blockers if b.status != "resolved")
    resolved_count = sum(1 for b in blockers if b.status == "resolved")
    high_open = sum(1 for b in blockers if b.status != "resolved" and b.severity.upper() == "HIGH")
    production_ready = bool(open_count == 0 and live_shape_ready and boundary_clean and rollback_binding_ready)
    if not production_ready:
        validation.warning("prep5.production_not_ready", "production writes remain blocked by open blocker register", "blockers")
    trace = trace_write_prep("qdt_production_write_blocker_burndown.build_production_write_blocker_burndown", validation, {
        "open_blocker_count": open_count,
        "resolved_blocker_count": resolved_count,
        "high_severity_open_count": high_open,
        "production_write_ready": production_ready,
        "live_write_executed": False,
    })
    return ProductionWriteBlockerBurnDownResult(
        register_id=write_prep5_result_id("production_blocker_burndown", tuple(b.blocker_id for b in blockers)),
        blockers=tuple(blockers),
        open_blocker_count=open_count,
        resolved_blocker_count=resolved_count,
        high_severity_open_count=high_open,
        production_write_ready=production_ready,
        validation=validation,
        trace_records=(trace,),
        metadata={
            "stage": "HGM-QDT-WRITE-PREP-5",
            "production_write_ready": production_ready,
            "live_write_executed": False,
            "system_commitgate_stage_called": False,
            "system_commitgate_commit_called": False,
        },
    )
