"""
Plain-language summary
----------------------
What this file is for: QSPIN bridge contract, gate, sandbox, or observability helper.
How it fits in the system: Documents and guards a future optional bridge; not part of normal live memory routing today.
Status: INERT
Important notes for non-coders: Project policy keeps QSPIN disabled unless a later stage explicitly authorizes guarded activation.

Technical notes (original):
QSPIN-PROD-8 production blocker burn-down planning.

Planning only. This module never closes production blockers without evidence and
never authorizes activation.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Mapping, Tuple
import json

class BlockerBurnDownMode(str, Enum):
    PLAN_ONLY = "plan_only"

class BlockerBurnDownStatus(str, Enum):
    PLANNED_HOLD = "planned_hold"
    BLOCKED = "blocked"

class BlockerBurnDownPriority(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"

class BlockerBurnDownCategory(str, Enum):
    LIVE_ROUTING_ENABLEMENT = "live_routing_enablement"
    REAL_PAYLOAD_TRANSFER_ENABLEMENT = "real_payload_transfer_enablement"
    REAL_WRITE_PATH_ENABLEMENT = "real_write_path_enablement"
    QH_REAL_STORE_INTEGRATION = "qh_real_store_integration"
    SHARED_SLOT_REAL_STORE_INTEGRATION = "shared_slot_real_store_integration"
    EXTERNAL_MEMORY_REAL_STORE_INTEGRATION = "external_memory_real_store_integration"
    REAL_COMMIT_GATE_EXECUTION = "real_commit_gate_execution"
    PRODUCTION_ACTIVATION_PATH = "production_activation_path"
    REAL_RUNTIME_INTEGRATION_VALIDATION = "real_runtime_integration_validation"
    PERFORMANCE_BENCHMARK_VALIDATION = "performance_benchmark_validation"
    SECURITY_REVIEW = "security_review"
    OPERATOR_APPROVAL = "operator_approval"
    ROLLBACK_LIVE_TEST = "rollback_live_test"
    OBSERVABILITY_LIVE_TEST = "observability_live_test"
    LIVE_CANARY_PLAN = "live_canary_plan"
    INCIDENT_RESPONSE_PLAN = "incident_response_plan"
    DATA_PRIVACY_REVIEW = "data_privacy_review"
    DEPLOYMENT_STAGING_REVIEW = "deployment_staging_review"

@dataclass(frozen=True)
class BlockerBurnDownItem:
    item_id: str
    category: BlockerBurnDownCategory
    priority: BlockerBurnDownPriority
    status: str
    owner_placeholder: str
    required_next_action: str
    acceptance_criteria: Tuple[str, ...]
    risk: str
    defer_reason: str = ""
    evidence: Tuple[str, ...] = ()

    def validate(self) -> "BlockerBurnDownItem":
        if not self.item_id or not self.owner_placeholder or not self.required_next_action:
            raise ValueError("blocker item requires id, owner placeholder, and next action")
        if self.status == "resolved" and not self.evidence:
            raise ValueError("resolved blocker requires evidence")
        if self.status == "deferred" and not self.defer_reason:
            raise ValueError("deferred blocker requires reason")
        if self.priority in (BlockerBurnDownPriority.P0, BlockerBurnDownPriority.P1) and not self.acceptance_criteria:
            raise ValueError("P0/P1 blockers require acceptance criteria")
        return self

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["category"] = self.category.value
        d["priority"] = self.priority.value
        return d

@dataclass(frozen=True)
class BlockerBurnDownMilestone:
    milestone_id: str
    title: str
    item_ids: Tuple[str, ...]
    exit_criteria: Tuple[str, ...]

    def validate(self) -> "BlockerBurnDownMilestone":
        if not self.milestone_id or not self.title or not self.exit_criteria:
            raise ValueError("milestone requires id, title, and exit criteria")
        return self

@dataclass(frozen=True)
class BlockerBurnDownPlan:
    mode: BlockerBurnDownMode
    items: Tuple[BlockerBurnDownItem, ...]
    milestones: Tuple[BlockerBurnDownMilestone, ...]
    minimum_safe_activation_gates: Tuple[str, ...]

    def validate(self) -> "BlockerBurnDownPlan":
        for item in self.items:
            item.validate()
        for milestone in self.milestones:
            milestone.validate()
        if not self.minimum_safe_activation_gates:
            raise ValueError("minimum safe activation gates required")
        return self

@dataclass(frozen=True)
class BlockerBurnDownReport:
    status: BlockerBurnDownStatus
    open_blockers: Tuple[str, ...]
    p0_blockers: Tuple[str, ...]
    p1_blockers: Tuple[str, ...]
    deferred_blockers: Tuple[str, ...]
    resolved_blockers: Tuple[str, ...]
    recommended_sequence: Tuple[str, ...]
    minimum_safe_activation_gates: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": self.status.value,
            "open_blockers": list(self.open_blockers),
            "p0_blockers": list(self.p0_blockers),
            "p1_blockers": list(self.p1_blockers),
            "deferred_blockers": list(self.deferred_blockers),
            "resolved_blockers": list(self.resolved_blockers),
            "recommended_sequence": list(self.recommended_sequence),
            "minimum_safe_activation_gates": list(self.minimum_safe_activation_gates),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        lines = ["# PROD-8 Production Blocker Burn-Down Plan", "", f"Status: `{self.status.value}`", "", "## P0 blockers"]
        lines.extend(f"- {x}" for x in self.p0_blockers)
        lines.append("\n## P1 blockers")
        lines.extend(f"- {x}" for x in self.p1_blockers)
        lines.append("\n## Minimum safe activation gates")
        lines.extend(f"- {x}" for x in self.minimum_safe_activation_gates)
        return "\n".join(lines) + "\n"

class ProductionBlockerBurnDownPlanner:
    def plan(self, plan: BlockerBurnDownPlan) -> BlockerBurnDownReport:
        plan.validate()
        open_items = tuple(item.item_id for item in plan.items if item.status in {"open", "blocked", "deferred"})
        p0 = tuple(item.item_id for item in plan.items if item.priority == BlockerBurnDownPriority.P0 and item.status != "resolved")
        p1 = tuple(item.item_id for item in plan.items if item.priority == BlockerBurnDownPriority.P1 and item.status != "resolved")
        deferred = tuple(item.item_id for item in plan.items if item.status == "deferred")
        resolved = tuple(item.item_id for item in plan.items if item.status == "resolved")
        sequence = tuple(m.milestone_id for m in plan.milestones)
        status = BlockerBurnDownStatus.BLOCKED if p0 else BlockerBurnDownStatus.PLANNED_HOLD
        return BlockerBurnDownReport(status, open_items, p0, p1, deferred, resolved, sequence, plan.minimum_safe_activation_gates)

def build_default_blocker_burndown_plan() -> BlockerBurnDownPlan:
    items: List[BlockerBurnDownItem] = []
    for idx, category in enumerate(BlockerBurnDownCategory):
        priority = BlockerBurnDownPriority.P0 if idx < 8 else (BlockerBurnDownPriority.P1 if idx < 14 else BlockerBurnDownPriority.P2)
        items.append(BlockerBurnDownItem(
            item_id=f"BB-{idx+1:03d}",
            category=category,
            priority=priority,
            status="open",
            owner_placeholder="TBD-owner",
            required_next_action=f"Create controlled evidence plan for {category.value}.",
            acceptance_criteria=(f"Independent evidence demonstrates {category.value} without violating safety gates.",),
            risk="Production activation must remain blocked until this item is resolved.",
        ))
    milestones = (
        BlockerBurnDownMilestone("M1", "Evidence planning", tuple(i.item_id for i in items[:6]), ("All P0 evidence plans approved.",)),
        BlockerBurnDownMilestone("M2", "Read-only live validation", tuple(i.item_id for i in items[6:12]), ("Read-only validation complete.",)),
        BlockerBurnDownMilestone("M3", "Activation readiness review", tuple(i.item_id for i in items[12:]), ("All remaining blockers resolved or explicitly deferred with approval.",)),
    )
    gates = (
        "explicit future write-permission command",
        "security review complete",
        "operator approval complete",
        "live rollback test complete",
        "observability live test complete",
        "commit-gate production execution validated",
    )
    return BlockerBurnDownPlan(BlockerBurnDownMode.PLAN_ONLY, tuple(items), milestones, gates).validate()
