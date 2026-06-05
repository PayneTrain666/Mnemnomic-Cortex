import json

from mnemonic_cortex.reasoning_depth import PersistenceRecoveryPlanner, PersistenceRecoveryConfig


def test_reason4b_recovery_plan_for_denied_record():
    planner = PersistenceRecoveryPlanner(PersistenceRecoveryConfig.enabled_default())
    plan = planner.plan(
        ledger_payload={
            "records": [
                {
                    "record_id": "r1",
                    "decision": {"approved": False, "status": "denied"},
                    "request": {"payload": {"payload_id": "p1"}},
                }
            ]
        }
    ).to_dict()

    assert plan["enabled"] is True
    assert plan["actions"][0]["action_kind"] == "quarantine_payload"
    assert plan["real_rollback_performed"] is False
    json.dumps(plan)
