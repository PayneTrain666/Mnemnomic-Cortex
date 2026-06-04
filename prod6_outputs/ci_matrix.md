# QSPIN-PROD-6 CI Matrix

Status: `passed`

| Case | Axis | Status | Reasons |
|---|---|---|---|
| ci_python_version | python_version | passed | synthetic_axis_ok |
| ci_os | os | passed | synthetic_axis_ok |
| ci_torch_available | torch_available | passed | synthetic_axis_ok |
| ci_pytest_available | pytest_available | passed | synthetic_axis_ok |
| ci_source_pack_available | source_pack_available | passed | synthetic_axis_ok |
| ci_previous_pack_available | previous_pack_available | passed | synthetic_axis_ok |
| ci_qh_write_attempt | qh_write_attempt | passed | blocked_unsafe_axis |
| ci_shared_slot_write_attempt | shared_slot_write_attempt | passed | blocked_unsafe_axis |
| ci_external_write_attempt | external_write_attempt | passed | blocked_unsafe_axis |
| ci_payload_transfer_attempt | payload_transfer_attempt | passed | blocked_unsafe_axis |
| ci_commit_attempt | commit_attempt | passed | blocked_unsafe_axis |
| ci_production_activation_attempt | production_activation_attempt | passed | blocked_unsafe_axis |
| ci_malformed_input | malformed_input | passed | synthetic_axis_ok |
| ci_timeout_pressure | timeout_pressure | passed | synthetic_axis_ok |
| ci_concurrency_pressure | concurrency_pressure | passed | synthetic_axis_ok |
| ci_redaction_pressure | redaction_pressure | passed | synthetic_axis_ok |
