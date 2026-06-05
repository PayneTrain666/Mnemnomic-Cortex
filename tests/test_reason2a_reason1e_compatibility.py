from mnemonic_cortex.reasoning_depth import (
    depth_capacity_validation_contract,
    depth_integration_readiness_contract,
    depth_lattice_benchmark_contract,
    reasoning_controller_contract,
    consolidation_gate_contract,
)


def test_reason2a_reason1e_contracts_remain_available():
    assert depth_capacity_validation_contract()["capacity_multiplier"] == 8
    assert depth_integration_readiness_contract()["no_mutation_by_default"] is True
    assert depth_lattice_benchmark_contract()["production_benchmark_claim"] is False
    assert reasoning_controller_contract()["default_enabled"] is False
    assert consolidation_gate_contract()["permanent_commit"] is False
