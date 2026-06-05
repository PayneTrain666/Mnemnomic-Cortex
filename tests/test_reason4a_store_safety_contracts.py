import json

from mnemonic_cortex.reasoning_depth import (
    StoreSafetyContractBuilder,
    StoreSafetyContractConfig,
    StoreOperationKind,
)


def test_reason4a_store_safety_contracts_json_safe():
    builder = StoreSafetyContractBuilder(StoreSafetyContractConfig.enabled_default())
    contract = builder.build(operation_kind=StoreOperationKind.COMMIT, lineage={"stage": "REASON-4A"}).to_dict()

    assert contract["enabled"] is True
    assert contract["write_permission_required"] is True
    assert contract["safety_flags"]["permanent_memory_store_mutation"] is False
    assert contract["safety_level"] == "blocked"
    json.dumps(contract)
