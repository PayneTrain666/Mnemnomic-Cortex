import json

from mnemonic_cortex.reasoning_depth import PersistenceLineClosure, PersistenceLineClosureConfig


def test_reason4c_persistence_line_closure_metadata_only():
    report = PersistenceLineClosure(PersistenceLineClosureConfig.enabled_default()).close(lineage={"stage": "REASON-4C"}).to_dict()

    assert report["enabled"] is True
    assert report["decision_record"]["decision"] == "closed_metadata_only"
    assert report["safety_flags"]["real_store_write_performed"] is False
    assert report["safety_flags"]["permanent_memory_store_mutation"] is False
    json.dumps(report)
