import torch

from mnemonic_cortex.working_memory import (
    build_qh_code_schema,
    QHCodeSchema,
    QuantumHolographicStorage,
    QuantumHolographicStorageConfig,
    SharedSlotStore,
    SharedSlotStoreConfig,
)


def test_qh_code_schema_contains_required_codes_and_composite():
    schema = build_qh_code_schema(
        depth_index=2,
        bank_name="unit_bank",
        geometry_name="holographic_phase",
        triplet_index=1,
        memory_type="ltm",
        task_mode="quantum_holographic",
        num_depths=8,
    )
    d = schema.to_dict()
    assert d["depth_code"] == "depth-02"
    assert d["geometry_code"] == "geo-hphase"
    assert d["triplet_code"] == "triplet-direction"
    assert d["memory_type_code"] == "mem-ltm"
    assert d["task_mode_code"] == "task-qh"
    assert d["composite_code"].startswith("qh-")


def test_qh_code_schema_rejects_bad_depth_and_triplet():
    try:
        build_qh_code_schema(depth_index=99, bank_name="b", geometry_name="euclidean", triplet_index=0, memory_type="wm")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for bad depth")

    try:
        build_qh_code_schema(depth_index=0, bank_name="b", geometry_name="euclidean", triplet_index=99, memory_type="wm")
    except ValueError:
        return
    raise AssertionError("Expected ValueError for bad triplet")


def test_qh_record_creation_links_to_shared_slot_and_trace():
    store = SharedSlotStore(SharedSlotStoreConfig(namespace="unit_qh", dim=16))
    content = torch.randn(16)
    write = store.write_slot(
        memory_type="ltm",
        local_slot_id="ltm_slot",
        content=content,
        write_permission=True,
        geometry_map="holographic_phase",
    )
    qh = QuantumHolographicStorage(QuantumHolographicStorageConfig(dim=16, num_depths=8), shared_slot_store=store)
    record = qh.create_from_shared_slot(
        canonical_slot_id=write.canonical_id,
        depth_index=0,
        bank_name="ltm_bank",
        geometry_name="holographic_phase",
        triplet_index=0,
        memory_type="ltm",
        task_mode="quantum_holographic",
        write_permission=True,
    )

    assert record.canonical_slot_id == write.canonical_id
    assert record.code_schema.depth_code == "depth-00"
    assert record.write_permission_granted is True
    slot_trace = store.qh_trace_for_slot(write.canonical_id)
    assert slot_trace["qh_record_count"] == 1
    assert slot_trace["qh_refs"][0]["qh_record_id"] == record.record_id
    summary = qh.trace_summary()
    assert summary["record_count"] == 1
    assert summary["notice"] == "metadata_interface_only_no_quantum_hardware_claim"


def test_qh_interference_detection_marks_conflict():
    store = SharedSlotStore(SharedSlotStoreConfig(namespace="unit_qh2", dim=8))
    qh = QuantumHolographicStorage(
        QuantumHolographicStorageConfig(dim=8, num_depths=8, interference_threshold=0.95),
        shared_slot_store=store,
    )
    content = torch.ones(8)
    write1 = store.write_slot(memory_type="ltm", local_slot_id="a", content=content, write_permission=True)
    write2 = store.write_slot(memory_type="mann", local_slot_id="b", content=content.clone(), write_permission=True)

    rec1 = qh.create_from_shared_slot(
        canonical_slot_id=write1.canonical_id,
        depth_index=0,
        bank_name="bank",
        geometry_name="holographic_phase",
        triplet_index=0,
        memory_type="ltm",
        write_permission=True,
    )
    rec2 = qh.create_from_shared_slot(
        canonical_slot_id=write2.canonical_id,
        depth_index=0,
        bank_name="bank",
        geometry_name="holographic_phase",
        triplet_index=0,
        memory_type="mann",
        write_permission=True,
    )

    assert rec1.interference.interference_detected is False
    assert rec2.interference.interference_detected is True
    assert rec1.record_id in rec2.interference.conflicting_record_ids
    assert store.registry.get(write2.canonical_id).conflict_state == "quarantined"


def test_qh_rejects_bad_vector_shape_and_nonfinite():
    qh = QuantumHolographicStorage(QuantumHolographicStorageConfig(dim=8))
    bad = torch.randn(2, 8)
    try:
        qh.create_record(
            canonical_slot_id="css-abc",
            vector=bad,
            depth_index=0,
            bank_name="bank",
            geometry_name="euclidean",
            triplet_index=0,
            memory_type="wm",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for bad vector shape")

    nonfinite = torch.randn(8)
    nonfinite[0] = float("nan")
    try:
        qh.create_record(
            canonical_slot_id="css-abc",
            vector=nonfinite,
            depth_index=0,
            bank_name="bank",
            geometry_name="euclidean",
            triplet_index=0,
            memory_type="wm",
        )
    except ValueError:
        return
    raise AssertionError("Expected ValueError for nonfinite vector")


def test_qh_requires_write_permission_when_config_enforced():
    store = SharedSlotStore(SharedSlotStoreConfig(namespace="unit_qh_perm", dim=8))
    content = torch.randn(8)
    write = store.write_slot(memory_type="ltm", local_slot_id="perm_slot", content=content, write_permission=True)
    qh = QuantumHolographicStorage(
        QuantumHolographicStorageConfig(dim=8, require_write_permission=True),
        shared_slot_store=store,
    )
    try:
        qh.create_record(
            canonical_slot_id=write.canonical_id,
            vector=content,
            depth_index=0,
            bank_name="bank",
            geometry_name="holographic_phase",
            triplet_index=0,
            memory_type="ltm",
            write_permission=False,
        )
    except PermissionError:
        pass
    else:
        raise AssertionError("Expected PermissionError when write_permission=False")
    assert qh.trace_summary()["record_count"] == 0


def test_qh_rejects_unregistered_canonical_slot():
    store = SharedSlotStore(SharedSlotStoreConfig(namespace="unit_qh_slot", dim=8))
    qh = QuantumHolographicStorage(QuantumHolographicStorageConfig(dim=8), shared_slot_store=store)
    try:
        qh.create_record(
            canonical_slot_id="css-missing",
            vector=torch.randn(8),
            depth_index=0,
            bank_name="bank",
            geometry_name="holographic_phase",
            triplet_index=0,
            memory_type="ltm",
            write_permission=True,
        )
    except KeyError:
        pass
    else:
        raise AssertionError("Expected KeyError for unknown shared-slot canonical id")
    assert qh.trace_summary()["record_count"] == 0
