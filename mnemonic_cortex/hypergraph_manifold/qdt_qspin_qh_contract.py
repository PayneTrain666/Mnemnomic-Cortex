"""Q-spin placeholder to QH code conversion contract previews."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from .enums import DepthLayer, GeometryType
from .hgm4_result import HGMBridgePayload, SharedSlotLatticeHook, TraceSafeMemoryPlan
from .hgm_qdt_write_prep_result import (
    HGMQDTWritePrepOptions,
    QSpinQHConversionContract,
    QSpinQHConversionRecord,
    trace_write_prep,
    write_prep_stable_hash,
)
from .qdt_write_contract_probe import _coerce_options
from .validation import ValidationResult

_GEOMETRY_MAP = {
    GeometryType.EUCLIDEAN: "euclidean",
    GeometryType.HYPERBOLIC: "hyperbolic",
    GeometryType.SPHERICAL: "spherical",
    GeometryType.TORUS: "torus",
    GeometryType.COMPLEX_PROJECTIVE: "complex_projective",
    GeometryType.GRASSMANN: "grassmann",
    GeometryType.FISHER_SIMPLEX: "fisher_simplex",
    GeometryType.PRODUCT: "product",
    GeometryType.SPCP: "spcp",
}


def depth_to_qdt_index(depth_layer: Any) -> int:
    return int(DepthLayer.coerce(depth_layer).value)


def geometry_to_qdt_map(geometry_type: Any) -> str:
    geom = GeometryType.coerce(geometry_type)
    return _GEOMETRY_MAP.get(geom, geom.value)


def _records_from_input(records_or_plan: Any) -> tuple[Any, ...]:
    if isinstance(records_or_plan, TraceSafeMemoryPlan):
        return tuple(records_or_plan.bridge_payloads or tuple()) + tuple(records_or_plan.slot_hooks or tuple())
    if isinstance(records_or_plan, (HGMBridgePayload, SharedSlotLatticeHook)):
        return (records_or_plan,)
    if isinstance(records_or_plan, Iterable) and not isinstance(records_or_plan, (str, bytes, Mapping)):
        return tuple(r for r in records_or_plan if isinstance(r, (HGMBridgePayload, SharedSlotLatticeHook)))
    return tuple()


def _record_id(record: Any) -> str:
    return str(getattr(record, "payload_id", None) or getattr(record, "hook_id", None) or getattr(record, "source_record_id", "unknown"))


def _schema_preview(depth_index: int, bank_name: str, geometry_map: str, triplet_index: int, memory_type: str, task_mode: str) -> Mapping[str, Any]:
    try:
        from mnemonic_cortex.working_memory.wm_quantum_holographic_storage import build_qh_code_schema
        schema = build_qh_code_schema(
            depth_index=depth_index,
            bank_name=bank_name,
            geometry_name=geometry_map,
            triplet_index=triplet_index,
            memory_type=memory_type,
            task_mode=task_mode,
        )
        return schema.to_dict()
    except Exception:
        composite = f"qh-{write_prep_stable_hash(depth_index, bank_name, geometry_map, triplet_index, memory_type, task_mode)}"
        return {
            "depth_code": f"depth-{depth_index:02d}",
            "bank_code": f"bank-{write_prep_stable_hash(bank_name)}",
            "geometry_code": f"geo-{write_prep_stable_hash(geometry_map)}",
            "triplet_code": ["triplet-anchor", "triplet-direction", "triplet-phase"][triplet_index],
            "memory_type_code": f"mem-{write_prep_stable_hash(memory_type)}",
            "task_mode_code": f"task-{write_prep_stable_hash(task_mode)}",
            "composite_code": composite,
            "notice": "preview_schema_only_no_qh_storage_write",
        }


def build_qspin_qh_conversion_contract(records_or_plan: Any, config=None, options: Optional[HGMQDTWritePrepOptions | Mapping[str, Any]] = None) -> QSpinQHConversionContract:
    opts = _coerce_options(options)
    validation = ValidationResult()
    traces = []
    records = _records_from_input(records_or_plan)
    if not records:
        validation.warning("qspin_qh.empty_records", "no HGM bridge/hook records supplied", "records")
    conversions = []
    for record in records[: opts.max_payloads + opts.max_hooks]:
        rid = _record_id(record)
        depth = getattr(record, "depth_layer", DepthLayer.D5_PROCEDURAL)
        geom = getattr(record, "geometry_type", GeometryType.PRODUCT)
        qspin = str(getattr(record, "qspin_signature_id", "") or f"qspin_placeholder_{write_prep_stable_hash(rid)}")
        depth_index = depth_to_qdt_index(depth)
        geometry_map = geometry_to_qdt_map(geom)
        schema = _schema_preview(depth_index, opts.default_bank_name, geometry_map, opts.default_triplet_index, opts.default_memory_type, opts.default_task_mode)
        composite = str(schema.get("composite_code", f"qh-{write_prep_stable_hash(rid, qspin)}"))
        qh_record_id = f"qhrec-{write_prep_stable_hash(rid, qspin, composite, length=24)}"
        if qspin.startswith("qspin_placeholder"):
            validation.warning("qspin_qh.placeholder_qspin", "q-spin placeholder requires real QSpinSignature/QH conversion before write stage", rid)
        trace = trace_write_prep("qdt_qspin_qh_contract.conversion", validation, {"record_id": rid, "qspin_signature_id": qspin, "qh_record_id_preview": qh_record_id})
        traces.append(trace)
        conversions.append(QSpinQHConversionRecord(
            conversion_id=f"qspin_qh_{write_prep_stable_hash(rid, qspin)}",
            qspin_signature_id=qspin,
            depth_index=depth_index,
            geometry_map=geometry_map,
            memory_type=opts.default_memory_type,
            task_mode=opts.default_task_mode,
            bank_name=opts.default_bank_name,
            triplet_index=opts.default_triplet_index,
            qh_composite_code=composite,
            qh_record_id_preview=qh_record_id,
            schema_preview=schema,
            trace_id=trace.trace_id,
            metadata={"read_only": True, "qh_storage_record_created": False, "source_record_id": rid},
        ))
    final_trace = trace_write_prep("qdt_qspin_qh_contract.build_qspin_qh_conversion_contract", validation, {"conversion_count": len(conversions)})
    traces.append(final_trace)
    return QSpinQHConversionContract(
        contract_id=f"qspin_qh_contract_{write_prep_stable_hash(len(conversions))}",
        conversions=tuple(conversions),
        validation=validation,
        trace_records=tuple(traces),
        metadata={"read_only": True, "qh_storage_mutated": False},
    )
