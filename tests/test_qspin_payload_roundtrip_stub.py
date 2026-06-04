from tests.qspin_prod3_module_loader import load_prod3_modules
mods = load_prod3_modules()
rt = mods["qspin_payload_roundtrip_stub"]

def mk(kind):
    return rt.QSpinPayloadRoundtripRequest(
        "req_" + kind.value, kind, "src", "dst", (2, 4), declared_bytes=32
    )

def test_dense_chrr_qh_roundtrip_metadata_only():
    for kind in [rt.QSpinPayloadCodecStubKind.DENSE, rt.QSpinPayloadCodecStubKind.CHRR, rt.QSpinPayloadCodecStubKind.QH]:
        result = rt.QSpinPayloadCodecRoundtripStub().roundtrip(mk(kind))
        assert result.decision.approved is True
        assert result.metadata_in_hash == result.metadata_out_hash
        assert result.transferred_payload is False
        assert result.stored_payload is False
        assert result.wrote_state is False

def test_roundtrip_rejects_raw_payload_tensor_transfer_write():
    base = dict(request_id="bad", stub_kind=rt.QSpinPayloadCodecStubKind.DENSE, source_id="src", target_id="dst", declared_shape=(2,), declared_bytes=8)
    assert rt.QSpinPayloadCodecRoundtripStub().roundtrip(rt.QSpinPayloadRoundtripRequest(**base, raw_payload={"x": 1})).decision.approved is False
    assert rt.QSpinPayloadCodecRoundtripStub().roundtrip(rt.QSpinPayloadRoundtripRequest(**base, raw_tensor=object())).decision.approved is False
    assert rt.QSpinPayloadCodecRoundtripStub().roundtrip(rt.QSpinPayloadRoundtripRequest(**base, transfer_requested=True)).decision.approved is False
    assert rt.QSpinPayloadCodecRoundtripStub().roundtrip(rt.QSpinPayloadRoundtripRequest(**base, write_requested=True)).decision.approved is False

def test_roundtrip_rejects_shape_budget_norm():
    assert rt.QSpinPayloadCodecRoundtripStub().roundtrip(rt.QSpinPayloadRoundtripRequest("s", rt.QSpinPayloadCodecStubKind.DENSE, "src", "dst", (0,), declared_bytes=8)).decision.approved is False
    assert rt.QSpinPayloadCodecRoundtripStub().roundtrip(rt.QSpinPayloadRoundtripRequest("b", rt.QSpinPayloadCodecStubKind.DENSE, "src", "dst", (2,), declared_bytes=999999999)).decision.approved is False
    assert rt.QSpinPayloadCodecRoundtripStub().roundtrip(rt.QSpinPayloadRoundtripRequest("n", rt.QSpinPayloadCodecStubKind.DENSE, "src", "dst", (2,), declared_bytes=8, norm_band=(2,1))).decision.approved is False
