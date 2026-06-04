from tests.qspin_prod2_module_loader import load_prod2_modules
mods=load_prod2_modules(); busm=mods["qspin_runtime_shadow_bus"]
def build_bus():
    b=busm.QSpinRuntimeAdapterShadowBus();
    b.register_adapter(busm.QSpinShadowAdapterRegistration("a", busm.QSpinShadowAdapterEndpoint("e1", busm.QSpinShadowAdapterKind.WM8_INTERNAL,"wm8","wm8")))
    b.register_adapter(busm.QSpinShadowAdapterRegistration("b", busm.QSpinShadowAdapterEndpoint("e2", busm.QSpinShadowAdapterKind.DEPTH_BRIDGE,"d0","d1")))
    return b
def good_req():
    msg=busm.QSpinShadowBusMessage("m","a","b","hash")
    return busm.QSpinShadowBusDispatchRequest("r",msg,True,True,True,True,True,True)
def test_shadow_bus_simulated_dispatch_and_idempotency():
    b=build_bus(); r=b.dispatch(good_req()); r2=b.dispatch(good_req())
    assert r.decision.allowed_simulation and not r.routed_live_data and r2.decision.status is busm.QSpinShadowBusStatus.IDEMPOTENT_REPLAY
def test_duplicate_adapter_rejected():
    b=build_bus()
    try: b.register_adapter(busm.QSpinShadowAdapterRegistration("a", busm.QSpinShadowAdapterEndpoint("e3", busm.QSpinShadowAdapterKind.WM8_INTERNAL,"x","y")))
    except ValueError: return
    raise AssertionError("duplicate adapter should fail")
def test_missing_gate_and_live_requests_block():
    b=build_bus(); req=busm.QSpinShadowBusDispatchRequest("bad",busm.QSpinShadowBusMessage("m2","a","b","hash"),False,False,False,False,False,False,live_routing_requested=True,payload_transfer_requested=True,write_requested=True,commit_requested=True,production_activation_requested=True)
    res=b.dispatch(req); assert not res.decision.allowed_simulation and len(res.decision.block_reasons)>=6
