from tests.qspin_prod2_module_loader import load_prod2_modules
mods=load_prod2_modules(); payload=mods["qspin_payload_dry_run"]; busm=mods["qspin_runtime_shadow_bus"]; simm=mods["qspin_guarded_dispatch_sim"]
def test_malformed_payload_request_fails_closed():
    r=payload.QSpinTraceSafePayloadSummarizer().dry_run(payload.QSpinPayloadDryRunRequest("x", None)); assert not r.decision.approved
def test_unsafe_adapter_registration_rejected():
    try: busm.QSpinShadowAdapterRegistration("x",busm.QSpinShadowAdapterEndpoint("e",busm.QSpinShadowAdapterKind.WM8_INTERNAL,"s","t"),mutates_runtime=True).validate()
    except ValueError: pass
    else: raise AssertionError("unsafe adapter should fail")
def test_dispatch_result_cannot_claim_live_effect():
    b=busm.QSpinRuntimeAdapterShadowBus(); msg=busm.QSpinShadowBusMessage("m","missing","missing2","hash"); req=busm.QSpinShadowBusDispatchRequest("r",msg,True,True,True,True,True,True); res=b.dispatch(req)
    try: busm.QSpinShadowBusDispatchResult(res.request,res.decision,res.trace,res.audit_event,routed_live_data=True).validate()
    except ValueError: pass
    else: raise AssertionError("live effect should fail")
def test_guarded_dispatch_live_flags_block():
    route=simm.QSpinBridgeDispatchRouteSummary(simm.QSpinBridgeDispatchKind.WM8_INTERNAL,"s","t",("s","t")); req=simm.QSpinBridgeDispatchSimulationRequest("bad",route,True,True,True,True,True,live_routing_requested=True,payload_transfer_requested=True,write_requested=True,commit_requested=True,production_activation_requested=True)
    res=simm.QSpinGuardedBridgeDispatchSimulator().simulate(req); assert not res.decision.simulated
