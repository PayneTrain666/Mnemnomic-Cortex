from tests.qspin_prod2_module_loader import load_prod2_modules
mods=load_prod2_modules(); payload=mods["qspin_payload_dry_run"]; busm=mods["qspin_runtime_shadow_bus"]; simm=mods["qspin_guarded_dispatch_sim"]
def test_full_prod2_simulation_flow():
    env=payload.QSpinPayloadDryRunEnvelopeSummary("dense","src","dst",payload.QSpinPayloadDryRunShapeSummary((2,3)),payload.QSpinPayloadDryRunBudgetSummary(24),(0,2))
    pr=payload.QSpinTraceSafePayloadSummarizer().dry_run(payload.QSpinPayloadDryRunRequest("p",env)); assert pr.decision.approved
    b=busm.QSpinRuntimeAdapterShadowBus(); b.register_adapter(busm.QSpinShadowAdapterRegistration("a",busm.QSpinShadowAdapterEndpoint("e1",busm.QSpinShadowAdapterKind.WM8_INTERNAL,"wm8","wm8"))); b.register_adapter(busm.QSpinShadowAdapterRegistration("b",busm.QSpinShadowAdapterEndpoint("e2",busm.QSpinShadowAdapterKind.DEPTH_BRIDGE,"d0","d1")))
    br=b.dispatch(busm.QSpinShadowBusDispatchRequest("bus",busm.QSpinShadowBusMessage("m","a","b",pr.trace.safe_summary["safe_hash"]),True,True,True,True,True,pr.decision.approved)); assert br.decision.allowed_simulation
    route=simm.QSpinBridgeDispatchRouteSummary(simm.QSpinBridgeDispatchKind.DEPTH_BRIDGE,"d0","d1",("d0","d1"))
    sr=simm.QSpinGuardedBridgeDispatchSimulator().simulate(simm.QSpinBridgeDispatchSimulationRequest("sim",route,pr.decision.approved,br.decision.allowed_simulation,True,True,True)); assert sr.decision.simulated and not sr.safety_report.commits
