from tests.qspin_prod2_module_loader import load_prod2_modules
m=load_prod2_modules()["qspin_guarded_dispatch_sim"]
def req(kind):
    route=m.QSpinBridgeDispatchRouteSummary(kind,"src","dst",("src","dst"))
    return m.QSpinBridgeDispatchSimulationRequest("r"+kind.value,route,True,True,True,True,True)
def test_all_bridge_kinds_simulate():
    sim=m.QSpinGuardedBridgeDispatchSimulator()
    for kind in m.QSpinBridgeDispatchKind:
        res=sim.simulate(req(kind)); assert res.decision.simulated and not res.safety_report.live_routing
def test_missing_approvals_block():
    route=m.QSpinBridgeDispatchRouteSummary(m.QSpinBridgeDispatchKind.WM8_INTERNAL,"s","t",("s","t"))
    res=m.QSpinGuardedBridgeDispatchSimulator().simulate(m.QSpinBridgeDispatchSimulationRequest("bad",route,False,False,False,False,False,live_routing_requested=True,payload_transfer_requested=True,write_requested=True,commit_requested=True,production_activation_requested=True))
    assert not res.decision.simulated and len(res.decision.block_reasons)>=5
