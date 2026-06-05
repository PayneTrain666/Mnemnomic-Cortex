
from __future__ import annotations
import importlib, json
from pathlib import Path
CONTRACT_GROUPS = {
    "WM-QD-1A": {"function":"wm_qd1a_foundation_contract", "modules":["context_geometry_maps","context_map_selector","context_to_wm_bridge","wm_context_mount","legacy_enhanced_curved_memory","wm_curved_core","curved_resonant_wm_core","curved_slot_state","curvature_metric_policy","geometry_aware_addressing","bounded_associative_spread","curved_local_trace","curved_shadow_write"]},
    "WM-QD-2A": {"function":"wm_qd2a_depth_contract", "modules":["wm_quaternion_depth","wm_intra_depth_transformer","wm_cross_depth_transformer","depth_specific_addressing","wm_depth_adapters","wm_depth_fusion","wm_triplet_state","wm_trace","qdt_working_memory"]},
    "WM-QD-3A": {"function":"wm_qd3a_attention_contract", "modules":["wm_retrieval_lanes","wm_geometry_scoring","wm_memory_augmented_attention","wm_geometry_linker","wm_evidence_attention","wm_trace_attention","wm_counterfactual_attention","wm_conflict_attention","wm_novelty_attention","wm_stability_attention"]},
    "WM-QD-4A": {"function":"wm_qd4a_external_memory_contract", "modules":["wm_external_memory_interfaces","wm_ltm_cross_attention","wm_mann_cross_attention","wm_spcp_cross_attention","wm_dual_fusion","wm_shared_slot_registry","wm_shared_slot_store","wm_quantum_holographic_storage"]},
    "WM-QD-5A": {"function":"wm_qd5a_commit_cortex_contract", "modules":["wm_system_commit_gate","wm_compatibility_wrapper","wm_cortex_integration","qdt_working_memory"]},
}
def verify_contracts():
    results={"groups":{}, "overall_pass": True}
    for group,cfg in CONTRACT_GROUPS.items():
        fn=cfg['function']; rows=[]
        for short in cfg['modules']:
            mod_name=f"mnemonic_cortex.working_memory.{short}"
            try:
                mod=importlib.import_module(mod_name)
                if not hasattr(mod, fn): raise AssertionError(f"missing {fn}")
                contract=getattr(mod, fn)(); json.dumps(contract)
                if contract.get('paamax_metadata',{}).get('trace_governance') is not True: raise AssertionError('trace_governance missing/false')
                rows.append({'module':mod_name,'contract_function':fn,'pass':True,'trace_type':contract.get('trace_type')})
            except Exception as exc:
                results['overall_pass']=False; rows.append({'module':mod_name,'contract_function':fn,'pass':False,'error':str(exc)})
        results['groups'][group]=rows
    return results
if __name__=='__main__':
    out=verify_contracts()
    Path('docs/qdt_wm_maae_quality/38_wm_qd6a_contract_verification.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))
    raise SystemExit(0 if out['overall_pass'] else 1)
