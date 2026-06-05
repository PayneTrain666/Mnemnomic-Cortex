
from __future__ import annotations
import json, time
from pathlib import Path
import torch
from mnemonic_cortex.working_memory import QDTWorkingMemory,QDTWorkingMemoryConfig,QDTWMCompatibilityConfig,QDTWMCompatibilityWrapper

def _time_call(fn, iterations=2):
    durations=[]; last=None
    for _ in range(iterations):
        t0=time.perf_counter(); last=fn(); durations.append(time.perf_counter()-t0)
    return durations,last

def _summary(ds):
    return {'iterations':len(ds),'mean_ms':sum(ds)/len(ds)*1000.0,'min_ms':min(ds)*1000.0,'max_ms':max(ds)*1000.0}

def run_benchmarks(output_path=None):
    torch.set_num_threads(1)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    cfg=QDTWorkingMemoryConfig(input_dim=32,hidden_dim=64,num_depths=8,num_slots=8,num_heads=4,transformer_layers=1)
    wm=QDTWorkingMemory(cfg); x=torch.randn(2,5,32)
    res={'benchmark_name':'qdt_wm_maae_quality_deepened_smoke','config':{'input_dim':32,'hidden_dim':64,'num_depths':8,'num_slots':8,'num_heads':4,'batch':2,'tokens':5,'iterations':2},'latency_smoke':{},'trace_size_smoke':{}}
    for op in ['read','process','write']:
        ds,last=_time_call(lambda op=op: wm(x, operation=op, context_map_name='quantum_holographic', return_trace=True), iterations=2)
        y,trace=last; tj=json.dumps(trace, default=str)
        res['latency_smoke'][op]=_summary(ds)
        res['trace_size_smoke'][op]={'json_bytes':len(tj.encode('utf-8')),'trace_items':len(trace.get('items',[])),'output_shape':list(y.shape),'finite':bool(torch.isfinite(y).all().item())}
    wrapper=QDTWMCompatibilityWrapper(QDTWMCompatibilityConfig(input_dim=32,hidden_dim=64,num_depths=8,num_slots=8,num_heads=4))
    ds,last=_time_call(lambda: wrapper.process(x, context_map_name='quantum_holographic', return_trace=True), iterations=2)
    wy,wtrace=last
    res['compatibility_wrapper_latency_smoke']=_summary(ds)
    res['compatibility_wrapper_trace']={'output_shape':list(wy.shape),'finite':bool(torch.isfinite(wy).all().item()),'operation':wtrace.get('operation')}
    growth=[]
    for i in range(2):
        y,trace=wm(torch.randn(2,5,32), operation='write', context_map_name='quantum_holographic', return_trace=True)
        growth.append({'step':i,'finite':bool(torch.isfinite(y).all().item()),'shared_slot_records':wm.shared_slot_store.registry.to_dict()['record_count'],'qh_records':wm.qh_storage.trace_summary()['record_count'],'commit_decisions':wm.system_commit_gate.trace_summary()['decision_count']})
    res['slot_qh_commit_growth_smoke']=growth
    from mnemonic_cortex.working_memory import qdt_working_memory
    contract=qdt_working_memory.wm_qd5a_commit_cortex_contract()
    res['contract_smoke']={'qdt_commit_contract_trace_type':contract.get('trace_type'),'trace_governance':contract.get('paamax_metadata',{}).get('trace_governance')}
    res['pass']=all(v['finite'] for v in res['trace_size_smoke'].values()) and res['compatibility_wrapper_trace']['finite'] and all(v['finite'] for v in growth) and res['contract_smoke']['trace_governance'] is True
    if output_path: Path(output_path).write_text(json.dumps(res, indent=2), encoding='utf-8')
    return res
if __name__=='__main__':
    out='docs/qdt_wm_maae_quality/41_wm_qd6a_benchmark_results.json'
    r=run_benchmarks(out); print(json.dumps(r, indent=2)); raise SystemExit(0 if r.get('pass') else 1)
