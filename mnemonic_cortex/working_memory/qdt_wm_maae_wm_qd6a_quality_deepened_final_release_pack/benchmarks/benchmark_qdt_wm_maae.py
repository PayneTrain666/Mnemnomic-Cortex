from __future__ import annotations
import json, time
from pathlib import Path
import torch
from mnemonic_cortex.working_memory import QDTWorkingMemory, QDTWorkingMemoryConfig, QDTWMCompatibilityConfig, QDTWMCompatibilityWrapper

def run_benchmarks(output_path: str | None = None):
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    cfg = QDTWorkingMemoryConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4, transformer_layers=1)
    wm = QDTWorkingMemory(cfg)
    x = torch.randn(1, 3, 32)
    results = {'benchmark_name': 'qdt_wm_maae_wm7a_smoke_benchmark', 'config': {'batch': 1, 'tokens': 3, 'dim': 32, 'iterations': 1}}
    latency, trace_sizes = {}, {}
    for op in ['read', 'process', 'write']:
        t0 = time.perf_counter()
        out, trace = wm(x, operation=op, context_map_name='quantum_holographic', return_trace=True)
        dt = (time.perf_counter() - t0) * 1000.0
        js = json.dumps(trace, default=str)
        latency[op] = {'iterations': 1, 'mean_ms': dt, 'min_ms': dt, 'max_ms': dt}
        trace_sizes[op] = {'json_bytes': len(js.encode('utf-8')), 'trace_items': len(trace.get('items', [])), 'output_shape': list(out.shape), 'finite': bool(torch.isfinite(out).all().item())}
    growth = []
    for i in range(2):
        out, trace = wm(torch.randn(1, 3, 32), operation='write', context_map_name='quantum_holographic', return_trace=True)
        growth.append({'step': i, 'finite': bool(torch.isfinite(out).all().item()), 'shared_slot_records': wm.shared_slot_store.registry.to_dict()['record_count'], 'qh_records': wm.qh_storage.trace_summary()['record_count'], 'commit_decisions': wm.system_commit_gate.trace_summary()['decision_count']})
    wrapper = QDTWMCompatibilityWrapper(QDTWMCompatibilityConfig(input_dim=32, hidden_dim=64, num_depths=8, num_slots=8, num_heads=4))
    t0 = time.perf_counter(); out, tr = wrapper.process(x, context_map_name='quantum_holographic', return_trace=True); dt = (time.perf_counter() - t0) * 1000.0
    results['latency_smoke'] = latency
    results['trace_size_smoke'] = trace_sizes
    results['slot_qh_commit_gate_growth_smoke'] = growth
    results['compatibility_wrapper_process_latency'] = {'iterations': 1, 'mean_ms': dt, 'min_ms': dt, 'max_ms': dt}
    results['pass'] = all(v['finite'] for v in trace_sizes.values()) and all(v['finite'] for v in growth)
    if output_path:
        Path(output_path).write_text(json.dumps(results, indent=2), encoding='utf-8')
    return results

if __name__ == '__main__':
    out = Path('docs/qdt_wm_maae/101_wm7a_benchmark_results.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(run_benchmarks(str(out)), indent=2))
