from __future__ import annotations
import importlib.util, sys, types
from pathlib import Path
MODULES=("qspin_synthetic_payload_harness","qspin_synthetic_qh_shared_slot_sandbox","qspin_commit_gate_expanded_dry_run","qspin_runtime_safety_regression","qspin_prod4_observability")
def load_prod4_modules():
    root=Path(__file__).resolve().parents[1]
    wm=root/"mnemonic_cortex"/"working_memory"
    sys.modules.setdefault("mnemonic_cortex",types.ModuleType("mnemonic_cortex"))
    pkg=sys.modules.get("mnemonic_cortex.working_memory")
    if pkg is None:
        pkg=types.ModuleType("mnemonic_cortex.working_memory"); pkg.__path__=[str(wm)]; sys.modules["mnemonic_cortex.working_memory"]=pkg
    out={}
    for short in MODULES:
        full=f"mnemonic_cortex.working_memory.{short}"
        if full not in sys.modules:
            spec=importlib.util.spec_from_file_location(full, wm/f"{short}.py")
            mod=importlib.util.module_from_spec(spec); sys.modules[full]=mod; assert spec.loader; spec.loader.exec_module(mod)
        out[short]=sys.modules[full]
    return out
