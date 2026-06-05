from __future__ import annotations
import importlib.util, sys, types
from pathlib import Path

MODULE_ORDER = (
    "qspin_production_config", "qspin_production_plan",
    "qspin_commit_gate_dry_run", "qspin_kill_switch", "qspin_rollback_harness", "qspin_runtime_shadow_activation",
)

def load_prod1_modules():
    root = Path(__file__).resolve().parents[1]
    wm_dir = root / "mnemonic_cortex" / "working_memory"
    sys.modules.setdefault("mnemonic_cortex", types.ModuleType("mnemonic_cortex"))
    pkg = sys.modules.get("mnemonic_cortex.working_memory")
    if pkg is None:
        pkg = types.ModuleType("mnemonic_cortex.working_memory")
        pkg.__path__ = [str(wm_dir)]
        sys.modules["mnemonic_cortex.working_memory"] = pkg
    mods = {}
    for short in MODULE_ORDER:
        full = f"mnemonic_cortex.working_memory.{short}"
        if full not in sys.modules:
            spec = importlib.util.spec_from_file_location(full, wm_dir / f"{short}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[full] = mod
            assert spec.loader is not None
            spec.loader.exec_module(mod)
        mods[short] = sys.modules[full]
    return mods
