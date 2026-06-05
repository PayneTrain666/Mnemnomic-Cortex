"""Shared test loader for QSPIN PROD-8 modules."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def load_module(short_name: str):
    if not short_name:
        raise ValueError("short_name is required")
    root = Path(__file__).resolve().parent
    wm_dir = root / "mnemonic_cortex" / "working_memory"
    full = f"mnemonic_cortex.working_memory.{short_name}"
    sys.modules.setdefault("mnemonic_cortex", types.ModuleType("mnemonic_cortex"))
    pkg = sys.modules.get("mnemonic_cortex.working_memory")
    if pkg is None:
        pkg = types.ModuleType("mnemonic_cortex.working_memory")
        pkg.__path__ = [str(wm_dir)]
        sys.modules["mnemonic_cortex.working_memory"] = pkg
    if full not in sys.modules:
        spec = importlib.util.spec_from_file_location(full, wm_dir / f"{short_name}.py")
        if spec is None or spec.loader is None:
            raise FileNotFoundError(f"Unable to locate module '{short_name}' under {wm_dir}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
    return sys.modules[full]
