from pathlib import Path
import importlib.util
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "mnemonic_cortex" / "working_memory"

def load_module(name: str):
    path = SRC / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
