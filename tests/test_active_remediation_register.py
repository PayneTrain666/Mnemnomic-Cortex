import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemonic_cortex.working_memory.qspin_active_remediation_register import build_remediation_register, register_to_markdown

def test_empty_register_is_valid():
    items = build_remediation_register([], {"checks": []}, {"skipped_missing": []})
    assert items == []
    assert "No active remediation" in register_to_markdown(items)
