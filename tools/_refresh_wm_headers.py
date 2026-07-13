#!/usr/bin/env python3
"""
Plain-language summary
----------------------
What this file is for: Rewrites key working-memory file headers with more specific status text.
How it fits in the system: Maintenance helper used once overrides were added after the first header pass.
Status: WORKING (maintenance)
Important notes for non-coders: Only touches a short list of important WM files.

Technical notes (original):
Refresh key WM headers that missed overrides on the first pass.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "_apply_plain_language_headers.py"

spec = importlib.util.spec_from_file_location("hdr", SCRIPT)
hdr = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(hdr)

KEYS = [
    "mnemonic_cortex/working_memory/qdt_working_memory.py",
    "mnemonic_cortex/working_memory/wm_config.py",
    "mnemonic_cortex/working_memory/wm_cortex_integration.py",
    "mnemonic_cortex/working_memory/wm_compatibility_wrapper.py",
    "mnemonic_cortex/working_memory/wm_triple_hybrid_ltm_adapter.py",
    "mnemonic_cortex/working_memory/wm_quantum_holographic_storage.py",
    "mnemonic_cortex/working_memory/legacy_enhanced_curved_memory.py",
    "mnemonic_cortex/working_memory/qspin_experimental_live_activation.py",
]


def main() -> None:
    for rel in KEYS:
        path = ROOT / rel
        text = path.read_text(encoding="utf-8")
        what, fits, status, notes = hdr._guess(rel)
        header = hdr._header(what, fits, status, notes)
        info = hdr._existing_docstring(text)
        if info:
            old = info[0]
            tech = None
            if "Technical notes (original):" in old:
                tech = old.split("Technical notes (original):", 1)[1].strip()
            merged = header
            if tech:
                merged = (
                    header.rstrip()[:-3].rstrip()
                    + "\n\nTechnical notes (original):\n"
                    + tech
                    + '\n"""\n'
                )
            lines = text.splitlines(keepends=True)
            _, start, end = info
            new_text = "".join(lines[: start - 1]) + merged + "".join(lines[end:])
        else:
            new_text = header + "\n" + text
        docstring = ast.get_docstring(ast.parse(new_text))
        assert docstring and hdr.MARKER in docstring
        path.write_text(new_text, encoding="utf-8")
        print(f"refreshed {rel} -> {status}")


if __name__ == "__main__":
    main()
