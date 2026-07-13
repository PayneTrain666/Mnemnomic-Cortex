#!/usr/bin/env python3
"""
Plain-language summary
----------------------
What this file is for: Fixes header placement so the plain-language summary sits before future imports.
How it fits in the system: Maintenance helper for documentation correctness.
Status: WORKING (maintenance)
Important notes for non-coders: Needed because Python module docs must be the first statement.

Technical notes (original):
Fix module docstring order: Plain-language summary must precede __future__ imports.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = "qdt_wm_maae_wm_qd6a_quality_deepened_final_release_pack"
MARKER = "Plain-language summary"

DOC_RE = re.compile(
    r'(?s)^(from __future__ import [^\n]+\n(?:#.*\n|\s*)*)'
    r'("""\nPlain-language summary\n.*?""")\n?'
)


def main() -> None:
    fixed = 0
    failed = 0
    for base in (ROOT / "mnemonic_cortex", ROOT / "tools"):
        paths = base.rglob("*.py") if base.name == "mnemonic_cortex" else base.glob("*.py")
        for path in paths:
            if EXCLUDE in str(path):
                continue
            text = path.read_text(encoding="utf-8")
            if MARKER not in text:
                continue
            match = DOC_RE.match(text)
            if not match:
                continue
            future_block = match.group(1)
            doc = match.group(2)
            rest = text[match.end() :]
            new_text = doc + "\n\n" + future_block + rest.lstrip("\n")
            try:
                tree = ast.parse(new_text)
                docstring = ast.get_docstring(tree)
                assert docstring and MARKER in docstring
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {path}: {exc}")
                failed += 1
                continue
            path.write_text(new_text, encoding="utf-8")
            fixed += 1
    print(f"fixed_future_order={fixed} failed={failed}")


if __name__ == "__main__":
    main()
