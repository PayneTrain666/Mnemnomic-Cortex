#!/usr/bin/env python3
"""
Plain-language summary
----------------------
What this file is for: Rewrites headers on a couple of maintenance scripts so they use the standard non-coder format.
How it fits in the system: Temporary documentation maintenance helper.
Status: WORKING (maintenance)
Important notes for non-coders: Safe to ignore unless you are updating documentation tooling.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hdr", ROOT / "tools" / "_apply_plain_language_headers.py")
hdr = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(hdr)

RELS = [
    "tools/_refresh_wm_headers.py",
    "tools/_apply_plain_language_headers.py",
]


def main() -> None:
    for rel in RELS:
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
            elif not old.startswith("Plain-language"):
                tech = old.strip()
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
        if text.startswith("#!") and not new_text.startswith("#!"):
            new_text = "#!/usr/bin/env python3\n" + new_text
        assert hdr.MARKER in (ast.get_docstring(ast.parse(new_text)) or "")
        path.write_text(new_text, encoding="utf-8")
        print(f"ok {rel} -> {status}")


if __name__ == "__main__":
    main()
