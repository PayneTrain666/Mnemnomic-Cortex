from pathlib import Path
import traceback
files = [
    "tests/test_qspin_commit_gate_approval_sim.py",
    "tests/test_qspin_payload_roundtrip_stub.py",
    "tests/test_qspin_permission_dry_run.py",
    "tests/test_qspin_active_dry_run_executor.py",
    "tests/test_qspin_prod3_observability.py",
    "tests/test_qspin_prod3_source_consideration_matrix.py",
    "tests/test_qspin_prod3_integration_active_dry_run.py",
    "tests/test_qspin_prod3_failure_abuse_cases.py",
]
passed = 0
failed = 0
for path in files:
    ns = {"__file__": str(Path(path).resolve())}
    try:
        code = Path(path).read_text(encoding="utf-8")
        exec(compile(code, path, "exec"), ns)
        for name, fn in sorted(ns.items()):
            if name.startswith("test_") and callable(fn):
                try:
                    fn()
                    passed += 1
                except Exception:
                    failed += 1
                    print("FAILED", path, name)
                    traceback.print_exc()
    except Exception:
        failed += 1
        print("FAILED_LOAD", path)
        traceback.print_exc()
print(f"PASSED {passed} FAILED {failed}")
raise SystemExit(1 if failed else 0)
