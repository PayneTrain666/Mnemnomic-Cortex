import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import subprocess, sys, os

def test_manual_runner_passes():
    r=subprocess.run([sys.executable, '-S', 'prod7_manual_runner.py'], cwd=os.path.dirname(os.path.dirname(__file__)), capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'FAIL=0' in r.stdout
