import subprocess, sys, json
from pathlib import Path

def test_prod6_manual_runner_passes():
    result = subprocess.run([sys.executable, '-S', 'prod6_manual_runner.py'], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'FAIL=0' in result.stdout
