import subprocess, sys

def test_prod8_manual_runner_outputs_pass(tmp_path):
    proc = subprocess.run([sys.executable, "-S", "prod8_manual_runner.py"], cwd=".", text=True, capture_output=True, check=True)
    assert "FAIL=0" in proc.stdout
