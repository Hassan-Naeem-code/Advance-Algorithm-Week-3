import subprocess
import sys
from pathlib import Path


def test_run_pipeline(tmp_path):
    figures_dir = tmp_path / "figures_test"
    cmd = [sys.executable, "src/main.py", "--figures-dir", str(figures_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    assert res.returncode == 0
    # Expect some output files
    assert figures_dir.exists()
