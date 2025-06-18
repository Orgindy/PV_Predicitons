import subprocess
import sys
from pathlib import Path


def test_check_db_connection(tmp_path):
    db_url = f"sqlite:///{tmp_path}/test.db"
    script = Path(__file__).resolve().parents[1] / "check_db_connection.py"
    result = subprocess.run(
        [sys.executable, str(script), "--db-url", db_url],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Successfully connected" in result.stdout
