from pathlib import Path
from utils.file_operations import SafeFileOps


def test_atomic_write(tmp_path):
    file_path = tmp_path / "test.txt"
    with SafeFileOps.atomic_write(file_path) as f:
        f.write("test")
    assert file_path.exists()
