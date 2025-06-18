from pathlib import Path
from utils.file_operations import SafeFileOps


def test_atomic_write(tmp_path):
    file_path = tmp_path / "test.txt"
    SafeFileOps.atomic_write(file_path, "test")
    assert file_path.exists()
    assert file_path.read_text() == "test"
