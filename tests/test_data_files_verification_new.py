import os
from data_files_verification import inspect_file, MAX_FILE_SIZE

def test_skip_large_file(tmp_path):
    path = tmp_path / "big.grib"
    path.write_bytes(b"0" * (MAX_FILE_SIZE + 1))
    info = inspect_file(str(path))
    assert info["error"] == "file too large"
    assert not info["valid"]

def test_missing_file():
    info = inspect_file("does_not_exist.grib")
    assert info["error"] == "file not found"
    assert not info["valid"]
