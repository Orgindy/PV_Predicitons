import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def _set_nc_data_dir(tmp_path, monkeypatch):
    """Use a temporary directory for NetCDF files during tests."""
    monkeypatch.setenv("NC_DATA_DIR", str(tmp_path))
