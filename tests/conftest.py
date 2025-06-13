import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


@pytest.fixture(autouse=True)
def _set_nc_data_dir(tmp_path, monkeypatch):
    """Use a temporary directory for NetCDF files during tests."""
    monkeypatch.setenv("NC_DATA_DIR", str(tmp_path))
