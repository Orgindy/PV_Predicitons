import os
from pathlib import Path
import yaml


def get_nc_dir() -> str:
    """Return directory containing NetCDF files."""
    # 1. Environment variable takes precedence
    env_dir = os.getenv("NC_DATA_DIR")
    if env_dir:
        return env_dir

    # 2. config.yaml next to this file
    config_path = Path(__file__).with_name("config.yaml")
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if isinstance(data, dict) and data.get("nc_data_dir"):
                return str(data["nc_data_dir"])
        except Exception:
            pass

    # 3. default
    return "netcdf_files"
