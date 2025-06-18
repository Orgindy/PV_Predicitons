import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

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
        except (OSError, yaml.YAMLError) as e:
            print(f"Warning: could not read {config_path}: {e}")

    # 3. default
    return "netcdf_files"


@dataclass
class AppConfig:
    """Central application configuration loaded from the environment."""

    memory_limit: float = 75.0
    disk_space_min_gb: float = 10.0
    max_file_size: int = 100 * 1024 * 1024
    resource_check_interval: int = 60

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        return cls(
            memory_limit=float(os.getenv("MAX_MEMORY_PERCENT", 75)),
            disk_space_min_gb=float(os.getenv("DISK_SPACE_MIN_GB", 10)),
            max_file_size=int(os.getenv("MAX_FILE_SIZE", 100 * 1024 * 1024)),
            resource_check_interval=int(os.getenv("RESOURCE_CHECK_INTERVAL", 60)),
        )

    def validate(self) -> bool:
        """Validate configuration values."""
        return (
            self.memory_limit > 0
            and self.disk_space_min_gb > 0
            and self.max_file_size > 0
            and self.resource_check_interval > 0
        )
