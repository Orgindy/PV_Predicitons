from dataclasses import dataclass
from pathlib import Path
import os
import yaml
from typing import Dict, Any, Optional

@dataclass
class AppConfig:
    """Application configuration."""

    # Resource limits
    memory_limit: float = 75.0
    disk_space_min_gb: float = 10.0
    max_file_size: int = 100 * 1024 * 1024
    cpu_max_percent: float = 90.0

    # Operational settings
    resource_check_interval: int = 60
    enable_file_locking: bool = True
    temp_cleanup_interval: int = 3600

    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Create config from environment variables."""
        return cls(
            memory_limit=float(os.getenv("MAX_MEMORY_PERCENT", cls.memory_limit)),
            disk_space_min_gb=float(os.getenv("DISK_SPACE_MIN_GB", cls.disk_space_min_gb)),
            max_file_size=int(os.getenv("MAX_FILE_SIZE", cls.max_file_size)),
            cpu_max_percent=float(os.getenv("CPU_MAX_PERCENT", cls.cpu_max_percent)),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> 'AppConfig':
        """Load config from YAML file."""
        if not path.exists():
            return cls()
        with path.open('r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def validate(self) -> Optional[str]:
        """Validate configuration values."""
        if not 0 < self.memory_limit <= 100:
            return "memory_limit must be between 0 and 100"
        if self.disk_space_min_gb <= 0:
            return "disk_space_min_gb must be positive"
        if self.max_file_size <= 0:
            return "max_file_size must be positive"
        if not 0 < self.cpu_max_percent <= 100:
            return "cpu_max_percent must be between 0 and 100"
        return None


def get_nc_dir() -> str:
    """Return directory containing NetCDF files."""
    env_dir = os.getenv("NC_DATA_DIR")
    if env_dir:
        return env_dir

    config_path = Path(__file__).with_name("config.yaml")
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if isinstance(data, Dict) and data.get("nc_data_dir"):
                return str(data["nc_data_dir"])
        except (OSError, yaml.YAMLError) as e:
            print(f"Warning: could not read {config_path}: {e}")

    return "netcdf_files"
