from dataclasses import dataclass
from pathlib import Path
import os
import yaml
from typing import Dict, Any, Optional

CONFIG_FILE = Path(__file__).with_name("config.yaml")


def load_config(path: Path = CONFIG_FILE) -> Dict[str, Any]:
    """Load configuration values from a YAML file."""
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}


CONFIG = load_config()


def get_path(key: str, default: Optional[str] = None) -> Optional[str]:
    """Return a configured path by key."""
    return CONFIG.get(key, default)

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


@dataclass
class TrainingConfig:
    """Paths for training datasets."""

    train_features: str = "data/X_train.npy"
    test_features: str = "data/X_test.npy"
    train_target: str = "data/y_train.npy"
    test_target: str = "data/y_test.npy"

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingConfig":
        if not path.exists():
            return cls()
        with path.open("r") as f:
            data = yaml.safe_load(f) or {}
        return cls(**{**cls().__dict__, **data.get("training", {})})


def get_nc_dir() -> str:
    """Return directory containing NetCDF files."""
    env_dir = os.getenv("NC_DATA_DIR")
    if env_dir:
        return env_dir

    return get_path("era5_path", "netcdf_files")
