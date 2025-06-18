# Configuration Guide

The application reads configuration from environment variables or optional YAML files.
The `AppConfig` dataclass in `config.py` defines all tunable settings.

## Environment Variables
- `MAX_MEMORY_PERCENT` – maximum allowed memory usage before warnings.
- `DISK_SPACE_MIN_GB` – minimum required free disk space in gigabytes.
- `MAX_FILE_SIZE` – maximum file size for safe reading and writing.
- `CPU_MAX_PERCENT` – CPU usage threshold.

## YAML Configuration
Create a YAML file and load it with `AppConfig.from_yaml(path)` to override defaults.

All values are validated via the `validate()` method.
