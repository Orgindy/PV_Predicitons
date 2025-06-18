import os
import psutil
import logging
from typing import Dict, Optional
from pathlib import Path


class ResourceMonitor:
    """Comprehensive system resource monitoring."""

    # Configuration with safer defaults
    MAX_MEMORY_PERCENT = float(os.getenv("MAX_MEMORY_PERCENT", 75))
    DISK_SPACE_MIN_GB = float(os.getenv("DISK_SPACE_MIN_GB", 10))
    CPU_MAX_PERCENT = float(os.getenv("CPU_MAX_PERCENT", 90))

    @staticmethod
    def check_system_resources() -> Dict[str, bool]:
        """Check all system resources."""
        return {
            "memory": ResourceMonitor.check_memory_usage(),
            "disk": ResourceMonitor.check_disk_space(),
            "cpu": ResourceMonitor.check_cpu_usage(),
        }

    @staticmethod
    def check_memory_usage(threshold: Optional[float] = None) -> bool:
        """Check if memory usage is below threshold."""
        mem = psutil.virtual_memory()
        limit = threshold or ResourceMonitor.MAX_MEMORY_PERCENT
        if mem.percent >= limit:
            logging.warning(f"Memory usage {mem.percent:.1f}% exceeds {limit}%")
            return False
        return True

    @staticmethod
    def check_disk_space(path: Path = Path.cwd()) -> bool:
        """Check if disk space is sufficient."""
        disk = psutil.disk_usage(str(path))
        gb_free = disk.free / (1024**3)
        if gb_free < ResourceMonitor.DISK_SPACE_MIN_GB:
            logging.warning(f"Low disk space: {gb_free:.1f}GB free")
            return False
        return True

    @staticmethod
    def check_cpu_usage() -> bool:
        """Check if CPU usage is acceptable."""
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent >= ResourceMonitor.CPU_MAX_PERCENT:
            logging.warning(f"High CPU usage: {cpu_percent:.1f}%")
            return False
        return True


class ResourceCleanup:
    """Resource cleanup utilities."""

    @staticmethod
    def cleanup_temp_files(directory: Path) -> None:
        """Clean up temporary files in directory."""
        pattern = "*.tmp"
        for tmp_file in directory.glob(pattern):
            try:
                tmp_file.unlink()
            except Exception as e:
                logging.error(f"Failed to remove {tmp_file}: {e}")
