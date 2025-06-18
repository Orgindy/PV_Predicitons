"""Utilities for monitoring and cleaning up system resources."""

from __future__ import annotations

import gc
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

import psutil


class ResourceMonitor:
    """Helper methods for tracking system resources."""

    MAX_MEMORY_PERCENT = float(os.getenv("MAX_MEMORY_PERCENT", 75))
    DISK_SPACE_MIN_GB = float(os.getenv("DISK_SPACE_MIN_GB", 10))
    CPU_THRESHOLD_PERCENT = float(os.getenv("CPU_THRESHOLD_PERCENT", 90))

    @staticmethod
    def check_memory_usage(threshold: Optional[float] = None) -> bool:
        """Return True if current memory usage is below the given threshold."""
        mem = psutil.virtual_memory()
        limit = threshold if threshold is not None else ResourceMonitor.MAX_MEMORY_PERCENT
        if mem.percent >= limit:
            logging.warning(
                f"Memory usage {mem.percent:.1f}% exceeds limit of {limit}%"
            )
            return False
        return True

    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """Return memory statistics in gigabytes and percent used."""
        mem = psutil.virtual_memory()
        gb = 1024 ** 3
        return {
            "total_gb": mem.total / gb,
            "available_gb": mem.available / gb,
            "used_gb": mem.used / gb,
            "percent_used": mem.percent,
        }

    @staticmethod
    def check_disk_space(path: str = "/") -> bool:
        """Return True if free disk space at *path* is above the minimum."""
        usage = psutil.disk_usage(path)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < ResourceMonitor.DISK_SPACE_MIN_GB:
            logging.warning(
                f"Disk space {free_gb:.1f}GB below minimum of {ResourceMonitor.DISK_SPACE_MIN_GB}GB"
            )
            return False
        return True

    @staticmethod
    def check_cpu_usage(threshold: Optional[float] = None) -> bool:
        """Return True if CPU usage is below the given threshold."""
        cpu = psutil.cpu_percent(interval=1)
        limit = threshold if threshold is not None else ResourceMonitor.CPU_THRESHOLD_PERCENT
        if cpu >= limit:
            logging.warning(
                f"CPU usage {cpu:.1f}% exceeds limit of {limit}%"
            )
            return False
        return True

    @staticmethod
    def check_system_resources() -> bool:
        """Check all monitored system resources."""
        memory_ok = ResourceMonitor.check_memory_usage()
        disk_ok = ResourceMonitor.check_disk_space()
        cpu_ok = ResourceMonitor.check_cpu_usage()
        return memory_ok and disk_ok and cpu_ok


class ResourceCleanup:
    """Context manager for cleaning up temporary resources."""

    @staticmethod
    @contextmanager
    def cleanup_context(tmp_dir: str | Path = "tmp"):
        try:
            yield
        finally:
            tmp_path = Path(tmp_dir)
            if tmp_path.exists():
                for path in tmp_path.glob("*"):
                    if path.is_file():
                        try:
                            path.unlink()
                        except OSError as exc:
                            logging.warning(f"Failed to delete {path}: {exc}")
            gc.collect()
