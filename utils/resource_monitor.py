"""Utilities for monitoring system resources and handling errors."""
import os
import logging
import psutil
from typing import Dict, Optional


class ResourceMonitor:
    """Helper methods for tracking memory usage."""

    MAX_MEMORY_PERCENT = float(os.getenv("MAX_MEMORY_PERCENT", 90))

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
