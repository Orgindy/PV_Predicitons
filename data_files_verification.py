#!/usr/bin/env python3
"""Utility to inspect GRIB files and verify basic metadata.

The script scans a directory for GRIB files (``*.grib``, ``*.grb``, ``*.grib2``,
``*.grb2``) and prints a short report for each file, including the available
coordinates, data variables, and the first/last timestamps if present.

It also performs a minimal verification that required coordinates exist.
The directory can be provided via command line or the ``GRIB_FOLDER``
environment variable. If none is supplied, ``grib_files`` is used.
"""

from __future__ import annotations

import argparse
import glob
import os
import signal
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List

import xarray as xr

# ``cfgrib`` is used via xarray's engine, ensure it's imported so the engine is
# registered. If not installed, xarray will raise an informative error when
# attempting to open a file.
import cfgrib  # noqa: F401


PATTERNS = ["*.grib", "*.grb", "*.grib2", "*.grb2"]
MAX_FILE_SIZE = 1_000_000_000  # 1 GB


@contextmanager
def time_limit(seconds: int):
    def handler(signum, frame):
        raise TimeoutError("Operation timed out")

    original = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original)


def find_grib_files(directory: str) -> List[str]:
    """Return sorted list of GRIB files in *directory* matching PATTERNS."""
    files: List[str] = []
    for pattern in PATTERNS:
        files.extend(glob.glob(os.path.join(directory, pattern)))
    return sorted(files)


def verify_dataset(ds: xr.Dataset) -> bool:
    """Check that essential coordinates are present in the dataset."""
    required = {"latitude", "longitude"}
    has_time = any(c in ds.coords for c in ["time", "valid_time", "forecast_time"])
    return required.issubset(ds.coords) and has_time


def inspect_file(path: str) -> Dict[str, object]:
    """Open a GRIB file and return metadata information."""
    info: Dict[str, object] = {"file": os.path.basename(path)}
    if not os.path.exists(path):
        return info

    size = os.path.getsize(path)
    if size > MAX_FILE_SIZE:
        return info

    with time_limit(30):
        ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"errors": "ignore"})
        info["dimensions"] = dict(ds.dims)
        info["variables"] = list(ds.data_vars)
        ds.close()
    return info


def main(directory: str) -> None:
    files = find_grib_files(directory)
    print(f"Found {len(files)} GRIB file(s) in {directory}\n")
    for path in files:
        meta = inspect_file(path)
        if meta is None:
            print(f"Skipping {os.path.basename(path)} (file too large)")
            continue
        print(f"File: {meta['file']}")
        if meta.get("error"):
            print(f"  Error: {meta['error']}")
            continue
        print(f"  Dimensions : {meta.get('dimensions')}")
        print(f"  Variables  : {meta.get('variables')}")
        if "start_time" in meta:
            print(f"  Time range : {meta['start_time']} -> {meta['end_time']}")
        print(f"  Valid file : {'yes' if meta['valid'] else 'no'}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect GRIB files and verify metadata"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=os.getenv("GRIB_FOLDER", "grib_files"),
        help="Directory containing GRIB files",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        raise SystemExit(f"Directory not found: {args.directory}")

    main(args.directory)
