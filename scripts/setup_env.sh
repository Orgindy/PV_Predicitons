#!/usr/bin/env bash
set -e

# Simple environment validation
if ! command -v python >/dev/null; then
    echo "Python is required but was not found" >&2
    exit 1
fi

PYVER=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PYVER" != "3.11" ]]; then
    echo "Warning: Python 3.11 recommended, found $PYVER" >&2
fi

# Ensure pip is up to date and install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
