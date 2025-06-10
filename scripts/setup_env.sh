#!/usr/bin/env bash
set -e

# Ensure pip is up to date and install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
