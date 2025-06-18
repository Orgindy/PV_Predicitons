import os
from pathlib import Path

REQUIRED_DIRS = [
    "data",
    "results",
    "results/maps",
    "smarts_inp_files",
    "smarts_out_files",
    "spectral_analysis_output",
]

for d in REQUIRED_DIRS:
    path = Path(d)
    path.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {path}")

