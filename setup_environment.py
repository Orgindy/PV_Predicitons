import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    logging.info(f"Created directory: {path}")

