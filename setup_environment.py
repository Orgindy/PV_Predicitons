import os
import logging
from pathlib import Path
from config_utils import get_path
import yaml
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f) or {}
except FileNotFoundError:
    config = {}


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

results_dir = get_path("results_path")
REQUIRED_DIRS = [
    os.path.join(results_dir, "data"),
    results_dir,
    os.path.join(results_dir, "maps"),
    get_path("smarts_inp_path"),
    get_path("smarts_out_path"),
    os.path.join(results_dir, "spectral_analysis_output"),
]

for d in REQUIRED_DIRS:
    path = Path(d)
    path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created directory: {path}")

