import os
import logging
from pathlib import Path
from config import PathConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

cfg = PathConfig.from_yaml(Path("config.yaml"))
REQUIRED_DIRS = [
    cfg.results_path,
    os.path.join(cfg.results_path, "maps"),
    cfg.smarts_inp_path,
    cfg.smarts_out_path,
    os.path.join(cfg.results_path, "spectral_analysis_output"),
]

for d in REQUIRED_DIRS:
    path = Path(d)
    path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created directory: {path}")

