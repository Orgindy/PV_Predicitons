# PV Predictions

This repository contains a collection of scripts to process climate data and evaluate radiative cooling (RC) technologies combined with photovoltaic (PV) systems.

## Features
- ERA5 aggregation and preprocessing
- Clustering workflows for RC/PV potential
- Thermal modeling utilities
- Streamlit dashboard (`app.py`)

## Requirements
Install dependencies from `requirements.txt` before running any of the scripts or tests.
You can do this manually or via the helper script in `scripts/setup_env.sh`:

```bash
# install packages manually
pip install -r requirements.txt

# or use the setup script
bash scripts/setup_env.sh
```

## Running Tests
Unit tests use `pytest`. Install the dependencies first (for example by running
`bash scripts/setup_env.sh` as shown above), then execute:

```bash
pytest -q
```

## Usage

Several scripts now accept file paths via command line arguments:

### Feature Preparation

```bash
python "Feature Preparation.py" \
  --input-file data/merged_dataset.csv \
  --validated-file data/validated_dataset.csv \
  --physics-file data/physics_dataset.csv \
  --netcdf-file data/processed_era5/ERA5_daily.nc \
  --results-dir results
```

### SMARTS Batch Processing

```bash
python run_smarts_batch.py \
  --smarts-exe /path/to/smarts295bat.exe \
  --inp-dir smarts_inp_files \
  --out-dir smarts_out_files
```

### Streamlit Dashboard

```bash
streamlit run app.py -- --data-path matched_dataset.csv
```

