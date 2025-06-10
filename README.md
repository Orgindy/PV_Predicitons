# PV Predictions

This repository contains a collection of scripts to process climate data and evaluate radiative cooling (RC) technologies combined with photovoltaic (PV) systems.

## Features
- ERA5 aggregation and preprocessing
- Clustering workflows for RC/PV potential
- Thermal modeling utilities
- Streamlit dashboard (`app.py`)

## Requirements
Install dependencies from `requirements.txt` before running any of the scripts or tests:

```bash
pip install -r requirements.txt
```

## Running Tests
Unit tests use `pytest`. Ensure the dependencies are installed first, then run:

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

### Multi-Year PV Technology Matching

Run the clustering and technology matching pipeline for several yearly datasets.
The script looks for `clustered_dataset_<year>.csv` files in a given directory
and produces technology-matched outputs for each year along with a combined
file.

```bash
python multi_year_controller.py
```

