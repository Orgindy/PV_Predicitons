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

### Linting

Run `flake8` to check for syntax errors in the codebase:

```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

## Database Integration

Several scripts can read from or write to a SQL database using SQLAlchemy. Set
the `PV_DB_URL` environment variable or use the `--db-url` and `--db-table`
options to enable this feature.

Example using SQLite:

```bash
export PV_DB_URL=sqlite:///path/to/pv.sqlite
python "Feature Preparation.py" --db-url $PV_DB_URL --db-table raw_pv_data
```

For PostgreSQL use a URL of the form `postgresql://user:pass@localhost/dbname`:

```bash
export PV_DB_URL=postgresql://user:secret@localhost/pvdb
python "Feature Preparation.py" --db-url $PV_DB_URL --db-table raw_pv_data
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


## Script Descriptions

- **Feature Preparation.py** – Prepares PV features from CSV or NetCDF, with optional database I/O.
- **Metadata Inspection.py** – Prints NetCDF metadata for inspection.
- **PV prediction.py** – Performs PV model training and cluster predictions.
- **RC_Clustering.py** – Clusters RC metrics and visualizes overlay maps.
- **Spatial Mapping.py** – Creates spatial interpolation maps from RC results.
- **aggregation_era5_parms_second_code.py** – Processes ERA5 NetCDF files and aggregates meteorological variables.
- **app.py** – Streamlit dashboard for exploring PV and RC data.
- **clustering.py** – Core clustering utilities used by the pipeline.
- **clustering_methods.py** – Wrapper functions to compare clustering algorithms.
- **compare_clustering_methods.py** – Script to evaluate clustering techniques.
- **create_smarts_inp.py** – Generates SMARTS input files from ERA5 data.
- **data_loader.py** – Helper for loading scenario arrays from disk.
- **database_utils.py** – SQLAlchemy helpers to read/write pandas DataFrames.
- **dynamic_materials.py** – Models emissivity changes for smart materials.
- **enhanced_thermal_model.py** – Computes surface temperature time series.
- **grib_to_cdf_new.py** – Converts GRIB weather data to NetCDF format.
- **humidity.py** – Calculates relative humidity via the Magnus formula.
- **main.py** – Runs the full RC–PV matching pipeline with optional DB support.
- **multi_year_controller.py** – Repeats the matching pipeline for several years.
- **pv_potential.py** – Calculates PV performance metrics.
- **rc_climate_zoning.py** – Creates RC climate zoning maps.
- **rc_cooling_combined_2025.py** – End-to-end RC potential computation and kriging.
- **run_smarts_batch.py** – Executes SMARTS simulations in parallel.
- **smarts_processor.py** – Parses SMARTS outputs to produce spectral datasets.
- **spectral_data_analisys.py** – Analyzes SMARTS spectra and produces plots.
- **synergy_index.py** – Computes synergy metrics between PV and RC technologies.
- **visualize_rc_maps.py** – Generates maps from RC datasets.
