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

## Database Integration

Several scripts can read from or write to a SQL database using SQLAlchemy. Set
the `PV_DB_URL` environment variable or use the `--db-url` and `--db-table`
options to enable this feature.

Example using SQLite:

```bash
export PV_DB_URL=sqlite:///path/to/pv.sqlite
python "Feature Preparation.py" --db-url $PV_DB_URL --db-table raw_pv_data
```

### Local database setup

If your database file is on your local drive, provide the absolute path with the
`sqlite:///` scheme. The same URL can be used with `main.py` or any script that
accepts the `--db-url` option:

```bash
export PV_DB_URL="sqlite:////full/path/to/pv.sqlite"
python main.py --db-url $PV_DB_URL --db-table pv_data

To quickly verify the connection works, run `check_db_connection.py`:
```bash
python check_db_connection.py --db-url $PV_DB_URL
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

