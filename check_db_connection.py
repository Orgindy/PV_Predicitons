import logging
import os
from database_utils import get_engine, DEFAULT_DB_URL
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from utils.errors import SynergyDatabaseError
from datetime import datetime
import pandas as pd
import xarray as xr


def _validate_data_path(path: str) -> None:
    """Validate a NetCDF or database file path."""
    if not os.path.exists(path):
        logging.error("Path does not exist: %s", path)
        raise FileNotFoundError(path)

    if os.path.isdir(path):
        entries = os.listdir(path)
        if not any(e.endswith((".nc", ".db")) for e in entries):
            logging.error("No .nc or .db files found in directory %s", path)
            raise FileNotFoundError(f"Missing .nc or .db files in {path}")
        logging.info("Validated directory %s", path)
        return

    if not path.endswith((".nc", ".db")):
        logging.error("Unsupported file extension for %s", path)
        raise ValueError("Expected .nc or .db file")

    if path.endswith(".nc"):
        try:
            with xr.open_dataset(path) as ds:
                required_vars = {"t2m", "ssrd"}
                missing = [v for v in required_vars if v not in ds.variables]
                if missing:
                    raise ValueError(f"Missing variables: {missing}")

                n_steps = len(ds.time)
                year = pd.to_datetime(ds.time.values[0]).year
                expected = 8784 if pd.Timestamp(year=year, month=12, day=31).is_leap_year else 8760
                if n_steps not in {8760, 8784}:
                    logging.warning("Unexpected number of time steps: %s", n_steps)
                elif n_steps != expected:
                    logging.warning(
                        "Time step count %s does not match expected %s for %s",
                        n_steps,
                        expected,
                        year,
                    )
        except Exception as exc:
            logging.error("Failed to validate NetCDF %s: %s", path, exc)
            raise

    logging.info("Validated file %s", path)


def main(db_url: str | None = None, path: str | None = None) -> int:
    """Check that a database or NetCDF path is reachable."""
    if not (db_url or path):
        logging.error("Either db_url or path must be provided")
        return 1

    if path:
        try:
            _validate_data_path(path)
            return 0
        except (FileNotFoundError, ValueError) as exc:
            logging.error("Data path validation failed: %s", exc)
            return 1

    url = db_url or DEFAULT_DB_URL
    try:
        engine = get_engine(url)
        if engine is None:
            logging.warning("Engine creation returned None")
            return 1
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        msg = "Successfully connected to the database."
        print(msg)
        logging.info(msg)
        return 0
    except (SQLAlchemyError, SynergyDatabaseError) as exc:
        logging.warning("Database connection failed: %s", exc)
        return 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check DB or data path")
    parser.add_argument("--db-url", help="Database URL")
    parser.add_argument("--path", help="NetCDF file or folder", default=None)
    args = parser.parse_args()
    main(args.db_url, args.path)
