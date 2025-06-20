import logging
import os
from database_utils import get_engine, DEFAULT_DB_URL
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from utils.errors import SynergyDatabaseError


def _validate_data_path(path: str) -> None:
    """Validate that the path points to an existing .nc or .db file or directory."""
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

    if not path.endswith(('.nc', '.db')):
        logging.error("Unsupported file extension for %s", path)
        raise ValueError("Expected .nc or .db file")
    logging.info("Validated file %s", path)


def main(db_url: str | None = None, path: str | None = None) -> int:
    """Check that a database or NetCDF path is reachable."""
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
