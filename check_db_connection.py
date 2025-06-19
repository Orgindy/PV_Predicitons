import logging
import os
from database_utils import get_engine, DEFAULT_DB_URL
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from utils.errors import SynergyDatabaseError


def main(db_url: str | None = None, path: str | None = None) -> int:
    """Check that a database or NetCDF path is reachable."""
    if path:
        if os.path.isdir(path) or os.path.isfile(path):
            logging.info("Found path: %s", path)
            return 0
        logging.warning("Path does not exist: %s - creating", path)
        try:
            os.makedirs(path, exist_ok=True)
            logging.info("Created path: %s", path)
            return 0
        except Exception as exc:
            logging.error("Could not create path %s: %s", path, exc)
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
