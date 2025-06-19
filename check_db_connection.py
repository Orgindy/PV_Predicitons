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
        logging.error("Path does not exist: %s", path)
        return 1

    url = db_url or DEFAULT_DB_URL
    try:
        engine = get_engine(url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logging.info("Successfully connected to the database.")
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
    raise SystemExit(main(args.db_url, args.path))
