import logging
from database_utils import get_engine, DEFAULT_DB_URL
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from utils.errors import SynergyDatabaseError


def main(db_url: str | None = None) -> int:
    """Check that a database is reachable."""
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
    raise SystemExit(main())
