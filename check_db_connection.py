import argparse
from database_utils import get_engine
from sqlalchemy.exc import SQLAlchemyError


def main(db_url=None):
    try:
        engine = get_engine(db_url)
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        print("Successfully connected to the database.")
    except SQLAlchemyError as e:
        print(f"Database connection failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check database connection")
    parser.add_argument("--db-url", default=None, help="Database URL")
    args = parser.parse_args()
    raise SystemExit(main(args.db_url))
