import os
import pandas as pd
from sqlalchemy import create_engine

DEFAULT_DB_URL = os.getenv("PV_DB_URL", "sqlite:///pv_data.sqlite")

def get_engine(db_url: str = None):
    """Return SQLAlchemy engine using db_url or env variable."""
    url = db_url or DEFAULT_DB_URL
    return create_engine(url)

def read_table(table_name: str, db_url: str = None):
    """Read entire table into a DataFrame."""
    engine = get_engine(db_url)
    return pd.read_sql_table(table_name, engine)

def write_dataframe(df: pd.DataFrame, table_name: str, db_url: str = None, if_exists: str = "replace"):
    """Write DataFrame to table."""
    engine = get_engine(db_url)
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
