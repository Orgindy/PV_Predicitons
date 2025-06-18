import os
import pandas as pd
from sqlalchemy import create_engine

DEFAULT_DB_URL = os.getenv("PV_DB_URL", "sqlite:///pv_data.sqlite")

def get_engine(db_url: str = None):
    """Return an SQLAlchemy engine for the given URL.

    Parameters
    ----------
    db_url : str, optional
        Database URL. If not provided, the ``PV_DB_URL`` environment variable or
        ``DEFAULT_DB_URL`` is used.

    Returns
    -------
    sqlalchemy.engine.Engine
        Engine instance connected to the specified database.
    """
    url = db_url or DEFAULT_DB_URL
    return create_engine(url)

def read_table(table_name: str, db_url: str = None):
    """Load an entire table into a :class:`pandas.DataFrame`.

    Parameters
    ----------
    table_name : str
        Name of the table to read.
    db_url : str, optional
        Database connection URL. Falls back to ``PV_DB_URL`` if omitted.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing all rows from ``table_name``.
    """
    engine = get_engine(db_url)
    with engine.connect() as conn:
        return pd.read_sql_table(table_name, conn)

def write_dataframe(df: pd.DataFrame, table_name: str, db_url: str = None, if_exists: str = "replace"):
    """Write a DataFrame to a database table.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to write.
    table_name : str
        Destination table name.
    db_url : str, optional
        Database URL. Uses ``PV_DB_URL`` if not provided.
    if_exists : str, optional
        How to behave if the table already exists. Passed directly to
        :func:`DataFrame.to_sql`. Default ``"replace"``.
    """
    engine = get_engine(db_url)
    with engine.begin() as conn:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
