import os
import logging
import pandas as pd
from sqlalchemy import create_engine

DEFAULT_DB_URL = os.getenv("PV_DB_URL", "sqlite:///pv_data.sqlite")

from utils.errors import SynergyDatabaseError

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
    if url.startswith("sqlite:///"):
        path = url.replace("sqlite:///", "")
        if not os.path.exists(path):
            logging.warning("SQLite database file not found: %s", path)
            raise SynergyDatabaseError("SQLite file does not exist", {"path": path})
    try:
        return create_engine(url)
    except Exception as exc:
        logging.warning("Failed to create engine: %s", exc)
        raise SynergyDatabaseError("Engine creation failed", {"url": url}) from exc

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
    return pd.read_sql_table(table_name, engine)

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
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
