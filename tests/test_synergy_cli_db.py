import subprocess
import sys
from pathlib import Path
import pandas as pd
from database_utils import read_table, write_dataframe


def test_synergy_cli_db(tmp_path):
    db_url = f"sqlite:///{tmp_path}/test.db"
    df = pd.DataFrame({"T_PV": [40.0], "T_RC": [30.0], "GHI": [800.0]})
    write_dataframe(df, "pv", db_url=db_url)

    script = Path(__file__).resolve().parents[1] / "synergy_index.py"
    subprocess.check_call([
        sys.executable,
        str(script),
        "--db-url", db_url,
        "--db-table", "pv",
        "--output", str(tmp_path / "out.csv"),
    ])

    result = read_table("pv", db_url=db_url)
    assert "Synergy_Index" in result.columns
