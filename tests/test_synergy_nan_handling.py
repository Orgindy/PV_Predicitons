import numpy as np
import pandas as pd
from synergy_index import calculate_synergy_index, add_synergy_index


def test_calculate_synergy_index_nan_inputs():
    idx = calculate_synergy_index(
        [30.0, np.nan, 32.0],
        [25.0, np.nan, 26.0],
        [800.0, np.nan, 1000.0],
    )
    assert not np.isnan(idx)


def test_add_synergy_index_zero_division():
    df = pd.DataFrame({"T_PV": [40.0, np.nan], "T_RC": [30.0, np.nan], "GHI": [1000.0, 0.0]})
    result = add_synergy_index(df)
    assert "Synergy_Index" in result.columns
    assert not result["Synergy_Index"].isna().any()
