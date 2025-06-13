import pandas as pd
from synergy_index import calculate_synergy_metrics_summary


def make_df():
    return pd.DataFrame({
        "Synergy_Index": [1.0, 2.0, 3.0],
        "group": ["A", "B", "A"],
    })


def test_metrics_summary_no_group():
    df = make_df()
    result = calculate_synergy_metrics_summary(df)
    assert isinstance(result, pd.DataFrame)


def test_metrics_summary_grouped():
    df = make_df()
    result = calculate_synergy_metrics_summary(df, group_by_cols=["group"])
    assert isinstance(result, pd.DataFrame)
