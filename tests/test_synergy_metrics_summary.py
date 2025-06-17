import pandas as pd

from synergy_index import calculate_synergy_metrics_summary


def test_summary_no_group():
    df = pd.DataFrame({"Synergy_Index": [1.0, 2.0, 3.0]})
    summary = calculate_synergy_metrics_summary(df)
    assert isinstance(summary, pd.DataFrame)
    assert summary["mean"].iloc[0] == 2.0


def test_summary_grouped():
    df = pd.DataFrame(
        {"Cluster_ID": [1, 1, 2, 2], "Synergy_Index": [1.0, 2.0, 3.0, 5.0]}
    )
    summary = calculate_synergy_metrics_summary(df, ["Cluster_ID"])
    assert summary.loc[1, "mean"] == 1.5
    assert summary.loc[2, "mean"] == 4.0
