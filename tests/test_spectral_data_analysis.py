import pandas as pd
from spectral_data_analysis import load_spectral_data

def test_load_spectral_data_missing(tmp_path):
    path = tmp_path / "missing.ext.txt"
    df = load_spectral_data(path)
    assert df is None


def test_load_spectral_data_basic(tmp_path):
    sample = "Wvlgth val\n300 1\n400 2\n"
    f = tmp_path / "sample.ext.txt"
    f.write_text(sample)
    df = load_spectral_data(f)
    assert df is not None
    assert list(df.columns) == ["Wvlgth", "val"]
    assert df.shape[0] == 2
