from smarts_processor import extract_metadata, parse_ext_file, define_spectral_bands


def test_extract_metadata_basic():
    sample = "Reference for this run: test_20220101_1200\n"
    meta = extract_metadata(sample)
    assert meta["date"] == "2022-01-01"
    assert meta["time"] == "12:00"


def test_parse_ext_file_basic():
    text = "Wvlgth Col\n300 1\n310 2\n"
    df = parse_ext_file(text)
    assert "wavelength_um" in df.columns
    assert df.shape[0] == 2


def test_define_spectral_bands_empty():
    import pandas as pd
    df = pd.DataFrame({"wavelength_um": [], "Global_tilted_irradiance": []})
    bands = define_spectral_bands(df)
    assert isinstance(bands, dict)
    assert set(bands.keys()) == {"UV", "Blue", "Green", "Red", "IR"}
