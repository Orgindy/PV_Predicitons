import xarray as xr

from grib_to_cdf_new import validate_metadata


def test_validate_metadata_pass(tmp_path):
    ds = xr.Dataset(
        {
            "t2m": (("time", "latitude", "longitude"), [[[280.0]]]),
            "sp": (("time", "latitude", "longitude"), [[[1013.0]]]),
            "u10": (("time", "latitude", "longitude"), [[[5.0]]]),
            "v10": (("time", "latitude", "longitude"), [[[0.0]]]),
        },
        coords={
            "time": [0],
            "latitude": [10.0],
            "longitude": [20.0],
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": "Unit Test",
            "institution": "Test Suite",
            "source": "Synthetic",
        },
    )
    path = tmp_path / "tmp.nc"
    ds.to_netcdf(path)

    assert validate_metadata(str(path)) is True
