import sys
import types
from pathlib import Path
import pandas as pd
import xarray as xr
import psutil

# Helper to import Feature Preparation with stubs


def import_feature_module():
    modules = {
        'matplotlib': types.ModuleType('matplotlib'),
        'matplotlib.pyplot': types.ModuleType('matplotlib.pyplot'),
        'seaborn': types.ModuleType('seaborn'),
        'sklearn': types.ModuleType('sklearn'),
        'sklearn.model_selection': types.ModuleType('sklearn.model_selection'),
        'sklearn.ensemble': types.ModuleType('sklearn.ensemble'),
        'sklearn.metrics': types.ModuleType('sklearn.metrics'),
        'sklearn.feature_selection': types.ModuleType('sklearn.feature_selection'),
    }
    modules['sklearn.model_selection'].train_test_split = lambda *a, **k: (None, None, None, None)
    modules['sklearn.ensemble'].RandomForestRegressor = object
    modules['sklearn.metrics'].mean_squared_error = lambda *a, **k: 0
    modules['sklearn.metrics'].mean_absolute_error = lambda *a, **k: 0
    modules['sklearn.metrics'].r2_score = lambda *a, **k: 0
    modules['sklearn.feature_selection'].mutual_info_regression = lambda *a, **k: 0
    for name, mod in modules.items():
        sys.modules.setdefault(name, mod)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'feature_preparation',
        str(Path(__file__).resolve().parents[1] / 'feature_preparation.py'),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Helper to import rc_cooling_combined_2025 with stubs


def import_rc_module():
    modules = {
        'pykrige': types.ModuleType('pykrige'),
        'pykrige.ok': types.ModuleType('pykrige.ok'),
        'matplotlib': types.ModuleType('matplotlib'),
        'matplotlib.pyplot': types.ModuleType('matplotlib.pyplot'),
        'cartopy': types.ModuleType('cartopy'),
        'cartopy.crs': types.ModuleType('cartopy.crs'),
        'cartopy.feature': types.ModuleType('cartopy.feature'),
        'geopandas': types.ModuleType('geopandas'),
        'sklearn_extra': types.ModuleType('sklearn_extra'),
        'sklearn_extra.cluster': types.ModuleType('sklearn_extra.cluster'),
        'sklearn': types.ModuleType('sklearn'),
        'sklearn.model_selection': types.ModuleType('sklearn.model_selection'),
        'sklearn.preprocessing': types.ModuleType('sklearn.preprocessing'),
    }
    modules['pykrige.ok'].OrdinaryKriging = object
    modules['sklearn_extra.cluster'].KMedoids = object
    modules['sklearn.model_selection'].train_test_split = lambda *a, **k: (None, None, None, None)
    modules['sklearn.preprocessing'].StandardScaler = object
    for name, mod in modules.items():
        sys.modules.setdefault(name, mod)

    import importlib
    return importlib.import_module('rc_cooling_combined_2025')


def test_load_netcdf_data_closes_file(tmp_path):
    module = import_feature_module()
    ds = xr.Dataset({'a': ('x', [1, 2, 3])})
    path = tmp_path / 'test.nc'
    ds.to_netcdf(path)

    module.load_netcdf_data(str(path), sample_fraction=1.0)
    open_files = [f.path for f in psutil.Process().open_files()]
    assert str(path) not in open_files


def test_add_effective_albedo_context(monkeypatch, tmp_path):
    module = import_rc_module()
    # Use lower-case "h" to avoid pandas deprecation warning
    times = pd.date_range('2020-01-01', periods=1, freq='h')
    df = pd.DataFrame({'time': times, 'LAT': [0.0], 'LON': [0.0]})
    grib_dir = tmp_path / 'grib'
    grib_dir.mkdir()
    grib_file = grib_dir / f'test{times[0].year}{times[0].month:02d}.grib'
    grib_file.write_text('')

    class DummyWrapper:
        def __init__(self):
            arr = xr.DataArray(
                [[[0.1]]],
                coords={'time': times, 'latitude': [0.0], 'longitude': [0.0]},
                dims=('time', 'latitude', 'longitude'),
            )
            self.ds = xr.Dataset({'fal': arr})
            self.closed = False

        def __enter__(self):
            return self.ds

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.closed = True

    wrapper = DummyWrapper()
    monkeypatch.setattr(module.xr, 'open_dataset', lambda *a, **k: wrapper)

    result = module.add_effective_albedo_optimized(df, str(grib_dir))
    assert wrapper.closed is True
    assert 'effective_albedo' in result.columns
