import warnings
import xarray as xr
import numpy as np
import sys
import types

def import_module_with_stubs():
    modules_to_stub = {
        'pykrige': types.ModuleType('pykrige'),
        'pykrige.ok': types.ModuleType('pykrige.ok'),
        'matplotlib': types.ModuleType('matplotlib'),
        'matplotlib.pyplot': types.ModuleType('matplotlib.pyplot'),
        'cartopy': types.ModuleType('cartopy'),
        'cartopy.crs': types.ModuleType('cartopy.crs'),
        'cartopy.feature': types.ModuleType('cartopy.feature'),
        'xarray': xr,
        'geopandas': types.ModuleType('geopandas'),
        'sklearn_extra': types.ModuleType('sklearn_extra'),
        'sklearn_extra.cluster': types.ModuleType('sklearn_extra.cluster'),
    }
    modules_to_stub['pykrige.ok'].OrdinaryKriging = object
    modules_to_stub['sklearn_extra.cluster'].KMedoids = object
    for name, module in modules_to_stub.items():
        sys.modules.setdefault(name, module)
    import importlib
    return importlib.import_module('rc_cooling_combined_2025')

def test_load_and_merge_rc_netcdf_years_no_resource_warning(tmp_path):
    times = np.array(['2000-01-01', '2000-01-02'], dtype='datetime64[ns]')
    for i in range(2):
        ds = xr.Dataset({'QNET': ('time', np.arange(len(times)) + i)}, coords={'time': times})
        ds.to_netcdf(tmp_path / f'data_{i}.nc')

    module = import_module_with_stubs()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        merged = module.load_and_merge_rc_netcdf_years(str(tmp_path), var_name='QNET')

    assert merged.dims['time'] == 4
    assert not any(issubclass(wi.category, ResourceWarning) for wi in w)
