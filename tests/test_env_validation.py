import argparse
import types
import importlib


def import_main_with_stubs():
    heavy = {
        'geopandas': types.ModuleType('geopandas'),
        'cartopy': types.ModuleType('cartopy'),
        'cartopy.crs': types.ModuleType('cartopy.crs'),
        'sklearn': types.ModuleType('sklearn'),
        'sklearn.preprocessing': types.ModuleType('sklearn.preprocessing'),
        'matplotlib': types.ModuleType('matplotlib'),
        'matplotlib.pyplot': types.ModuleType('matplotlib.pyplot'),
        'matplotlib.ticker': types.ModuleType('matplotlib.ticker'),
        'shapely.geometry': types.ModuleType('shapely.geometry'),
        'rasterio': types.ModuleType('rasterio'),
        'rasterio.plot': types.ModuleType('rasterio.plot'),
        'pyproj': types.ModuleType('pyproj'),
        'contextily': types.ModuleType('contextily'),
        'sklearn_extra': types.ModuleType('sklearn_extra'),
        'sklearn_extra.cluster': types.ModuleType('sklearn_extra.cluster'),
        'sklearn.model_selection': types.ModuleType('sklearn.model_selection'),
        'sklearn.metrics': types.ModuleType('sklearn.metrics'),
        'sklearn.cluster': types.ModuleType('sklearn.cluster'),
        'sklearn.ensemble': types.ModuleType('sklearn.ensemble'),
        'sklearn.tree': types.ModuleType('sklearn.tree'),
        'xgboost': types.ModuleType('xgboost'),
    }
    import sys
    backup = {}
    for name, module in heavy.items():
        backup[name] = sys.modules.get(name)
        sys.modules[name] = module

    # minimal attributes for stubs
    heavy['sklearn_extra.cluster'].KMedoids = object
    heavy['sklearn.preprocessing'].StandardScaler = object
    heavy['sklearn.metrics'].silhouette_score = lambda *a, **k: 0
    heavy['sklearn.cluster'].KMeans = object
    heavy['shapely.geometry'].Point = object
    heavy['rasterio.plot'].show = lambda *a, **k: None
    heavy['pyproj'].Transformer = object
    heavy['sklearn.model_selection'].train_test_split = lambda *a, **k: ([], [])
    heavy['sklearn.ensemble'].RandomForestRegressor = object
    heavy['sklearn.ensemble'].GradientBoostingRegressor = object
    heavy['sklearn.tree'].DecisionTreeRegressor = object
    heavy['sklearn.metrics'].r2_score = lambda y, ypred: 0
    heavy['sklearn.metrics'].mean_squared_error = lambda y, ypred: 0
    heavy['xgboost'].XGBRegressor = object

    module = importlib.import_module('main')

    for name, mod in backup.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod

    return module

def test_invalid_db_url():
    module = import_main_with_stubs()
    args = argparse.Namespace(db_url="invalid", db_table="tbl")
    assert not module.validate_environment(args)

def test_missing_db_table():
    module = import_main_with_stubs()
    args = argparse.Namespace(db_url="sqlite:///db.sqlite", db_table="")
    assert not module.validate_environment(args)

def test_valid_db_config():
    module = import_main_with_stubs()
    args = argparse.Namespace(db_url="sqlite:///db.sqlite", db_table="tbl")
    assert module.validate_environment(args)
