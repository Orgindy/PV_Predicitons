import sys
import types
import pandas as pd


def import_module_with_stubs():
    """Import rc_cooling_combined_2025 with heavy dependencies stubbed."""
    modules_to_stub = {
        'pykrige': types.ModuleType('pykrige'),
        'pykrige.ok': types.ModuleType('pykrige.ok'),
        'matplotlib': types.ModuleType('matplotlib'),
        'matplotlib.pyplot': types.ModuleType('matplotlib.pyplot'),
        'cartopy': types.ModuleType('cartopy'),
        'cartopy.crs': types.ModuleType('cartopy.crs'),
        'cartopy.feature': types.ModuleType('cartopy.feature'),
        'xarray': types.ModuleType('xarray'),
        'geopandas': types.ModuleType('geopandas'),
        'sklearn_extra': types.ModuleType('sklearn_extra'),
        'sklearn_extra.cluster': types.ModuleType('sklearn_extra.cluster'),
        'sklearn': types.ModuleType('sklearn'),
        'sklearn.model_selection': types.ModuleType('sklearn.model_selection'),
        'sklearn.preprocessing': types.ModuleType('sklearn.preprocessing'),
    }
    modules_to_stub['pykrige.ok'].OrdinaryKriging = object
    modules_to_stub['sklearn_extra.cluster'].KMedoids = object
    modules_to_stub['sklearn.model_selection'].train_test_split = lambda *a, **k: ([], [])
    modules_to_stub['sklearn.preprocessing'].StandardScaler = object

    for name, module in modules_to_stub.items():
        sys.modules.setdefault(name, module)

    import importlib
    return importlib.import_module('rc_cooling_combined_2025')


def test_aggregate_rc_metrics_no_error(tmp_path):
    df = pd.DataFrame({"Cluster_ID": [1, 1, 2, 2], "RC_Kriged": [10, 20, 30, 40]})
    kriged_file = tmp_path / "kriged.csv"
    df.to_csv(kriged_file, index=False)

    output_file = tmp_path / "metrics.csv"
    module = import_module_with_stubs()
    module.aggregate_rc_metrics(str(kriged_file), str(output_file))

    result = pd.read_csv(output_file)
    assert set(result.columns) >= {"Cluster_ID", "RC_mean"}
    assert len(result) == 2


def test_aggregate_rc_metrics_columns(tmp_path):
    df = pd.DataFrame({"Cluster_ID": [1, 1, 2, 2], "RC_Kriged": [5, 15, 25, 35]})
    kriged_file = tmp_path / "kriged.csv"
    df.to_csv(kriged_file, index=False)

    output_file = tmp_path / "metrics.csv"
    module = import_module_with_stubs()
    module.aggregate_rc_metrics(str(kriged_file), str(output_file))

    result = pd.read_csv(output_file)
    expected = [
        "Cluster_ID",
        "RC_mean",
        "RC_median",
        "RC_std",
        "RC_min",
        "RC_max",
        "RC_sum",
        "RC_count",
    ]
    assert list(result.columns) == expected
