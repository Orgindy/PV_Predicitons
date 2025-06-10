import numpy as np
import pandas as pd
import pytest
import importlib.util
import ast
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Dynamically load modules with spaces in file names
spec_fp = importlib.util.spec_from_file_location('feature_prep', Path('Feature Preparation.py'))
feature_prep = importlib.util.module_from_spec(spec_fp)
spec_fp.loader.exec_module(feature_prep)

# Extract prepare_features_for_ml without executing heavy imports
source_pv = Path('PV prediction.py').read_text()
module_ast = ast.parse(source_pv)
func_code = None
for node in module_ast.body:
    if isinstance(node, ast.FunctionDef) and node.name == 'prepare_features_for_ml':
        func_code = ast.get_source_segment(source_pv, node)
        break
namespace = {'StandardScaler': StandardScaler}
exec(func_code, globals(), namespace)

calculate_pv_potential = feature_prep.calculate_pv_potential
prepare_features_for_ml = namespace['prepare_features_for_ml']

from synergy_index import calculate_synergy_index

# Load prepare_features_for_clustering without heavy imports
source_clust = Path('clustering.py').read_text()
module_ast_clust = ast.parse(source_clust)
func_code_clust = None
for node in module_ast_clust.body:
    if isinstance(node, ast.FunctionDef) and node.name == 'prepare_features_for_clustering':
        func_code_clust = ast.get_source_segment(source_clust, node)
        break
namespace_clust = {'StandardScaler': StandardScaler, 'np': __import__('numpy'), 'pd': __import__('pandas')}
exec(func_code_clust, globals(), namespace_clust)
prepare_features_for_clustering = namespace_clust['prepare_features_for_clustering']

# Load RC power functions without importing heavy dependencies
source_rc = Path('RC_Clustering.py').read_text()
module_ast_rc = ast.parse(source_rc)
funcs_rc = {}
for node in module_ast_rc.body:
    if isinstance(node, ast.FunctionDef) and node.name in {'calculate_sky_temperature_improved', 'calculate_rc_power_improved'}:
        code = ast.get_source_segment(source_rc, node)
        exec(code, globals(), funcs_rc)

calculate_sky_temperature_improved = funcs_rc['calculate_sky_temperature_improved']
calculate_rc_power_improved = funcs_rc['calculate_rc_power_improved']


def test_calculate_pv_potential_basic():
    ghi = np.array([1000.0])
    t_air = np.array([25.0])
    rc = np.array([50.0])
    red = np.array([300.0])
    total = np.array([1000.0])
    pv = calculate_pv_potential(ghi, t_air, rc, red, total)
    assert np.isclose(pv, 700, atol=1)
    assert 0 <= pv <= 0.9 * ghi


def test_calculate_pv_potential_zero_ghi():
    result = calculate_pv_potential(np.array([0.0]), np.array([20.0]), np.array([0.0]), np.array([0.0]), np.array([1.0]))
    assert result == 0


def test_calculate_rc_power_improved_defaults():
    df = pd.DataFrame({'T_air': [25], 'GHI': [800]})
    out = calculate_rc_power_improved(df)
    sigma = 5.67e-8
    t_air_k = 25 + 273.15
    t_sky = calculate_sky_temperature_improved(25, RH=50, cloud_cover=0)
    t_sky_k = t_sky + 273.15
    q_rad = 0.95 * sigma * (t_air_k**4 - t_sky_k**4)
    q_solar = (1 - 0.3) * 800
    expected = q_rad - q_solar
    assert np.isclose(out['P_rc_net'].iloc[0], expected, atol=1e-6)


def test_calculate_rc_power_improved_positive_rc():
    df = pd.DataFrame({'T_air': [20], 'GHI': [0], 'RH': [0], 'TCC': [0]})
    out = calculate_rc_power_improved(df)
    assert out['P_rc_net'].iloc[0] > 0


def test_calculate_synergy_index_basic():
    t_pv = [50, 50]
    t_rc = [40, 40]
    ghi = [1000, 1000]
    rc_energy = [50, 50]
    index = calculate_synergy_index(t_pv, t_rc, ghi, rc_cooling_energy=rc_energy)
    assert index > 0


def test_calculate_synergy_index_mismatched_lengths():
    with pytest.raises(ValueError):
        calculate_synergy_index([1, 2], [1], [1, 2])


def test_calculate_synergy_index_zero_normalization():
    index = calculate_synergy_index([25], [25], [0])
    assert index == 0


def test_prepare_features_for_ml_shape():
    data = {
        'GHI': [1000, 800],
        'T_air': [20, 25],
        'RC_potential': [50, 60],
        'Wind_Speed': [5, 4],
        'Dew_Point': [10, 12],
        'Cloud_Cover': [0.1, 0.2],
        'Red_band': [300, 250],
        'Blue_band': [200, 180],
        'IR_band': [150, 160],
        'Total_band': [1000, 900],
        'PV_Potential': [700, 600]
    }
    df = pd.DataFrame(data)
    X_scaled, y, names, scaler = prepare_features_for_ml(df)
    assert X_scaled.shape == (2, len(names))
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-7)


def test_prepare_features_for_clustering():
    df = pd.DataFrame({
        'GHI': [1000, 800],
        'T_air': [20, 25],
        'RC_potential': [50, 60],
        'Wind_Speed': [5, 4],
        'Dew_Point': [10, 12],
        'Cloud_Cover': [0.1, 0.2],
        'Red_band': [300, 250],
        'Blue_band': [200, 180],
        'IR_band': [150, 160],
        'Total_band': [1000, 900],
        'PV_Potential_physics': [700, 600],
        'Predicted_PV_Potential': [710, 610]
    })
    X_scaled, features, scaler = prepare_features_for_clustering(df)
    assert X_scaled.shape[0] == len(df)
    assert len(features) >= 3


def test_prepare_features_for_clustering_insufficient():
    df = pd.DataFrame({'GHI': [1], 'T_air': [2]})
    X_scaled, features, scaler = prepare_features_for_clustering(df)
    assert X_scaled is None
    assert features == []
    assert scaler is None
