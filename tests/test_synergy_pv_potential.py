import numpy as np
import pytest

from synergy_index import calculate_synergy_index
from pv_potential import calculate_pv_potential


def test_calculate_pv_potential_physical_limits():
    with pytest.raises(ValueError, match="GHI contains values above 1500"):
        calculate_pv_potential(
            GHI=np.array(2000.0),
            T_air=np.array(20.0),
            RC_potential=np.array(50.0),
            Red_band=np.array(40.0),
            Total_band=np.array(100.0),
        )

    with pytest.raises(ValueError, match="T_air contains values above 60"):
        calculate_pv_potential(
            GHI=np.array(800.0),
            T_air=np.array(70.0),
            RC_potential=np.array(50.0),
            Red_band=np.array(40.0),
            Total_band=np.array(100.0),
        )


def test_calculate_pv_potential_division_by_zero():
    result = calculate_pv_potential(
        GHI=np.array(800.0),
        T_air=np.array(20.0),
        RC_potential=np.array(50.0),
        Red_band=np.array(40.0),
        Total_band=np.array(0.0),
    )
    assert not np.isnan(result).any()
    assert result >= 0


def test_calculate_pv_potential_nan_handling():
    result = calculate_pv_potential(
        GHI=np.array(800.0),
        T_air=np.array(20.0),
        RC_potential=np.array(50.0),
        Red_band=np.array(np.nan),
        Total_band=np.array(100.0),
    )
    assert not np.isnan(result).any()
    assert result >= 0


def test_calculate_synergy_index_positive():
    idx = calculate_synergy_index(
        [30, 32],
        [25, 26],
        [800, 1000],
        gamma_pv=-0.004,
    )
    assert idx > 0


def test_calculate_synergy_index_length_error():
    with pytest.raises(ValueError):
        calculate_synergy_index([30], [25, 26], [800, 1000])
