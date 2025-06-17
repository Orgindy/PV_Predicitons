import numpy as np
import pytest

import pv_potential
from synergy_index import calculate_synergy_index


def test_calculate_pv_potential_basic():
    result = pv_potential.calculate_pv_potential(
        GHI=np.array(800.0),
        T_air=np.array(20.0),
        RC_potential=np.array(50.0),
        Red_band=np.array(40.0),
        Total_band=np.array(100.0),
    )
    assert np.isclose(result, 560.0)


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
