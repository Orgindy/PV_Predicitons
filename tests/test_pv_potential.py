import numpy as np
import pytest

from pv_potential import calculate_pv_potential
from constants import PV_CONSTANTS, PHYSICAL_LIMITS


def test_temperature_coefficient_validation():
    """Invalid coefficient should produce warning but not error."""
    with pytest.warns(UserWarning):
        result = calculate_pv_potential(
            GHI=np.array(800.0),
            T_air=np.array(20.0),
            RC_potential=np.array(50.0),
            Red_band=np.array(40.0),
            Total_band=np.array(100.0),
            temp_coeff=-0.02,
        )
    assert result >= 0

