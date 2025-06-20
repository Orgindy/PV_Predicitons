import numpy as np
import logging
import warnings
from typing import Union, Tuple

from constants import PV_CONSTANTS, PHYSICAL_LIMITS


logger = logging.getLogger(__name__)


def validate_temperature_coefficient(temp_coeff: float) -> float:
    """Validate temperature coefficient, clamping to safe bounds."""
    try:
        coeff = float(temp_coeff)
    except Exception:
        warnings.warn(
            "Temperature coefficient must be numeric, using default -0.004",
            stacklevel=2,
        )
        logger.warning("Invalid temperature coefficient '%s', using default", temp_coeff)
        return -0.004

    if coeff < -0.01:
        warnings.warn("Temperature coefficient below -0.01, clamping", stacklevel=2)
        logger.warning("Temperature coefficient %s below -0.01, clamping", coeff)
        coeff = -0.01
    elif coeff > 0:
        warnings.warn("Temperature coefficient above 0, clamping", stacklevel=2)
        logger.warning("Temperature coefficient %s above 0, clamping", coeff)
        coeff = 0.0

    return coeff


def validate_pv_inputs(
    GHI: np.ndarray,
    T_air: np.ndarray,
    RC_potential: np.ndarray,
    Red_band: np.ndarray,
    Total_band: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate inputs for PV potential calculation."""
    inputs = {
        'GHI': (GHI, *PHYSICAL_LIMITS['GHI']),
        'T_air': (T_air, *PHYSICAL_LIMITS['T_air']),
        'RC_potential': (RC_potential, *PHYSICAL_LIMITS['RC_potential']),
        'Red_band': (Red_band, *PHYSICAL_LIMITS['Red_band']),
        'Total_band': (Total_band, *PHYSICAL_LIMITS['Total_band']),
    }

    validated = {}
    for name, (value, min_val, max_val) in inputs.items():
        arr = np.asarray(value, dtype=float)
        if min_val is not None and np.any(arr < min_val):
            warnings.warn(
                f"{name} values below {min_val} clamped", stacklevel=2
            )
            logger.warning("%s values below %s clamped", name, min_val)
            arr = np.clip(arr, min_val, None)
        if max_val is not None and np.any(arr > max_val):
            warnings.warn(
                f"{name} values above {max_val} clamped", stacklevel=2
            )
            logger.warning("%s values above %s clamped", name, max_val)
            arr = np.clip(arr, None, max_val)
        validated[name] = arr

    return (
        validated['GHI'],
        validated['T_air'],
        validated['RC_potential'],
        validated['Red_band'],
        validated['Total_band'],
    )


def calculate_pv_potential(
    GHI: Union[float, np.ndarray],
    T_air: Union[float, np.ndarray],
    RC_potential: Union[float, np.ndarray],
    Red_band: Union[float, np.ndarray],
    Total_band: Union[float, np.ndarray],
    temp_coeff: float = PV_CONSTANTS['temperature_coefficients']['Silicon'],
) -> np.ndarray:
    """Calculate PV potential with validated inputs and proper error handling."""
    # Validate temperature coefficient
    temp_coeff = validate_temperature_coefficient(temp_coeff)

    # Validate other inputs
    GHI, T_air, RC_potential, Red_band, Total_band = validate_pv_inputs(
        GHI, T_air, RC_potential, Red_band, Total_band
    )

    # Use constants from configuration
    T_cell = T_air + (PV_CONSTANTS['NOCT'] - 20) / 800 * GHI
    Temp_Loss = temp_coeff * (T_cell - 25)
    RC_Gain = 0.01 * (RC_potential / 50)

    with np.errstate(divide='ignore', invalid='ignore'):
        Actual_Red_Fraction = np.divide(
            Red_band,
            Total_band,
            out=np.full_like(Red_band, PV_CONSTANTS['Reference_Red_Fraction']),
            where=(Total_band > 0) & ~np.isnan(Total_band),
        )

    Spectral_Adjust = Actual_Red_Fraction - PV_CONSTANTS['Reference_Red_Fraction']
    min_pr, max_pr = PV_CONSTANTS['PR_bounds']
    PR_corrected = np.clip(
        PV_CONSTANTS['PR_ref'] + Temp_Loss + RC_Gain + Spectral_Adjust,
        min_pr,
        max_pr,
    )

    PV_Potential = GHI * PR_corrected
    PV_Potential = np.nan_to_num(PV_Potential, nan=0.0)

    return PV_Potential
