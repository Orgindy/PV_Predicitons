import numpy as np
from typing import Union, Tuple


def validate_temperature_coefficient(temp_coeff: float) -> float:
    """Validate temperature coefficient is within physical bounds."""
    if not isinstance(temp_coeff, (int, float)):
        raise TypeError("Temperature coefficient must be a number")
    if not -0.01 <= temp_coeff <= 0:
        raise ValueError("Temperature coefficient must be between -0.01 and 0")
    return float(temp_coeff)


def validate_pv_inputs(
    GHI: np.ndarray,
    T_air: np.ndarray,
    RC_potential: np.ndarray,
    Red_band: np.ndarray,
    Total_band: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate inputs for PV potential calculation."""
    inputs = {
        'GHI': (GHI, 0, 1500),
        'T_air': (T_air, -50, 60),
        'RC_potential': (RC_potential, -100, 300),
        'Red_band': (Red_band, 0, None),
        'Total_band': (Total_band, 0, None),
    }

    validated = {}
    for name, (value, min_val, max_val) in inputs.items():
        arr = np.asarray(value, dtype=float)
        if min_val is not None and np.any(arr < min_val):
            raise ValueError(f"{name} contains values below {min_val}")
        if max_val is not None and np.any(arr > max_val):
            raise ValueError(f"{name} contains values above {max_val}")
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
) -> np.ndarray:
    """Calculate PV potential with validated inputs and proper error handling."""
    GHI, T_air, RC_potential, Red_band, Total_band = validate_pv_inputs(
        GHI, T_air, RC_potential, Red_band, Total_band
    )

    NOCT = 45  # Nominal Operating Cell Temperature [°C]
    Reference_Red_Fraction = 0.42  # From AM1.5 standard
    PR_ref = 0.80  # Reference performance ratio

    T_cell = T_air + (NOCT - 20) / 800 * GHI
    Temp_Loss = -0.0045 * (T_cell - 25)
    RC_Gain = 0.01 * (RC_potential / 50)

    with np.errstate(divide='ignore', invalid='ignore'):
        Actual_Red_Fraction = np.divide(
            Red_band,
            Total_band,
            out=np.full_like(Red_band, Reference_Red_Fraction),
            where=(Total_band > 0) & ~np.isnan(Total_band),
        )

    Spectral_Adjust = Actual_Red_Fraction - Reference_Red_Fraction
    PR_corrected = np.clip(PR_ref + Temp_Loss + RC_Gain + Spectral_Adjust, 0.7, 0.9)

    PV_Potential = GHI * PR_corrected
    PV_Potential = np.nan_to_num(PV_Potential, nan=0.0)

    return PV_Potential
