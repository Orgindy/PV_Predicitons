import numpy as np


def calculate_pv_potential(GHI, T_air, RC_potential, Red_band, Total_band):
    """Compute PV potential using simplified physics model."""
    NOCT = 45  # °C
    Reference_Red_Fraction = 0.42
    PR_ref = 0.80

    T_cell = T_air + (NOCT - 20) / 800 * GHI
    Temp_Loss = -0.0045 * (T_cell - 25)
    RC_Gain = 0.01 * (RC_potential / 50)

    with np.errstate(divide='ignore', invalid='ignore'):
        Actual_Red_Fraction = np.divide(
            Red_band, Total_band, out=np.zeros_like(Red_band), where=Total_band != 0
        )

    Spectral_Adjust = Actual_Red_Fraction - Reference_Red_Fraction
    PR_corrected = np.clip(PR_ref + Temp_Loss + RC_Gain + Spectral_Adjust, 0.7, 0.9)

    PV_Potential = GHI * PR_corrected
    return PV_Potential
