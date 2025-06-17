# utils/humidity.py

import numpy as np


def compute_relative_humidity(T_air_K, T_dew_K):
    """
    Compute relative humidity (%) using the Magnus formula.

    Parameters:
        T_air_K (float or np.ndarray or pd.Series): Air temperature in Kelvin
        T_dew_K (float or np.ndarray or pd.Series): Dew point temperature in Kelvin

    Returns:
        Relative humidity in percentage (0–100%)
    """
    T_air = np.array(T_air_K) - 273.15
    T_dew = np.array(T_dew_K) - 273.15

    a = 17.625
    b = 243.04

    e_s = np.exp((a * T_air) / (b + T_air))
    e_d = np.exp((a * T_dew) / (b + T_dew))
    RH = 100.0 * (e_d / e_s)

    return np.clip(RH, 0, 100)
