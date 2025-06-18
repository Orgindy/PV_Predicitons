"""Utility functions related to sky temperature calculations."""

from typing import Union
import numpy as np


def calculate_sky_temperature_improved(T_air: Union[float, np.ndarray], RH: Union[float, np.ndarray] = 50,
                                        cloud_cover: Union[float, np.ndarray] = 0) -> np.ndarray:
    """Calculate sky temperature using basic atmospheric physics.

    Parameters
    ----------
    T_air : float or array-like
        Air temperature in degrees Celsius.
    RH : float or array-like, optional
        Relative humidity in percent. Defaults to ``50``.
    cloud_cover : float or array-like, optional
        Cloud fraction between 0 and 1. Defaults to ``0``.

    Returns
    -------
    numpy.ndarray
        Calculated sky temperature in degrees Celsius.
    """
    T_air_K = np.array(T_air) + 273.15

    # Swinbank's formula for clear sky emissivity
    eps_clear = 0.741 + 0.0062 * np.array(RH)

    # Cloud correction (Duffie & Beckman)
    eps_sky = eps_clear + (1 - eps_clear) * np.array(cloud_cover)

    # Clip emissivity to a physical range
    eps_sky = np.clip(eps_sky, 0.7, 1.0)

    # Sky temperature from Stefan-Boltzmann law
    T_sky_K = T_air_K * np.power(eps_sky, 0.25)

    return T_sky_K - 273.15
