# -*- coding: utf-8 -*-
"""
Created on Mon May 26 11:29:13 2025

@author: Gindi002
"""
import numpy as np
from scipy.optimize import root_scalar

# Stefan-Boltzmann constant
SIGMA = 5.670374419e-8  # W/m²·K⁴

material_config = {
    "alpha_solar": 0.90,  # Absorptivity in solar range (0–1)
    "epsilon_IR": 0.92,  # Emissivity in thermal IR range (8–13 μm)
    "thickness_m": 0.003,  # Effective thermal mass thickness [m]
    "density": 2500,  # Density [kg/m³]
    "cp": 900,  # Specific heat [J/kg·K]
    "h_conv_base": 5,  # Base convective coefficient [W/m²·K] at 0 m/s
    "h_conv_wind_coeff": 4,  # Convective gain per 1 m/s wind
    "use_dynamic_emissivity": False,
    "emissivity_curve": None,  # Optional function: T_surface → emissivity
}


def compute_temperature_series(
    ghi_array,
    tair_array,
    ir_down_array,
    wind_array,
    material_config,
    zenith_array=None,
    switching_profile=None,
    emissivity_profile=None,
    alpha_profile=None,
):
    """
    Compute surface temperature time series from hourly input data.
    Supports both static and dynamic material properties.

    Parameters:
        ghi_array (np.ndarray): Global horizontal irradiance [W/m²]
        tair_array (np.ndarray): Air temperature [K]
        ir_down_array (np.ndarray): Downwelling IR radiation [W/m²]
        wind_array (np.ndarray): Wind speed [m/s]
        material_config (dict): Surface material configuration
        zenith_array (np.ndarray, optional): Solar zenith angles [degrees]
        switching_profile (dict, optional): Dynamic material switching logic
        emissivity_profile (dict, optional): Emissivity values per state
        alpha_profile (dict, optional): Absorptivity values per state

    Returns:
        np.ndarray: Array of estimated surface temperatures [K]
    """
    n = len(ghi_array)
    T_surface_series = np.zeros(n)

    # Check if dynamic materials are requested
    use_dynamic = all(
        x is not None
        for x in [zenith_array, switching_profile, emissivity_profile, alpha_profile]
    )

    for i in range(n):
        GHI = ghi_array[i]
        T_air = tair_array[i]
        IR_down = ir_down_array[i]
        wind_speed = wind_array[i]

        # Handle dynamic materials if enabled
        if use_dynamic:
            try:
                from dynamic_materials import get_material_properties

                solar_zenith = zenith_array[i]

                # Get dynamic material properties
                props = get_material_properties(
                    T_surface=T_air,  # Initial guess
                    GHI=GHI,
                    solar_zenith=solar_zenith,
                    profile=switching_profile,
                    emissivity_profile=emissivity_profile,
                    alpha_profile=alpha_profile,
                )

                # Update material config with dynamic values
                dynamic_config = material_config.copy()
                dynamic_config["alpha_solar"] = props["alpha"]
                dynamic_config["epsilon_IR"] = props["emissivity"]

                config_to_use = dynamic_config
            except ImportError:
                print("⚠️ Dynamic materials not available, using static properties")
                config_to_use = material_config
        else:
            config_to_use = material_config

        try:
            T_surf = solve_surface_temperature(
                GHI, T_air, IR_down, wind_speed, config_to_use
            )
        except RuntimeError as e:
            print(f"[{i}] Solver failed: {e}")
            T_surf = np.nan  # mark as missing

        T_surface_series[i] = T_surf

    return T_surface_series


def compute_convective_loss(T_surface, T_air, wind_speed, config):
    """
    Compute convective heat loss based on surface and air temperature and wind speed.

    Parameters:
        T_surface (float): Surface temperature [K]
        T_air (float): Ambient air temperature [K]
        wind_speed (float): Wind speed at surface [m/s]
        config (dict): Material configuration

    Returns:
        Q_conv (float): Convective heat loss [W/m²]
    """
    h = config["h_conv_base"] + config["h_conv_wind_coeff"] * wind_speed
    Q_conv = h * (T_surface - T_air)
    return Q_conv


def compute_net_longwave_flux(T_surface, T_sky, config):
    """
    Compute net thermal IR radiation from surface to sky.

    Parameters:
        T_surface (float): Surface temperature [K]
        T_sky (float): Sky equivalent blackbody temperature [K]
        config (dict): Material configuration

    Returns:
        Q_net_rad (float): Net outgoing radiation [W/m²]
    """
    epsilon = config["epsilon_IR"]

    # Optional: support for dynamic emissivity
    if config.get("use_dynamic_emissivity") and config.get("emissivity_curve"):
        epsilon = config["emissivity_curve"](T_surface)

    Q_emit = epsilon * SIGMA * T_surface**4
    Q_absorb = epsilon * SIGMA * T_sky**4
    Q_net_rad = Q_emit - Q_absorb
    return Q_net_rad


def solve_surface_temperature(
    GHI, T_air, IR_down, wind_speed, material_config, T_guess=300.0, max_iter=100
):
    """
    Solve for surface temperature using energy balance:
    Q_solar + Q_IR_in - Q_IR_out - Q_conv = 0

    Parameters:
        GHI (float): Global horizontal irradiance [W/m²]
        T_air (float): Air temperature [K]
        IR_down (float): Downwelling thermal radiation [W/m²]
        wind_speed (float): Wind speed at surface [m/s]
        material_config (dict): Material and thermal config
        T_guess (float): Initial temperature guess [K]
        max_iter (int): Max iterations for solver

    Returns:
        T_surface (float): Estimated surface temperature [K]
    """

    alpha = material_config["alpha_solar"]
    epsilon = material_config["epsilon_IR"]

    # Estimate sky temperature from downwelling IR
    T_sky = (IR_down / (epsilon * SIGMA)) ** 0.25

    def energy_balance(T_surface):
        Q_solar = alpha * GHI
        Q_conv = compute_convective_loss(T_surface, T_air, wind_speed, material_config)
        Q_rad = compute_net_longwave_flux(T_surface, T_sky, material_config)
        return Q_solar - Q_conv - Q_rad

    result = root_scalar(
        energy_balance,
        bracket=[T_air - 30, T_air + 60],
        method="brentq",
        maxiter=max_iter,
    )

    if result.converged:
        return result.root
    else:
        raise RuntimeError("Failed to converge surface temperature solver.")


def estimate_pv_cell_temperature(GHI, T_air, wind_speed, model="NOCT", noct=45):
    """
    Estimate PV cell temperature using simple empirical models.

    Parameters:
        GHI (float or np.ndarray): Global irradiance [W/m²]
        T_air (float or np.ndarray): Ambient air temperature [°C]
        wind_speed (float or np.ndarray): Wind speed at surface [m/s]
        model (str): 'NOCT' or 'Sandia'
        noct (float): Nominal Operating Cell Temperature [°C]

    Returns:
        np.ndarray: Estimated PV cell temperature [°C]
    """

    G_ref = 800  # Reference irradiance for NOCT
    T_ref = 20  # Reference ambient temperature

    if model == "NOCT":
        return T_air + ((noct - T_ref) / G_ref) * GHI

    elif model == "Sandia":
        # Coefficients can be tuned per PV module datasheet
        a = -3.56  # wind coefficient
        b = 0.943  # irradiance coefficient
        return T_air + (GHI / 1000) * np.exp(a + b * wind_speed)

    else:
        raise ValueError("Unsupported model: choose 'NOCT' or 'Sandia'")
