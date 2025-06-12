# -*- coding: utf-8 -*-
"""
Created on Mon May 26 11:33:12 2025

@author: Gindi002
"""
import pandas as pd
try:
    from pvlib.solarposition import get_solarposition
    PVLIB_AVAILABLE = True
except ImportError:
    print("⚠️ pvlib not available - solar position functions will be limited")
    PVLIB_AVAILABLE = False

def get_material_state(T_surface, GHI, profile):
    """
    Determine the dynamic material state based on environmental conditions.
    Fixed logic: Hot+Sunny → Bright (reflective), Cool+Dark → Dark (absorptive)

    Parameters:
        T_surface (float): Surface temperature [K]
        GHI (float): Global horizontal irradiance [W/m²]
        profile (dict): Switching logic dictionary with thresholds

    Returns:
        str: State name (e.g., 'bright', 'dark', 'static')
    """
    # Convert temperature to Celsius for easier threshold setting
    T_celsius = T_surface - 273.15 if T_surface > 200 else T_surface
    
    # Check each state condition in priority order
    for state, conditions in profile["state_map"].items():
        meets_conditions = True
        
        # Temperature conditions
        if "T_max" in conditions:
            meets_conditions &= T_celsius <= conditions["T_max"]
        if "T_min" in conditions:
            meets_conditions &= T_celsius >= conditions["T_min"]
        
        # Irradiance conditions  
        if "GHI_max" in conditions:
            meets_conditions &= GHI <= conditions["GHI_max"]
        if "GHI_min" in conditions:
            meets_conditions &= GHI >= conditions["GHI_min"]
        
        if meets_conditions:
            return state
    
    return profile.get("default", "static")

def get_emissivity(state, emissivity_profile):
    """
    Return emissivity based on the material state.

    Parameters:
        state (str): Material state name (e.g. 'bright', 'dark', etc.)
        emissivity_profile (dict): Mapping of state → emissivity value

    Returns:
        float: Emissivity value (0–1)
    """
    return emissivity_profile.get(state, emissivity_profile.get("default", 0.90))

def get_alpha_solar(state, alpha_profile):
    """
    Return solar absorptivity based on the material state.

    Parameters:
        state (str): Material state name (e.g. 'bright', 'dark', etc.)
        alpha_profile (dict): Mapping of state → absorptivity value

    Returns:
        float: Solar absorptivity (0–1)
    """
    return alpha_profile.get(state, alpha_profile.get("default", 0.90))

def get_material_properties(T_surface, GHI, solar_zenith, profile, emissivity_profile, alpha_profile):
    """
    Return state, emissivity, and solar absorptivity based on switching logic.

    Parameters:
        T_surface (float): Surface temperature [K]
        GHI (float): Global horizontal irradiance [W/m²]
        solar_zenith (float): Solar zenith angle [degrees]
        profile (dict): State-switching thresholds
        emissivity_profile (dict): Emissivity values by state
        alpha_profile (dict): Absorptivity values by state

    Returns:
        dict: {'state': str, 'emissivity': float, 'alpha': float}
    """

    # Inject zenith-based override (optional)
    profile = profile.copy()
    if "zenith_threshold" in profile:
        if solar_zenith >= profile["zenith_threshold"]:
            profile["state_map"] = {"dark": {"T_min": 0}}  # force dark at sunset
        elif solar_zenith <= (90 - profile["zenith_threshold"]):
            profile["state_map"] = {"bright": {"T_max": 1000}}  # force bright in strong daylight

    state = get_material_state(T_surface, GHI, profile)
    epsilon = get_emissivity(state, emissivity_profile)
    alpha = get_alpha_solar(state, alpha_profile)

    return {
        "state": state,
        "emissivity": epsilon,
        "alpha": alpha
    }

def get_solar_zenith(lat, lon, times, tz="UTC"):
    """
    Compute solar zenith angle series for a location and times.
    Fallback implementation if pvlib not available.
    """
    if not PVLIB_AVAILABLE:
        print("⚠️ Using simplified zenith calculation - install pvlib for accuracy")
        # Simple approximation - replace with basic solar geometry if needed
        return pd.Series(45.0, index=times)  # Rough approximation
    
    times_local = times.tz_convert(tz) if times.tz is not None else times.tz_localize(tz)
    sp = get_solarposition(times_local, lat, lon)
    return sp["zenith"]

def get_solar_elevation(lat, lon, times, tz="UTC"):
    """
    Compute solar elevation angle series.
    """
    if not PVLIB_AVAILABLE:
        print("⚠️ Using simplified elevation calculation - install pvlib for accuracy")
        return pd.Series(45.0, index=times)  # Rough approximation
        
    times_local = times.tz_convert(tz) if times.tz is not None else times.tz_localize(tz)
    sp = get_solarposition(times_local, lat, lon)
    return sp["elevation"]

if __name__ == "__main__":
    # Example test profile
    profile = {
        "state_map": {
            "bright": {"T_max": 295, "GHI_max": 200},
            "dark":   {"T_min": 295, "GHI_min": 200}
        },
        "default": "static"
    }

    emissivity_profile = {
        "bright": 0.95,
        "dark": 0.80,
        "static": 0.92,
        "default": 0.90
    }

    alpha_profile = {
        "bright": 0.10,
        "dark":   0.90,
        "static": 0.85,
        "default": 0.90
    }
    
    dynamic_profile = {
        "zenith_threshold": 85,  # if sun is near horizon, go into 'dark' mode
        "state_map": {
            "bright": {"T_max": 295, "GHI_max": 200},
            "dark":   {"T_min": 295, "GHI_min": 200}
        },
        "default": "static"
    }

    state = get_material_state(T_surface=298, GHI=300, profile=profile)
    epsilon = get_emissivity(state, emissivity_profile)
    alpha = get_alpha_solar(state, alpha_profile)

    print(f"Material state: {state}")
    print(f"Emissivity: {epsilon}")
    print(f"Absorptivity: {alpha}")

