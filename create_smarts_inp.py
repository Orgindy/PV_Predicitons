#!/usr/bin/env python3
"""
EU-Focused SMARTS ERA5 Input Generator

This script generates SMARTS 2.9.5 input files for computing spectral solar irradiance
using ERA5 reanalysis data and external elevation data extracted from CSV.
It is optimized for EU territories and uses as many ERA5 parameters as possible to 
improve model accuracy.

Usage:
    python smarts_era5_generator_eu.py --input <era5_data.nc> --output <output_dir> [options]

Examples:
    python smarts_era5_generator_eu.py --input era5_data.nc --output ./smarts_inputs
    python smarts_era5_generator_eu.py --input era5_data.nc --output ./smarts_inputs --elevation-file elevation_summary.csv
    python smarts_era5_generator_eu.py --input era5_data.nc --output ./smarts_inputs --config config.yaml
"""
import os
import argparse
import pandas as pd
import numpy as np
import math
import logging
import json
import re
import yaml
import time
import unittest
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import xarray as xr
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# ------------------ Logging Setup ------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('smarts_era5_generator')

# ------------------ SMARTS Constants ------------------
DEFAULT_COMMENT = "ERA5_SMARTS_input"
DEFAULT_CO2 = 420.0        # ppmv
DEFAULT_SOLAR_CONSTANT = 1361.0  # W/m²

# ------------------ Reference Atmosphere Models ------------------
# Define available reference atmosphere models in SMARTS
REFERENCE_ATMOSPHERES = {
    'TROP': "Tropical atmosphere (15°N annual average)",
    'MLS': "Mid-latitude summer (45°N, July)",
    'MLW': "Mid-latitude winter (45°N, January)",
    'SAS': "Sub-arctic summer (60°N, July)",
    'SAW': "Sub-arctic winter (60°N, January)",
    'USSA': "U.S. Standard Atmosphere 1976 (global annual average)",
}

# ------------------ Ground Albedo Table ------------------
GROUND_ALBEDO = {
    "URBAN": 18,
    "GRASS": 30,
    "LIGHT_SOIL": 38,
    "CONIFER": 10,
    "DECIDUOUS": 15,
    "WATER": 5,
    "SNOW": 85,
    "SAND": 40,
    "CONCRETE": 32
}

# ------------------ EU Geographic Boundaries ------------------
EU_BOUNDS = {
    'lat_min': 40,
    'lat_max': 70.0,
    'lon_min': -30.0,
    'lon_max': 40.0
}

# ------------------ ERA5 Variable Mapping ------------------
# Here we map ERA5 variable names to internal SMARTS keys and provide conversion functions.
ERA5_VARIABLE_MAP = {
    'sp':    ('pressure', lambda x: x / 100),                   # Surface pressure: Pa → hPa
    't2m':   ('temperature', lambda x: x - 273.15),             # 2m temperature: K → °C
    'd2m':   ('dewpoint', lambda x: x - 273.15),                # 2m dewpoint: K → °C
    'tco3':  ('ozone', lambda x: x * 46729.0),                  # Total column ozone: kg/m² → DU
    'tcwv':  ('precipitable_water', lambda x: x / 10),          # Total column water vapor: kg/m² → cm
    'skt':   ('skin_temperature', lambda x: x - 273.15),        # Skin temperature: K → °C
    'tcc':   ('cloud_cover', lambda x: x * 100),                # Total cloud cover: (0–1) → %
    'ssrd':  ('solar_radiation_down', lambda x: x),             # Surface solar radiation down
    'strd':  ('thermal_radiation_down', lambda x: x),           # Surface thermal radiation downwards
    'fal':   ('forecast_albedo', lambda x: x * 100),            # Forecast albedo: (0–1) → %
    'asn':   ('snow_albedo', lambda x: x * 100),                # Snow albedo: (0–1) → %
    'aluvp': ('uv_albedo_direct', lambda x: x * 100),           # UV visible albedo direct: (0–1) → %
    'aluvd': ('uv_albedo_diffuse', lambda x: x * 100),          # UV visible albedo diffuse: (0–1) → %
    'alnip': ('nir_albedo_direct', lambda x: x * 100),          # Near IR albedo direct: (0–1) → %
    'alnid': ('nir_albedo_diffuse', lambda x: x * 100),         # Near IR albedo diffuse: (0–1) → %
    'hcc':   ('high_cloud_cover', lambda x: x * 100),           # High cloud cover: (0–1) → %
    'mcc':   ('medium_cloud_cover', lambda x: x * 100),         # Medium cloud cover: (0–1) → %
    'lcc':   ('low_cloud_cover', lambda x: x * 100),            # Low cloud cover: (0–1) → %
    'ssrdc': ('clear_sky_solar_rad', lambda x: x),              # Clear-sky solar rad
    'cdir':  ('clear_sky_direct_rad', lambda x: x),             # Clear-sky direct solar rad
    'fdir':  ('direct_solar_rad', lambda x: x),                 # Direct solar radiation
    'duvr':  ('uv_radiation_down', lambda x: x),                # Downward UV radiation
    'tcdw':  ('cloud_water', lambda x: x),                      # Cloud water (units assumed consistent)
    'tp':    ('precipitation', lambda x: x * 1000),             # Total precipitation: m → mm
    'mx2t':  ('max_temperature', lambda x: x - 273.15),         # Maximum temperature: K → °C
    'mn2t':  ('min_temperature', lambda x: x - 273.15),         # Minimum temperature: K → °C
    'u10':   ('u_wind', lambda x: x),                           # 10m u-component
    'v10':   ('v_wind', lambda x: x),                           # 10m v-component
    'tisr':  ('toa_incident_solar_rad', lambda x: x),           # TOA incident solar radiation
    'tcrw':  ('rain_water', lambda x: x * 1000),                # Total column rain water: m → mm
    'e':     ('evaporation', lambda x: x * 1000),               # Evaporation: m → mm (if applicable)
    'ptype': ('precipitation_type', lambda x: x),               # Precipitation type: categorical/string
    'slt':   ('soil_type', lambda x: x),                        # Soil type indicator: categorical
    'k_index': ('k_index', lambda x: x),                        # Convective instability index
    # Added new parameters for improved modeling
    'ssr':   ('net_solar_radiation', lambda x: x),              # Surface net solar radiation
    'ssrc':  ('net_solar_radiation_clear', lambda x: x),        # Surface net solar radiation, clear sky 
    'str':   ('net_thermal_radiation', lambda x: x),            # Surface net thermal radiation
    'slhf':  ('latent_heat_flux', lambda x: x),                 # Surface latent heat flux
    'sshf':  ('sensible_heat_flux', lambda x: x),               # Surface sensible heat flux
    'tsr':   ('top_net_solar_radiation', lambda x: x),          # Top net solar radiation
    'tsrc':  ('top_net_solar_radiation_clear', lambda x: x),    # Top net solar radiation, clear sky
    'ttr':   ('top_net_thermal_radiation', lambda x: x),        # Top net thermal radiation
    'ttrc':  ('top_net_thermal_radiation_clear', lambda x: x),  # Top net thermal radiation, clear sky
    'strdc': ('thermal_radiation_down_clear', lambda x: x),     # Surface thermal radiation downward, clear sky
}

# Subset of critical variables that must be present in the ERA5 dataset
CRITICAL_ERA5_VARS = ['sp', 't2m', 'd2m', 'tco3', 'tcwv']

# ------------------ Representative Days and Time Points ------------------
REPRESENTATIVE_DAYS = [
    {"date": "2023-01-15", "description": "Mid-Winter"},
    {"date": "2023-02-05", "description": "Late Winter"},
    {"date": "2023-03-21", "description": "Spring Equinox"},
    {"date": "2023-04-15", "description": "Mid-Spring"},
    {"date": "2023-06-21", "description": "Summer Solstice"},
    {"date": "2023-07-15", "description": "Mid-Summer"},
    {"date": "2023-08-15", "description": "Late Summer"},
    {"date": "2023-09-23", "description": "Fall Equinox"},
    {"date": "2023-10-15", "description": "Mid-Fall"},
    {"date": "2023-12-01", "description": "Early Winter"},
]

TIME_POINTS = [
    {"hour": 8, "minute": 0, "name": "morning"},
    {"hour": 12, "minute": 0, "name": "noon"},
    {"hour": 17, "minute": 0, "name": "evening"},
]

# ------------------ EU Representative Cities and Geographic Features ------------------
EU_COUNTRIES = [
    ["Austria", "Vienna", 48.21, 16.37, "URBAN"],
    ["Belgium", "Brussels", 50.85, 4.35, "URBAN"],
    ["Bulgaria", "Sofia", 42.70, 23.32, "URBAN"],
    ["Croatia", "Zagreb", 45.81, 15.98, "URBAN"],
    ["Cyprus", "Nicosia", 35.17, 33.36, "URBAN"],
    ["Czech Republic", "Prague", 50.08, 14.44, "URBAN"],
    ["Denmark", "Copenhagen", 55.68, 12.57, "URBAN"],
    ["Estonia", "Tallinn", 59.44, 24.75, "URBAN"],
    ["Finland", "Helsinki", 60.17, 24.94, "URBAN"],
    ["France", "Paris", 48.85, 2.35, "URBAN"],
    ["Germany", "Berlin", 52.52, 13.40, "URBAN"],
    ["Greece", "Athens", 37.98, 23.73, "URBAN"],
    ["Hungary", "Budapest", 47.50, 19.04, "URBAN"],
    ["Ireland", "Dublin", 53.35, -6.26, "URBAN"],
    ["Italy", "Rome", 41.90, 12.50, "URBAN"],
    ["Latvia", "Riga", 56.95, 24.11, "URBAN"],
    ["Lithuania", "Vilnius", 54.69, 25.28, "URBAN"],
    ["Luxembourg", "Luxembourg", 49.61, 6.13, "URBAN"],
    ["Malta", "Valletta", 35.90, 14.51, "URBAN"],
    ["Netherlands", "Amsterdam", 52.37, 4.90, "URBAN"],
    ["Poland", "Warsaw", 52.23, 21.01, "URBAN"],
    ["Portugal", "Lisbon", 38.72, -9.14, "URBAN"],
    ["Romania", "Bucharest", 44.43, 26.11, "URBAN"],
    ["Slovakia", "Bratislava", 48.15, 17.11, "URBAN"],
    ["Slovenia", "Ljubljana", 46.06, 14.51, "URBAN"],
    ["Spain", "Madrid", 40.42, -3.70, "URBAN"],
    ["Sweden", "Stockholm", 59.33, 18.07, "URBAN"],
]

EU_GEOGRAPHIC_FEATURES = [
    ["Alps", 46.50, 9.83, "SNOW"],
    ["Pyrenees", 42.63, 0.65, "SNOW"],
    ["Baltic Sea", 58.00, 20.00, "WATER"],
    ["Mediterranean Sea", 40.00, 10.00, "WATER"],
    ["North Sea", 56.00, 3.00, "WATER"],
    ["Atlantic Coast", 43.50, -1.50, "WATER"],
    ["Carpathian Mountains", 47.00, 25.00, "CONIFER"],
    ["Black Forest", 48.27, 8.35, "CONIFER"],
    ["Po Valley", 45.00, 9.00, "GRASS"],
    ["Andalusian Plains", 37.50, -5.50, "GRASS"],
    ["Apennine Mountains", 42.50, 13.50, "DECIDUOUS"],
    ["Scandinavian Mountains", 63.00, 12.00, "SNOW"],
    ["Danube Delta", 45.00, 29.00, "WATER"],
    ["Northern European Plain", 52.00, 15.00, "GRASS"],
    ["Sicily", 37.60, 14.01, "LIGHT_SOIL"],
    ["Sardinia", 40.12, 9.01, "LIGHT_SOIL"],
    ["Corsica", 42.00, 9.00, "LIGHT_SOIL"],
    ["Adriatic Coast", 44.00, 14.00, "WATER"],
]

# ------------------ Enhanced Parameter Calculations -----------------
def select_representative_days(input_file, output_file, method='solstice', sample_days=4):
    """
    Selects representative days for each season to reduce computational load.
    
    Parameters:
    - input_file (str): Path to the climate or irradiance CSV file.
    - output_file (str): Path to the output CSV file.
    - method (str): Selection method ('solstice', 'quartile', 'average', 'fixed').
    - sample_days (int): Number of representative days per season (default 4).
    
    Returns:
    - None
    """
    # Load input data
    df = pd.read_csv(input_file)
    
    # Extract month and day information
    df['date'] = pd.to_datetime(df['timestamp'])
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
    # Define seasons (Northern Hemisphere)
    seasons = {
        'winter': [12, 1, 2],
        'spring': [3, 4, 5],
        'summer': [6, 7, 8],
        'autumn': [9, 10, 11]
    }
    
    # Select representative days
    selected_days = []
    for season, months in seasons.items():
        season_df = df[df['month'].isin(months)]
        
        if method == 'solstice':
            # Use solstices and equinoxes
            if season == 'winter':
                selected_days.append(season_df[season_df['month'] == 12].iloc[[0]])
                selected_days.append(season_df[season_df['month'] == 1].iloc[[-1]])
            elif season == 'summer':
                selected_days.append(season_df[season_df['month'] == 6].iloc[[0]])
                selected_days.append(season_df[season_df['month'] == 8].iloc[[-1]])
            else:
                selected_days.append(season_df.sample(n=sample_days))
        
        elif method == 'quartile':
            # Use quartiles for each season
            q1 = season_df.sample(frac=0.25).head(1)
            q2 = season_df.sample(frac=0.25).tail(1)
            q3 = season_df.sample(frac=0.75).head(1)
            q4 = season_df.sample(frac=0.75).tail(1)
            selected_days.extend([q1, q2, q3, q4])
        
        elif method == 'average':
            # Use average day for each month
            monthly_avg = season_df.groupby('month').mean().reset_index()
            selected_days.append(monthly_avg)
        
        elif method == 'fixed':
            # Fixed number of random days per season
            selected_days.append(season_df.sample(n=sample_days))
    
    # Combine and save
    final_df = pd.concat(selected_days).reset_index(drop=True)
    final_df.to_csv(output_file, index=False)
    print(f"✅ Representative days saved to {output_file}")


def extract_era5_values(ds, lat, lon, time, stats=None):
    """
    Extract relevant ERA5 parameters for a given location and time.

    Args:
        ds (xarray.Dataset): ERA5 dataset
        lat (float): Latitude
        lon (float): Longitude
        time (datetime): Timestamp
        stats (dict): Statistics dictionary for tracking missing/invalid data

    Returns:
        dict: Dictionary of processed and validated parameters
    """
    # ERA5 variables needed
    needed_vars = list(ERA5_VARIABLE_MAP.keys())
    point_data = {}

    # Find nearest grid point (assumes regular lat-lon grid)
    lat_idx = np.abs(ds.latitude - lat).argmin().item()
    lon_idx = np.abs(ds.longitude - lon).argmin().item()

    # Time selection
    if time not in ds.time.values:
        closest_time_idx = np.argmin(np.abs(ds.time.values - np.datetime64(time)))
        actual_time = pd.to_datetime(ds.time.values[closest_time_idx])
        if stats is not None:
            stats['time_warnings'] = stats.get('time_warnings', 0) + 1
    else:
        actual_time = pd.to_datetime(time)

    # Loop over variables to extract
    for var, (key, transform) in ERA5_VARIABLE_MAP.items():
        try:
            val = ds[var].sel(time=actual_time, latitude=ds.latitude[lat_idx], longitude=ds.longitude[lon_idx], method='nearest').values.item()
            point_data[key] = transform(val)
        except KeyError:
            if stats is not None:
                stats['missing_critical'][var] = stats.get('missing_critical', {}).get(var, 0) + 1
            point_data[key] = None
        except Exception as e:
            if stats is not None:
                stats['extraction_errors'][var] = stats.get('extraction_errors', {}).get(var, 0) + 1
                stats['total_extraction_errors'] = stats.get('total_extraction_errors', 0) + 1
            point_data[key] = None

    # Derived parameters
    t_c = point_data.get('temperature')
    td_c = point_data.get('dewpoint')

    if t_c is not None and td_c is not None:
        point_data['rh'] = calculate_rh_from_dewpoint(t_c, td_c)
    else:
        point_data['rh'] = validate_param('rh', None, stats)

    # Wind speed
    u = point_data.get('u_wind')
    v = point_data.get('v_wind')
    point_data['wind_speed'] = calculate_wind_speed(u, v)

    # Solar geometry
    zenith, azimuth, _ = calculate_solar_position(lat, lon, actual_time)
    point_data['solar_zenith'] = validate_param('solar_zenith', zenith, stats)
    point_data['solar_azimuth'] = azimuth

    # Albedo
    point_data['albedo'] = calculate_enhanced_albedo(point_data)

    # Aerosol optical depth (ERA5-based estimate)
    point_data['aod550'] = estimate_aod_era5_only(point_data)

    # Aerosol model
    point_data['aerosol_type'] = determine_advanced_aerosol_type(point_data)

    # Add timestamp (for metadata)
    point_data['time'] = actual_time

    return point_data

def calculate_cloud_effects(params):
    """
    Calculate cloud modification factors for direct and diffuse radiation.
    
    Args:
        params: Dictionary of parameters containing cloud cover information
        
    Returns:
        Dictionary with direct_mod and diffuse_increase factors
    """
    high_cloud = params.get('high_cloud_cover', 0)
    mid_cloud = params.get('medium_cloud_cover', 0)
    low_cloud = params.get('low_cloud_cover', 0)
    
    # Different cloud types have different optical properties
    # High clouds (cirrus) are more transparent than low clouds (stratus)
    high_transmittance = 1.0 - (high_cloud/100 * 0.3)  # High clouds block ~30% of direct
    mid_transmittance = 1.0 - (mid_cloud/100 * 0.5)    # Mid clouds block ~50% of direct
    low_transmittance = 1.0 - (low_cloud/100 * 0.8)    # Low clouds block ~80% of direct
    
    # Calculate combined effect
    direct_transmittance = high_transmittance * mid_transmittance * low_transmittance
    
    return {
        'direct_mod': direct_transmittance,
        'diffuse_increase': 1.0 + (1.0 - direct_transmittance) * 0.5  # Portion of blocked direct becomes diffuse
    }

def determine_advanced_aerosol_type(params):
    """
    More sophisticated aerosol type determination using multiple parameters.
    
    Args:
        params: Dictionary of parameters extracted from ERA5 data
        
    Returns:
        String code for the appropriate aerosol model
    """
    # Safe fallbacks to prevent NoneType errors
    lat = params.get('lat') or 45.0
    lon = params.get('lon') or 10.0
    elevation = params.get('elevation') or 200.0
    land_cover = params.get('land_cover', 'LIGHT_SOIL') or 'LIGHT_SOIL'
    rh = params.get('rh', 50)
    wind_speed = params.get('wind_speed') or 0.0
    precipitation = params.get('precipitation') or 0.0

    # Maritime conditions
    if (
        land_cover == 'WATER' or 
        (elevation < 100 and (
            abs(lon - 14.0) < 5.0 or 
            abs(lon - 20.0) < 5.0 or 
            abs(lon - 3.0) < 5.0 or 
            lon < -5.0))
    ):
        return 'S&F_MARIT'

    # Urban conditions with pollution indicators
    if land_cover == 'URBAN':
        return 'S&F_URBAN'

    # Clean conditions after heavy rain
    if precipitation > 5.0:
        return 'S&F_TROPO'

    # High elevation = cleaner air
    if elevation > 1500:
        return 'S&F_TROPO'

    # Default rural model with humidity consideration
    if rh > 70:
        return 'S&F_RURAL'
    else:
        return 'S&F_RURAL'

def calculate_enhanced_albedo(params):
    """
    Calculate more accurate albedo using additional ERA5 parameters.
    
    Args:
        params: Dictionary of parameters extracted from ERA5 data
        
    Returns:
        Enhanced albedo value (0-100)
    """
    # Check if all albedo components are available
    has_all_albedo_components = all(x in params and params[x] is not None for x in [
            'uv_albedo_direct', 'uv_albedo_diffuse', 
            'nir_albedo_direct', 'nir_albedo_diffuse'])
    
    if has_all_albedo_components:
        # Calculate solar zenith dependent albedo
        solar_zenith = params.get('solar_zenith', 45)
        zenith_factor = min(1.0, 1.0 + (solar_zenith - 30) / 60.0)
        
        # Blend direct vs diffuse based on cloud cover
        cloud_cover = params.get('cloud_cover', 0) / 100
        diffuse_fraction = max(0.2, min(0.9, cloud_cover * 0.7 + 0.2))
        
        # Calculate UV and NIR components
        uv_albedo = (params['uv_albedo_diffuse'] * diffuse_fraction + 
                     params['uv_albedo_direct'] * (1 - diffuse_fraction))
        nir_albedo = (params['nir_albedo_diffuse'] * diffuse_fraction + 
                      params['nir_albedo_direct'] * (1 - diffuse_fraction))
        
        cloud_cover = params.get('cloud_cover', 0)
        cloud_cover = 0 if cloud_cover is None else cloud_cover

        # Account for snow if available
        if 'snow_albedo' in params and params['snow_albedo'] is not None:
            snow_cover = params.get('snow_cover', 0) / 100
            if snow_cover > 0.05:  # More than 5% snow cover
                return snow_cover * params['snow_albedo'] + (1 - snow_cover) * (0.4 * uv_albedo + 0.6 * nir_albedo)
        if 'snow_cover' not in params:
            if params.get('temperature', 5) < 0 and params.get('precipitation', 0) > 2.0:
                params['snow_cover'] = 0.5  # Assume partial snow cover

        if 'snow_cover' in params and 'snow_albedo' in params and params['snow_albedo'] is not None:
            snow_cover = min(1.0, params['snow_cover'])
            albedo = snow_cover * params['snow_albedo'] + (1 - snow_cover) * (0.4 * uv_albedo + 0.6 * nir_albedo)
            
            return albedo

        # Return weighted average - apply zenith factor
        return (0.4 * uv_albedo + 0.6 * nir_albedo) * zenith_factor
    
    # Alternative: check if we have forecast or generic albedo
    if 'forecast_albedo' in params and params['forecast_albedo'] is not None:
        return params['forecast_albedo']
    
    # Fall back to standard albedo calculation based on land cover
    land_cover = params.get('land_cover', 'LIGHT_SOIL')
    return GROUND_ALBEDO.get(land_cover, 30)

def add_elevation_and_pressure(input_file, elevation_file, output_file):
    """
    Adds elevation and pressure corrections to SMARTS input files.
    
    Parameters:
    - input_file (str): Path to the SMARTS input file.
    - elevation_file (str): Path to the elevation CSV file.
    - output_file (str): Path to the corrected SMARTS input file.
    
    Returns:
    - None
    """
    # Load the main SMARTS input data
    df = pd.read_csv(input_file)

    # Load elevation data
    elevation_df = pd.read_csv(elevation_file)
    
    # Merge on location
    merged_df = pd.merge(df, elevation_df, on=['latitude', 'longitude'], how='left')
    
    # Check for missing elevation values
    if merged_df['elevation'].isna().sum() > 0:
        print("❌ Missing elevation data for some locations. Check your elevation file.")
        return
    
    # Calculate pressure at altitude (barometric formula)
    # P = P0 * exp(-g * M * h / (R * T))
    # Assuming P0 = 1013.25 hPa (sea level), T = 288.15 K, M = 0.029 kg/mol, g = 9.81 m/s², R = 8.314 J/(mol·K)
    P0 = 1013.25  # hPa
    M = 0.029     # kg/mol
    g = 9.81      # m/s²
    R = 8.314     # J/(mol·K)
    T = 288.15    # K

    merged_df['pressure'] = P0 * np.exp(-M * g * merged_df['elevation'] / (R * T))

    # Save the updated file
    merged_df.to_csv(output_file, index=False)
    print(f"✅ Elevation and pressure added to {output_file}")


def determine_enhanced_atmosphere(params):
    lat = params.get('lat', 45.0)
    temp = params.get('temperature', 15.0)
    month = params.get('time').month if 'time' in params and params['time'] else 7

    is_summer = 5 <= month <= 9
    abs_lat = abs(lat)

    if abs_lat < 23.5:
        return 'TROP'
    elif 23.5 <= abs_lat < 45:
        return 'MLS' if is_summer else 'MLW'
    elif 45 <= abs_lat < 55:
        # Mid-latitudes still; use temp as secondary classifier
        if temp > 15:
            return 'MLS'
        elif temp < 5:
            return 'MLW'
        else:
            return 'USSA'
    elif abs_lat >= 55:
        return 'SAS' if is_summer else 'SAW'
    else:
        return 'USSA'
    
    # Fall back to standard method if temperature not available
    is_summer = 5 <= month <= 9
    
    if abs(lat) < 23.5:
        return 'TROP'
    elif 23.5 <= abs(lat) < 50:
        return 'MLS' if is_summer else 'MLW'
    elif abs(lat) >= 50:
        return 'SAS' if is_summer else 'SAW'
    else:
        return 'USSA'

def estimate_aod_era5_only(params):
    lat = params.get('lat', 45)
    land_cover = params.get('land_cover', 'LIGHT_SOIL')
    rh = params.get('rh', 50)
    wind_speed = params.get('wind_speed', 0)
    temp = params.get('temperature', 15)
    precip = params.get('precipitation', 0)
    elevation = params.get('elevation', 200)

    # Base AOD defaults by land type and region
    if land_cover == 'URBAN':
        base_aod = 0.18
    elif land_cover == 'WATER':
        base_aod = 0.08
    elif land_cover in ['SAND', 'LIGHT_SOIL']:
        base_aod = 0.15 if lat < 45 else 0.10
    elif land_cover in ['SNOW', 'CONIFER']:
        base_aod = 0.06
    else:
        base_aod = 0.10

    # RH correction (hygroscopic growth)
    if rh > 60:
        rh_factor = 1 + 0.25 * ((min(rh, 95) - 60) / 35)
        base_aod *= rh_factor

    # Precipitation washout (strong effect)
    if precip > 0.5:
        base_aod *= max(0.5, np.exp(-0.4 * precip))  # stronger washout

    # Wind boost only if dust-prone
    if wind_speed > 5 and land_cover in ['SAND', 'LIGHT_SOIL']:
        dust_boost = 1.0 + 0.1 * (wind_speed - 5)
        base_aod *= dust_boost

    # Elevation correction
    base_aod *= np.exp(-elevation / 3000.0)

    # Temperature can enhance photochemistry (mild effect)
    if temp > 25:
        base_aod *= 1.05

    return max(0.03, min(base_aod, 0.4))  # clamp to realistic range

# ------------------ Configuration Handling ------------------
def load_config(config_path):
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None

def apply_config_to_args(args, config):
    """Apply configuration values to command-line arguments."""
    if not config:
        return args
    
    for key, value in config.items():
        if hasattr(args, key) and value is not None:
            setattr(args, key, value)
            logger.debug(f"Config: Set {key} = {value}")
    
    return args

# ------------------ Utility Functions ------------------
def validate_param(name, value, stats=None):
    """Validate and clean SMARTS input parameters."""
    defaults = {
        'temperature': 15.0,
        'pressure': 1013.25,
        'albedo': 30.0,
        'solar_zenith': 45.0,
        'rh': 50.0,
        'ozone': 300.0,
        'aod550': 0.12,
    }
    ranges = {
        'temperature': (-80, 60),
        'pressure': (300, 1100),
        'albedo': (0, 100),
        'solar_zenith': (0, 89.9),  # Cap at 89.9 for SMARTS compatibility
        'rh': (1, 99),  # Use narrower range for RH
        'ozone': (100, 500),
        'aod550': (0.01, 1.5),
    }

    # Handle missing or NaN values
    if value is None or isinstance(value, str) or (isinstance(value, float) and math.isnan(value)):
        if stats is not None and name in defaults:
            stats['missing_params'][name] = stats.get('missing_params', {}).get(name, 0) + 1
        return defaults.get(name)

    # Handle out-of-range values
    if name in ranges and (value < ranges[name][0] or value > ranges[name][1]):
        if stats is not None:
            stats['invalid_params'][name] = stats.get('invalid_params', {}).get(name, 0) + 1
        
        # Clamp to range rather than reset to default
        if value < ranges[name][0]:
            logger.warning(f"⚠️ {name} value {value} below range minimum, clamping to {ranges[name][0]}")
            return ranges[name][0]
        else:
            logger.warning(f"⚠️ {name} value {value} above range maximum, clamping to {ranges[name][1]}")
            return ranges[name][1]

    return value

def validate_era5_dataset(ds):
    """Validate that the ERA5 dataset contains required variables."""
    missing_vars = [var for var in CRITICAL_ERA5_VARS if var not in ds.data_vars]
    
    if missing_vars:
        logger.warning(f"Missing critical variables in ERA5 dataset: {missing_vars}")
        return False
    
    # Check dimensions
    required_dims = ['time', 'latitude', 'longitude']
    missing_dims = [dim for dim in required_dims if dim not in ds.dims]
    if missing_dims:
        logger.warning(f"Missing critical dimensions in ERA5 dataset: {missing_dims}")
        return False
        
    logger.info(f"ERA5 dataset validated successfully. Contains {len(ds.data_vars)} variables.")
    return True

def calculate_solar_position(lat, lon, time):
    """
    More accurate solar position calculation using a better model.
    """
    # Convert time to decimal day of year
    day_of_year = time.timetuple().tm_yday
    hour = time.hour + time.minute / 60.0 + time.second / 3600.0
    
    # Calculate solar declination with more terms
    angle = 2 * math.pi * (day_of_year - 1) / 365.25
    declination = 0.006918 - 0.399912 * math.cos(angle) + 0.070257 * math.sin(angle) \
                 - 0.006758 * math.cos(2*angle) + 0.000907 * math.sin(2*angle) \
                 - 0.002697 * math.cos(3*angle) + 0.001480 * math.sin(3*angle)
    
    # More accurate equation of time
    eot = 229.18 * (0.000075 + 0.001868 * math.cos(angle) - 0.032077 * math.sin(angle) 
                  - 0.014615 * math.cos(2*angle) - 0.040849 * math.sin(2*angle))
    
    # Improved timezone and longitude correction
    timezone = calculate_timezone(lon, lat)
    lon_correction = 4 * (lon - timezone * 15)
    time_correction = eot + lon_correction
    
    # Solar time with corrections
    solar_time = hour + time_correction / 60.0
    
    # Hour angle (in degrees)
    hour_angle = 15 * (solar_time - 12)
    
    # Solar zenith angle with higher precision
    lat_rad = math.radians(lat)
    decl_rad = math.radians(declination)
    hour_angle_rad = math.radians(hour_angle)
    
    # Improved formula for zenith calculation
    cos_zenith = (math.sin(lat_rad) * math.sin(decl_rad) + 
                 math.cos(lat_rad) * math.cos(decl_rad) * math.cos(hour_angle_rad))
    cos_zenith = max(-1, min(1, cos_zenith))  # Ensure within valid range
    solar_zenith = math.degrees(math.acos(cos_zenith))
    
    # Handle sun below horizon cases properly
    if solar_zenith > 90:
        logger.debug(f"Sun below horizon at lat={lat}, lon={lon}, time={time}. Zenith = {solar_zenith:.2f}°")
        # Don't cap - return actual value but flag it
        # SMARTS can handle high zenith angles appropriately
        # Only cap at extreme values that would break SMARTS
        if solar_zenith > 95:
            solar_zenith = 95.0  # More reasonable cap for SMARTS
    
    # More accurate azimuth calculation
    x = (math.sin(hour_angle_rad) * math.cos(decl_rad))
    y = (math.cos(hour_angle_rad) * math.sin(lat_rad) * math.cos(decl_rad) - 
         math.sin(decl_rad) * math.cos(lat_rad))
    
    solar_azimuth = math.degrees(math.atan2(x, y))
    
    # Normalize to compass bearing (0-360°)
    solar_azimuth = (solar_azimuth + 180) % 360
    
    return solar_zenith, solar_azimuth, time_correction
def calculate_timezone(lon, lat):
    """
    Calculate timezone based on longitude, but with awareness of political boundaries.
    For European countries, this is a reasonable approximation.
    
    Args:
        lon: Longitude in degrees
        lat: Latitude in degrees
    
    Returns:
        Timezone offset from UTC in hours
    """
    # Standard approach: divide by 15 (15 degrees per timezone)
    standard_tz = round(lon / 15.0)
    
    # Special cases for European countries that don't follow standard timezone rules
    # Spain and France are mostly in UTC+1 despite geography
    if -10 <= lon <= 3 and 36 <= lat <= 51:  # Western Europe
        return 1
    # Eastern edges of Finland/Romania/Greece use UTC+2
    elif 25 <= lon <= 30 and 40 <= lat <= 70:  # Eastern Europe
        return 2
    
    return standard_tz

def calculate_pressure(elevation, latitude=45.0):
    """
    Estimate atmospheric pressure (in mb) based on elevation (meters) and latitude.
    """
    altitude_km = elevation / 1000.0
    p0 = 1013.25
    if abs(latitude) < 15:
        h0 = 8.0
    elif abs(latitude) < 45:
        h0 = 7.8
    else:
        h0 = 7.5
    return p0 * math.exp(-altitude_km / h0)

def determine_season(month, lat):
    """
    Determine season: 'SUMMER' for 3<=month<=8 (N Hemisphere) or vice versa.
    More accurate than simple hemisphere division.
    """
    # Northern Hemisphere
    if lat >= 0:
        if 5 <= month <= 8:  # May through August
            return 'SUMMER'
        elif month >= 11 or month <= 2:  # November through February
            return 'WINTER'
        else:  # March, April, September, October
            return 'SUMMER' if lat < 50 else 'WINTER'  # Higher latitudes have longer winters
    # Southern Hemisphere (not really relevant for EU but included for completeness)
    else:
        if 11 <= month <= 12 or 1 <= month <= 2:  # November through February
            return 'SUMMER'
        elif 5 <= month <= 8:  # May through August
            return 'WINTER'
        else:  # March, April, September, October
            return 'WINTER' if lat > -50 else 'SUMMER'  # Higher latitudes have longer winters

def calculate_rh_from_dewpoint(temp_c, dewpoint_c):
    """Estimate relative humidity (RH) from temperature and dewpoint (in °C) using Magnus-Tetens."""
    try:
        # Ensure dewpoint doesn't exceed temperature (physical constraint)
        dewpoint_c = min(dewpoint_c, temp_c)
        a, b = 17.27, 237.7
        svp = 6.112 * np.exp((a * temp_c) / (temp_c + b))
        actual_vp = 6.112 * np.exp((a * dewpoint_c) / (dewpoint_c + b))
        rh = 100.0 * (actual_vp / svp)
        if temp_c < -10:
            rh = min(rh, 85.0)  # Prevent RH > 90% in cold air

        return max(1.0, min(99.0, rh))
    except Exception as e:
        logger.debug(f"Error calculating RH from dewpoint: {e}")
        return 50.0

def calculate_wind_speed(u_wind, v_wind):
    """
    Calculate wind speed from u and v components.
    
    Args:
        u_wind: U component of wind (m/s)
        v_wind: V component of wind (m/s)
        
    Returns:
        Wind speed in m/s
    """
    if u_wind is None or v_wind is None:
        return None
    return np.sqrt(u_wind**2 + v_wind**2)

def estimate_turbidity_from_wind_and_aod(wind_speed, aod550):
    base_turbidity = aod550 / 0.1
    if wind_speed and wind_speed > 5:
        base_turbidity *= 1.0 + 0.03 * (wind_speed - 5)  # milder than before
    return max(1.0, min(5.0, base_turbidity))

def run_unit_tests():
    """
    Run unit tests to validate the code.
    Returns True if all tests pass, False otherwise.
    """
    class TestSmartsGenerator(unittest.TestCase):
        def test_calculate_timezone(self):
            # Test standard calculation
            self.assertEqual(calculate_timezone(15, 50), 1)
            self.assertEqual(calculate_timezone(30, 50), 2)
            
            # Test Western Europe special case
            self.assertEqual(calculate_timezone(0, 45), 1)  # Spain/France
            
            # Test Eastern Europe special case
            self.assertEqual(calculate_timezone(26, 60), 2)  # Finland
            
        def test_calculate_solar_position(self):
            # Test summer solstice at noon in Berlin
            test_time = pd.Timestamp('2023-06-21 12:00:00')
            zenith, azimuth, _ = calculate_solar_position(52.52, 13.40, test_time)
            
            # Approximate check - exact values would need a reference calculation
            self.assertTrue(25 <= zenith <= 35)  # Low solar zenith at summer solstice
            self.assertTrue(160 <= azimuth <= 200)  # Near south at noon
            
            # Test winter at morning in Stockholm
            test_time = pd.Timestamp('2023-12-21 08:00:00')
            zenith, azimuth, _ = calculate_solar_position(59.33, 18.07, test_time)
            
            # High zenith in winter, morning
            self.assertTrue(zenith > 70)
            
        def test_calculate_rh_from_dewpoint(self):
            # Test perfect saturation
            self.assertAlmostEqual(calculate_rh_from_dewpoint(20, 20), 99.0, delta=1.0)
            
            # Test typical case
            self.assertTrue(40 <= calculate_rh_from_dewpoint(20, 10) <= 60)
            
            # Test very dry case
            self.assertTrue(calculate_rh_from_dewpoint(30, 0) < 30)
            
        def test_calculate_wind_speed(self):
            # Test simple cases
            self.assertAlmostEqual(calculate_wind_speed(3, 4), 5.0, delta=0.1)
            self.assertAlmostEqual(calculate_wind_speed(0, 5), 5.0, delta=0.1)
            self.assertAlmostEqual(calculate_wind_speed(-3, 0), 3.0, delta=0.1)
            
        def test_validate_param(self):
            # Test valid param
            self.assertEqual(validate_param('temperature', 25), 25)
            
            # Test out of range
            self.assertNotEqual(validate_param('temperature', 100), 100)
            
            # Test None
            self.assertIsNotNone(validate_param('temperature', None))
            
        def test_determine_advanced_aerosol_type(self):
            # Test urban
            params = {'lat': 50, 'lon': 15, 'elevation': 100, 'land_cover': 'URBAN', 'rh': 50}
            self.assertEqual(determine_advanced_aerosol_type(params), 'S&F_URBAN')
            
            # Test high elevation
            params = {'lat': 45, 'lon': 10, 'elevation': 2000, 'land_cover': 'GRASS'}
            self.assertEqual(determine_advanced_aerosol_type(params), 'S&F_TROPO')
            
            # Test water
            params = {'lat': 40, 'lon': 10, 'elevation': 0, 'land_cover': 'WATER'}
            self.assertEqual(determine_advanced_aerosol_type(params), 'S&F_MARIT')
        
        def test_estimate_aod_era5_only(self):
            # Test urban case
            params = {'lat': 50, 'lon': 15, 'land_cover': 'URBAN', 'rh': 80, 'temperature': 25}
            aod = estimate_aod_era5_only(params)
            self.assertTrue(0.25 <= aod <= 0.4, f"Urban AOD {aod} outside expected range")
            
            # Test clean maritime case
            params = {'lat': 60, 'lon': 10, 'land_cover': 'WATER', 'wind_speed': 5, 'rh': 70}
            aod = estimate_aod_era5_only(params)
            self.assertTrue(0.05 <= aod <= 0.15, f"Maritime AOD {aod} outside expected range")
            
        def test_determine_enhanced_atmosphere(self):
            # Test tropical case
            params = {'lat': 15, 'temperature': 28}
            self.assertEqual(determine_enhanced_atmosphere(params), 'TROP')
            
            # Test mid-latitude summer
            params = {'lat': 45, 'temperature': 22}
            self.assertEqual(determine_enhanced_atmosphere(params), 'MLS')
            
            # Test sub-arctic winter
            params = {'lat': 60, 'temperature': -5}
            self.assertEqual(determine_enhanced_atmosphere(params), 'SAW')
    
    # Run tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.loadTestsFromTestCase(TestSmartsGenerator)
    test_runner = unittest.TextTestRunner()
    result = test_runner.run(test_suite)
    
    return result.wasSuccessful()

# ------------------ Elevation Data Functions ------------------
def load_elevation_data(elevation_file):
    """
    Load elevation data from a CSV file.
    
    Args:
        elevation_file: Path to the CSV file containing elevation data
        
    Returns:
        DataFrame with elevation data or None if file not found
    """
    try:
        if not os.path.exists(elevation_file):
            logger.error(f"Elevation file not found: {elevation_file}")
            return None
            
        df = pd.read_csv(elevation_file)
        required_cols = ['latitude', 'longitude', 'elevation']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Elevation file missing required columns: {missing_cols}")
            return None
            
        logger.info(f"Loaded elevation data for {len(df)} locations")
        return df
    except Exception as e:
        logger.error(f"Error loading elevation data: {e}")
        return None

def get_elevation_for_location(elevation_df, lat, lon):
    """
    Find the elevation for a location by finding the closest point in the elevation data.
    
    Args:
        elevation_df: DataFrame with elevation data
        lat: Latitude of the location
        lon: Longitude of the location
        
    Returns:
        Elevation in meters or None if not found
    """
    if elevation_df is None or elevation_df.empty:
        return None
        
    try:
        # Calculate distance to all points
        elevation_df['distance'] = np.sqrt(
            (elevation_df['latitude'] - lat)**2 + 
            (elevation_df['longitude'] - lon)**2
        )
        
        # Find the closest point
        closest = elevation_df.loc[elevation_df['distance'].idxmin()]
        
        # Only use if reasonably close (within 0.5 degrees)
        if closest['distance'] > 0.5:
            logger.debug(f"No elevation data within 0.5 degrees of {lat}, {lon}")
            return None
            
        return closest['elevation']
    except Exception as e:
        logger.debug(f"Error finding elevation for {lat}, {lon}: {e}")
        return None
    
def create_smarts_input_header(params):
    """Create the header section (Cards 1-5) of the SMARTS input file."""
    content = []
    
    # Card 1: Comment
    loc_name = params.get('name', 'location')
    time_str = params['time'].strftime("%Y%m%d_%H%M") if 'time' in params else datetime.now().strftime("%Y%m%d_%H%M")
    comment = f"'{DEFAULT_COMMENT} - {loc_name} at {time_str}' !Card 1 Comment"
    content.append(comment)
    
    # Card 2: Site pressure
    pressure = validate_param('pressure', params.get('pressure'))
    content.append(f"{pressure:.1f} 0. 0. !Card 2 Pressure, altitude, height")
    
    # Card 3: Atmospheric model
    ref_atmosphere = determine_enhanced_atmosphere(params)
    content.append(f"1 !Card 3 IATMOS")
    content.append(f"'{ref_atmosphere}' !Card 3a Atmos")
    
    # Card 4: Water vapor
    if 'precipitable_water' in params and params['precipitable_water'] is not None:
        precip_water = validate_param('precipitable_water', params['precipitable_water'])
        content.append(f"1 !Card 4 IH2O")
        content.append(f"{precip_water:.3f} !Card 4a Water vapor amount")
    else:
        content.append("0 !Card 4 IH2O")  # Calculate from reference atmosphere
    
    # Card 5: Ozone
    if 'ozone' in params and params['ozone'] is not None:
        ozone = validate_param('ozone', params['ozone'])
        content.append(f"1 !Card 5 IO3")
        content.append(f"{ozone:.1f} !Card 5a Ozone amount")
    else:
        content.append("0 !Card 5 IO3")  # Use reference atmosphere
        
    return content, ref_atmosphere

def create_smarts_input_atmosphere(params):
    """Create the atmospheric section (Cards 6-10) of the SMARTS input file."""
    content = []
    
    # Card 6: Gaseous absorption
    content.append("1 !Card 6 IGAS")
    
    # Card 7: CO2 concentration
    content.append(f"{DEFAULT_CO2:.1f} !Card 7 qCO2")
    
    # Card 8: Extraterrestrial spectrum
    content.append("0 !Card 7a ISPCTR")
    # content.append(f"{DEFAULT_SOLAR_CONSTANT:.1f}")
    
    # Card 9: Aerosol model
    aerosol_type = params.get('aerosol_type', 'S&F_RURAL')
    content.append(f"'{aerosol_type}' !Card 8 Aeros")
    
    # Card 10: Turbidity factor
    if 'angstrom_beta' in params and params['angstrom_beta'] is not None:
        beta = validate_param('angstrom_beta', params['angstrom_beta'])
        content.append(f"0 !Card 9 ITURB")
        content.append(f"{beta:.3f} !Card 9a Turbidity coeff. (TAU5)")
    elif 'aod550' in params and params['aod550'] is not None:
        aod = validate_param('aod550', params['aod550'])
        beta = aod / 0.1  # Approximation
        content.append(f"0 !Card 9 ITURB")
        content.append(f"{beta:.3f} !Card 9a Turbidity coeff. (TAU5)")
    else:
        content.append("0 !Card 9 ITURB")  # Default
        
    return content

def create_smarts_input_surface(params):
    """Create the surface and geometry section (Cards 11-17) of the SMARTS input file."""
    content = []

    # Card 10: Albedo
    albedo = validate_param('albedo', params.get('albedo'))
    content.append(f"{int(round(albedo))} !Card 10 IALBDX")

    # Card 10b: Tilt specification
    content.append("1 !Card 10b ITILT")

    if 'tilt_angle' not in params or params['tilt_angle'] is None:
        params['tilt_angle'] = int(round(abs(params['lat'])))
    if 'azimuth' not in params or params['azimuth'] is None:
        params['azimuth'] = 180 if params['lat'] >= 0 else 0

    # ✅ Card 10c: Proper order = tilt, azimuth, albedo
    content.append(f"{params['tilt_angle']} {params['azimuth']} {int(round(albedo))} !Card 10c Tilt variables")

    # Card 11: Wavelength range
    content.append("280 4000 1.0 1367.0 !Card 11 Input wavelengths; solar spectrum")

    # Card 12: Output format
    content.append("2 !Card 12 IPRT")
    content.append("280 4000 .5 !Card12a Print limits")
    content.append("4 !Card12b # Variables to Print")
    content.append("8 9 10 30 !Card12c Variable codes")

    # Card 13-16: Experimental geometry
    content.append("1 !Card 13 ICIRC")
    content.append("0 2.9 0 !Card 13a Receiver geometry")
    content.append("0 !Card 14 ISCAN")
    content.append("0 !Card 15 ILLUM")
    content.append("0 !Card 16 IUV")

    # Card 17: Air mass
    content.append("2 !Card 17 IMASS")

    elevation = params.get('elevation', 0)  # Optional elevation in meters

    # Extract or calculate zenith angle
# Extract or calculate zenith angle
    if 'solar_zenith' in params and params['solar_zenith'] is not None:
        zenith = validate_param('solar_zenith', params['solar_zenith'])
    elif 'lat' in params and 'lon' in params and 'time' in params:
        zenith, _, _ = calculate_solar_position(params['lat'], params['lon'], params['time'])
    else:
        zenith = 48.19  # Fallback

    # Handle high zenith angles properly for air mass calculation
    if zenith > 89:
        air_mass = 10.0  # Maximum air mass for SMARTS
    else:
        zenith = min(89.0, zenith)  # Cap only for air mass calculation
        air_mass = 1 / (math.cos(math.radians(zenith)) + 0.50572 * (96.07995 - zenith) ** -1.6364)
        air_mass *= math.exp(-elevation / 8434.5)
        air_mass = min(10.0, max(1.0, air_mass))
    content.append(f"{air_mass:.1f} !Card 17a Air mass")

    return content

def save_smarts_metadata(metadata_dir, filepath, params, ref_atmosphere):
    """Save metadata for a SMARTS input file."""
    metadata_path = os.path.join(metadata_dir, f"{os.path.basename(filepath)}.json")
    with open(metadata_path, 'w') as f:
        # Include selective metadata
        meta_params = {
            'name': params.get('name'),
            'lat': params.get('lat'),
            'lon': params.get('lon'),
            'elevation': params.get('elevation'),
            'land_cover': params.get('land_cover'),
            'time': params.get('time').isoformat() if 'time' in params else None,
            'solar_zenith': params.get('solar_zenith'),
            'solar_azimuth': params.get('solar_azimuth'),
            'temperature': params.get('temperature'),
            'pressure': params.get('pressure'),
            'rh': params.get('rh'),
            'wind_speed': params.get('wind_speed'),
            'aod550': params.get('aod550'),
            'precipitable_water': params.get('precipitable_water'),
            'ozone': params.get('ozone'),
            'cloud_cover': params.get('cloud_cover'),
            'albedo': params.get('albedo'),
            'aerosol_type': params.get('aerosol_type'),
            'reference_atmosphere': ref_atmosphere,
            'direct_modification': params.get('direct_mod'),
            'diffuse_increase': params.get('diffuse_increase'),
            'high_cloud_cover': params.get('high_cloud_cover'),
            'medium_cloud_cover': params.get('medium_cloud_cover'),
            'low_cloud_cover': params.get('low_cloud_cover'),
            'parameter_source': 'ERA5',
            'format_version': '2.0',
            'generation_timestamp': datetime.now().isoformat()
        }
        json.dump(meta_params, f, indent=2, default=str)
        
    
  # ------------------ SMARTS Input File Creation ------------------
def create_smarts_input(output_path, params, template_name="smarts_input", metadata_dir=None, format_type="standard"):
    """
    Create a SMARTS input file using the provided parameters with exact PDF format,
    ensuring proper separation of Cards 10, 10b, and 10c on separate lines.
    
    Args:
        output_path: Directory to write the file
        params: Dictionary of parameters
        template_name: Base name for the template
        metadata_dir: Directory to save metadata JSON file
        format_type: Format type (standard, compact, detailed)
        
    Returns:
        Path to the created file
    """
    # Check if required parameters are present
    valid, messages = validate_smarts_input(params)
    if not valid:
        missing_msg = "; ".join(messages)
        raise ValueError(f"Missing required parameters for SMARTS input: {missing_msg}")
    
    # Create appropriate filename
    loc_name = params.get('name', 'location')
    safe_loc_name = ''.join(c if c.isalnum() else '_' for c in loc_name).lower()
    
    # Format timestamp
    if 'time' in params:
        time_str = params['time'].strftime("%Y%m%d_%H%M")
    else:
        time_str = datetime.now().strftime("%Y%m%d_%H%M")
        
    filename = f"smarts_{safe_loc_name}_{time_str}.inp"
    filepath = os.path.join(output_path, filename)
    
    # Extract and validate parameters from ERA5 data
    
    # Card 1: Comment (using location/time)
    if format_type == 'example':
        comment = "'Example_6:USSA_AOD_0.084'"
    else:
        time_str = params['time'].strftime("%Y%m%d_%H%M") if 'time' in params else datetime.now().strftime("%Y%m%d_%H%M")
        location = params.get('name', 'location')
        comment = f"'ERA5_SMARTS - {location} at {time_str}'"
    
    # Card 2-2a: Pressure
    pressure = validate_param('pressure', params.get('pressure', 1013.25))
    
    # Card 3-3a: Atmosphere model
    ref_atmosphere = determine_enhanced_atmosphere(params)
    
    # Calculate other parameters
    
    # Card 7: CO2
    co2 = DEFAULT_CO2
    
    # Card 8: Aerosol model
    aerosol_type = params.get('aerosol_type', 'S&F_RURAL')
    if aerosol_type is None:
        aerosol_type = determine_advanced_aerosol_type(params)
    
    # Card 9-9a: Turbidity
    if 'angstrom_beta' in params and params['angstrom_beta'] is not None:
        beta = validate_param('angstrom_beta', params['angstrom_beta'])
    elif 'aod550' in params and params['aod550'] is not None:
        aod = validate_param('aod550', params['aod550'])
        beta = aod / 0.1  # Approximation
    else:
        # Generate AOD estimate if not available
        aod = estimate_aod_era5_only(params)
        beta = aod / 0.1
    
    # Card 10: Albedo
    if 'albedo' in params and params['albedo'] is not None:
        albedo = int(round(validate_param('albedo', params['albedo'])))
    else:
        # Calculate enhanced albedo if not available
        albedo = int(round(calculate_enhanced_albedo(params)))
    
    # Card 17: Air mass
    if 'solar_zenith' in params and params['solar_zenith'] is not None:
        zenith = validate_param('solar_zenith', params['solar_zenith'])
        zenith = min(80, zenith)  # Limit zenith angle to prevent extreme air mass values
        air_mass = 1.0 / math.cos(math.radians(zenith))
        air_mass = min(10.0, max(1.0, air_mass))  # Keep within reasonable bounds
    else:
        # Calculate solar position if not available
        if 'lat' in params and 'lon' in params and 'time' in params:
            zenith, _, _ = calculate_solar_position(params['lat'], params['lon'], params['time'])
            zenith = min(80, zenith)
            air_mass = 1.0 / math.cos(math.radians(zenith))
            air_mass = min(10.0, max(1.0, air_mass))
        else:
            air_mass = 1.5  # Default
    
    # Format exactly matching the PDF example - with Cards 10, 10b, and 10c on separate lines
    content = [
        f"{comment} !Card 1 Comment",
        "1 !Card 2 ISPR",
        f"{pressure:.2f} 0. 0. !Card 2a Pressure, altitude, height",
        "1 !Card 3 IATMOS",
        f"'{ref_atmosphere}' !Card 3a Atmos",
        "1 !Card 4 IH2O",
        "1 !Card 5 IO3",
        "1 !Card 6 IGAS",
        f"{co2:.1f} !Card 7 qCO2",
        "0 !Card 7a ISPCTR",
        f"'{aerosol_type}' !Card 8 Aeros",
        "0 !Card 9 ITURB",
        f"{beta:.3f} !Card 9a Turbidity coeff. (TAU5)",
        # Cards 10, 10b, and 10c properly separated on individual lines
        f"{albedo} !Card 10 IALBDX",
        "1 !Card 10b ITILT",
        f"{albedo} 37 180 !Card 10c Tilt variables",
        "280 4000 1.0 1367.0 !Card 11 Input wavelengths; solar spectrum",
        "2 !Card 12 IPRT",
        "280 4000 .5 !Card12a Print limits",
        "4 !Card12b # Variables to Print",
        "8 9 10 30 !Card12c Variable codes",
        "1 !Card 13 ICIRC",
        "0 2.9 0 !Card 13a Receiver geometry",
        "0 !Card 14 ISCAN",
        "0 !Card 15 ILLUM",
        "0 !Card 16 IUV",
        "2 !Card 17 IMASS",
        f"{air_mass:.1f} !Card 17a Air mass"
    ]
    
    # Save file
    try:
        with open(filepath, 'w') as f:
            f.write('\n'.join(content))
        
        # Save metadata if requested
        if metadata_dir:
            save_smarts_metadata(metadata_dir, filepath, params, ref_atmosphere)
        
        return filepath
    except Exception as e:
        logger.error(f"Error creating SMARTS input file: {e}")
        raise
        
def validate_smarts_input(params):
    """
    Validates that all required parameters for SMARTS are present and reasonable.
    Returns (is_valid, message_list) tuple.
    """
    messages = []
    
    # Essential parameters for SMARTS
    required_params = {
        'lat': "Location latitude",
        'lon': "Location longitude",
        'pressure': "Atmospheric pressure",
        'temperature': "Ambient temperature",
    }
    
    # Check essential parameters
    missing_required = [desc for param, desc in required_params.items() 
                       if param not in params or params[param] is None]
    
    if missing_required:
        messages.append(f"Missing essential parameters: {', '.join(missing_required)}")
        return False, messages
    
    # Check valid ranges for critical parameters
    range_checks = {
        'pressure': (300, 1100),     # hPa
        'temperature': (-80, 60),    # °C
        'solar_zenith': (0, 90),     # degrees
        'rh': (0, 100),              # %
        'ozone': (100, 500),         # DU
        'aod550': (0.01, 1.5),       # unitless
    }
    
    for param, (min_val, max_val) in range_checks.items():
        if param in params and params[param] is not None:
            if params[param] < min_val or params[param] > max_val:
                messages.append(f"Parameter {param} value {params[param]} outside valid range [{min_val}, {max_val}]")
    
    # Only issue warnings if values are invalid but not missing required ones
    return len(missing_required) == 0, messages

def generate_eu_grid(n_points=300, elevation_df=None):
    """
    Generate a grid of EU locations. First include representative cities and features,
    then add systematic grid points if needed.
    Returns a list of [name, lat, lon, elevation, land_cover].
    
    Args:
        n_points: Maximum number of locations to generate
        elevation_df: Optional DataFrame with elevation data
    """
    locations = []
    
    # Add EU country capitals
    for country_info in EU_COUNTRIES:
        country, city, lat, lon, land_cover = country_info
        
        # Get elevation from elevation data if available
        elevation = None
        if elevation_df is not None:
            elevation = get_elevation_for_location(elevation_df, lat, lon)
            
        # Use default if no elevation found
        if elevation is None:
            if land_cover == "URBAN":
                elevation = 100  # Default for cities
            elif land_cover == "WATER":
                elevation = 0    # Default for water
            else:
                elevation = 200  # Default for other areas
            
        locations.append([f"{city}, {country}", lat, lon, elevation, land_cover])
    
    # Add geographic features
    for feature_info in EU_GEOGRAPHIC_FEATURES:
        feature, lat, lon, land_cover = feature_info
        
        # Assign reasonable default elevations based on land cover
        if land_cover == "SNOW":
            default_elev = 2000
        elif land_cover == "WATER":
            default_elev = 0
        elif land_cover == "CONIFER":
            default_elev = 800
        else:
            default_elev = 200
            
        # Get elevation from elevation data if available
        elevation = None
        if elevation_df is not None:
            elevation = get_elevation_for_location(elevation_df, lat, lon)
            
        # Use default if no elevation found
        if elevation is None:
            elevation = default_elev
            
        locations.append([feature, lat, lon, elevation, land_cover])
    
    # Fill remaining points with a systematic grid
    remaining = n_points - len(locations)
    if remaining > 0:
        lat_span = EU_BOUNDS['lat_max'] - EU_BOUNDS['lat_min']
        lon_span = EU_BOUNDS['lon_max'] - EU_BOUNDS['lon_min']
        grid_size = int(math.ceil(math.sqrt(remaining)))
        lat_step = lat_span / grid_size
        lon_step = lon_span / grid_size
        grid_points = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                lat_point = EU_BOUNDS['lat_min'] + (i + 0.5) * lat_step
                lon_point = EU_BOUNDS['lon_min'] + (j + 0.5) * lon_step
                
                # Skip if too close to existing locations
                if any(abs(loc[1] - lat_point) < 0.5 and abs(loc[2] - lon_point) < 0.5 for loc in locations):
                    continue
                
                # Determine land cover based on geographic position
                # Using a more deterministic approach instead of random assignment
                if lat_point > 60:
                    # Northern regions: more likely to be conifer forests or snow
                    if lon_point < 10:  # Western Scandinavia - more mountains
                        land_cover = "SNOW"
                    else:  # Eastern Scandinavia - more forests
                        land_cover = "CONIFER"
                elif lat_point < 43 and lon_point > 10:
                    # Mediterranean regions: more likely to be light soil
                    if lon_point > 20:  # Eastern Med
                        land_cover = "LIGHT_SOIL"
                    else:  # Western Med
                        land_cover = "GRASS"
                elif lon_point < -5 or (lon_point > 10 and lat_point < 50):
                    # Atlantic coast or Mediterranean: could be water
                    if (lon_point < -8) or (lon_point > 15 and lat_point < 45):
                        land_cover = "WATER"
                    else:
                        land_cover = "GRASS"
                else:
                    # Central Europe: mix of deciduous forest and grassland
                    if 48 < lat_point < 55 and 5 < lon_point < 20:
                        land_cover = "DECIDUOUS"
                    else:
                        land_cover = "GRASS"
                
                # Get elevation from elevation data if available
                elevation = None
                if elevation_df is not None:
                    elevation = get_elevation_for_location(elevation_df, lat_point, lon_point)
                
                # Use reasonable defaults based on land cover if no elevation data
                if elevation is None:
                    if land_cover == "SNOW":
                        elevation = 1500
                    elif land_cover == "WATER":
                        elevation = 0
                    elif land_cover == "CONIFER":
                        elevation = 600
                    else:
                        elevation = 250
                
                grid_points.append([f"EU Grid {i * grid_size + j}", lat_point, lon_point, elevation, land_cover])
                
                if len(grid_points) >= remaining:
                    break
            
            if len(grid_points) >= remaining:
                break
                
        locations.extend(grid_points[:remaining])
    
    return locations[:n_points]

def batch_process_locations(locations, days, time_points, ds, args, stats, elevation_df=None):
    """
    Process all locations and time points sequentially (for now).
    
    Args:
        locations: List of location data [name, lat, lon, elevation, land_cover]
        days: List of days to process
        time_points: List of time points to process
        ds: ERA5 dataset (xarray.Dataset)
        args: Command line arguments
        stats: Statistics dictionary for tracking progress
        elevation_df: Optional DataFrame with elevation data
        
    Returns:
        Tuple of (success_count, error_count, skipped_count, successful_files, failed_files)
    """
    # Initialize statistics
    success_count = 0
    error_count = 0
    skipped_count = 0
    successful_files = []
    failed_files = []

    tasks = []
    for location in locations:
        for day in days:
            for tp in time_points:
                tasks.append((location, day, tp, ds, args, stats, elevation_df))

    logger.info(f"Processing {len(tasks)} tasks sequentially")
    
    for i, task in enumerate(tasks):
        try:
            success, filepath, error_msg = process_location_time(task)
            if success:
                if error_msg == "Skipped (already exists)":
                    skipped_count += 1
                    print(f"🔁 Skipped: {filepath}")
                else:
                    success_count += 1
                    successful_files.append(filepath)
                    print(f"✅ Written: {filepath}")
            else:
                error_count += 1
                failed_files.append(error_msg)
                print(f"❌ Failed: {error_msg}")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"⚠️ Unexpected error in task {i}: {error_msg}")
            error_count += 1
            failed_files.append(error_msg)

    return success_count, error_count, skipped_count, successful_files, failed_files


def process_location_time(task):
    try:
        location, day, time_point, ds, args, stats, elevation_df = task
        
        loc_name, lat, lon, elev, land_cover = location
        date = pd.to_datetime(day['date'])
        current_time = pd.Timestamp(
            year=date.year, 
            month=date.month, 
            day=date.day, 
            hour=time_point['hour'], 
            minute=time_point['minute']
        )

        safe_loc_name = ''.join(c if c.isalnum() else '_' for c in loc_name).lower()
        time_str = current_time.strftime("%Y%m%d_%H%M")
        filename = f"smarts_{safe_loc_name}_{time_str}.inp"
        full_path = os.path.join(args.output, filename)

        if args.resume and os.path.exists(full_path):
            print(f"🔁 Skipping {filename} (already exists)")
            return (True, filename, "Skipped (already exists)")

        print(f"⚙️ Extracting ERA5 values for {loc_name} at {current_time}")
        params = extract_era5_values(ds, lat, lon, current_time, stats)

        if not params:
            print(f"🚫 No parameters returned for {loc_name} at {current_time}")
            return (False, None, "Missing parameters")

        params['name'] = loc_name
        params['lat'] = lat
        params['lon'] = lon

        if elev is None and elevation_df is not None:
            elev = get_elevation_for_location(elevation_df, lat, lon)

        params['elevation'] = elev or 0
        params['land_cover'] = land_cover
        params['time'] = current_time

        print(f"📝 Generating SMARTS input file for {loc_name} at {current_time}")
        filepath = create_smarts_input(
            args.output,
            params,
            template_name=f"smarts_{time_point['name']}",
            metadata_dir=args.metadata,
            format_type=args.format
        )
        print(f"✅ Created: {filepath}")
        return (True, filepath, None)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Error processing {loc_name} at {current_time}: {error_msg}")
        return (False, None, error_msg)

def main():
    """
    Main function: Load ERA5 data, process locations & times, extract parameters,
    and generate SMARTS input files.
    
    This function handles command-line arguments, loads data, processes locations,
    and coordinates the overall generation of SMARTS input files.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate SMARTS input files from ERA5 data for EU locations')
    parser.add_argument('--input', help='Path to ERA5 NetCDF file')
    parser.add_argument('--output', help='Output directory for SMARTS input files')
    parser.add_argument('--locations', help='CSV file with location data (optional)')
    parser.add_argument('--elevation-file', help='CSV file with elevation data (optional)')
    parser.add_argument('--metadata', help='Directory to save metadata JSON files')
    parser.add_argument('--days', help='Comma-separated list of days (e.g., 2023-01-15,2023-06-21)')
    parser.add_argument('--times', help='Comma-separated list of time points (morning,noon,evening)')
    parser.add_argument('--limit', type=int, default=300, help='Maximum number of locations to process')
    parser.add_argument('--resume', action='store_true', help='Resume from previous run; skip existing files')
    parser.add_argument('--format', choices=['standard', 'compact', 'detailed'], default='standard',
                        help='Output format for SMARTS input files')
    parser.add_argument('--config', help='Path to YAML configuration file')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    parser.add_argument('--max-workers', type=int, help='Maximum number of worker processes')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--test', action='store_true', help='Run unit tests')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Validate inputs without generating files')
    
    args = parser.parse_args()
   
    # Add these lines after parsing arguments
    args.input = r"C:\Users\gindi002\DATASET\New_Era5_dataset_netcdf\era5_2023_merged.nc"  # Replace with your actual path
    args.output = r"C:\Users\gindi002\DATASET\smarts_inp_files"  # Replace with your actual path
    args.metadata = r"C:\Users\gindi002\DATASET\smarts_inp_files\metadata"  # Replace with your actual path
    # Optional: hardcode other parameters as needed
    args.elevation_file = r"C:\Users\gindi002\DATASET\Nasa_Power\elevation_summary.csv"  # Replace with your actual path
    args.format = "standard"  # Use standard, compact, or detailed
    args.parallel = True  # Enable parallel processing


    # Run unit tests if requested
    if args.test:
        if run_unit_tests():
            logger.info("All unit tests passed!")
            return 0
        else:
            logger.error("Some unit tests failed. Please check the code.")
            return 1
    
    # Check required arguments
    if not args.input and not args.dry_run:
        logger.error("Input file (--input) is required unless in dry-run mode")
        return 1
        
    if not args.output and not args.dry_run:
        logger.error("Output directory (--output) is required unless in dry-run mode")
        return 1
    
    # Apply configuration if specified
    if args.config:
        config = load_config(args.config)
        args = apply_config_to_args(args, config)
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        
    # Dry run mode - just validate inputs and configuration
    if args.dry_run:
        logger.info("Running in dry-run mode - validating configuration only")
        
        # Check if locations file exists
        if args.locations and not os.path.exists(args.locations):
            logger.error(f"Locations file not found: {args.locations}")
            return 1
            
        # Check if elevation file exists
        if args.elevation_file and not os.path.exists(args.elevation_file):
            logger.warning(f"Elevation file not found: {args.elevation_file}")
            
        # Validate days format
        if args.days:
            try:
                test_days = [pd.to_datetime(day.strip()) for day in args.days.split(',')]
                logger.info(f"Validated {len(test_days)} days")
            except Exception as e:
                logger.error(f"Invalid day format: {e}")
                return 1
                
        # Validate times format
        if args.times:
            time_names = [t.strip().lower() for t in args.times.split(',')]
            valid_times = [tp for tp in TIME_POINTS if tp['name'].lower() in time_names]
            if not valid_times:
                logger.warning("No valid time points specified")
                
        logger.info("Dry run completed successfully - configuration is valid")
        return 0
        
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
        
    # Create output directories
    os.makedirs(args.output, exist_ok=True)
    if args.metadata:
        os.makedirs(args.metadata, exist_ok=True)

    # Setup error logging
    error_log_path = os.path.join(args.output, "error_log.csv")
    failed_files = []

    # Track processing statistics
    stats = {
        'total_locations': 0,
        'total_days': 0,
        'total_time_points': 0,
        'successful': 0,
        'errors': 0,
        'skipped': 0,
        'missing_elevation': 0,
        'missing_params': {},
        'invalid_params': {},
        'missing_critical': {},
        'nan_variables': {},
        'skipped_variables': {},
        'extraction_errors': {},
        'total_extraction_errors': 0,
        'time_warnings': 0,
        'validation_messages': [],
        'graceful_degradation': 0,
        'processing_time': 0
    }

    start_time = time.time()
    
    # Load elevation data if provided
    elevation_df = None
    if args.elevation_file:
        elevation_df = load_elevation_data(args.elevation_file)
    
    # Load or generate locations
    if args.locations and os.path.exists(args.locations):
        try:
            locations_df = pd.read_csv(args.locations)
            locations = []
            for _, row in locations_df.iterrows():
                try:
                    name = row['name']
                    lat = float(row['lat'])
                    lon = float(row['lon'])
                    elevation = float(row['elevation']) if 'elevation' in row and not pd.isna(row['elevation']) else None
                    land_cover = row['land_cover'].upper() if 'land_cover' in row and not pd.isna(row['land_cover']) else "LIGHT_SOIL"
                    
                    # If no elevation in locations file, try to get from elevation data
                    if elevation is None and elevation_df is not None:
                        elevation = get_elevation_for_location(elevation_df, lat, lon)
                        
                    locations.append([name, lat, lon, elevation, land_cover])
                except Exception as e:
                    logger.warning(f"Error processing location row: {row} - {e}")
            logger.info(f"Loaded {len(locations)} locations from CSV")
        except Exception as e:
            logger.error(f"Error loading locations from CSV: {e}")
            return 1
    else:
        try:
            locations = generate_eu_grid(args.limit, elevation_df)
            logger.info(f"Generated EU grid with {len(locations)} locations")
        except Exception as e:
            logger.error(f"Error generating EU grid: {e}")
            return 1
            
    if args.limit and args.limit < len(locations):
        locations = locations[:args.limit]
        
    stats['total_locations'] = len(locations)

    # Parse days and time points
    if args.days:
        days = [{"date": day.strip(), "description": day.strip()} for day in args.days.split(',')]
    else:
        days = REPRESENTATIVE_DAYS
    stats['total_days'] = len(days)

    if args.times:
        time_names = [t.strip().lower() for t in args.times.split(',')]
        time_points = [tp for tp in TIME_POINTS if tp['name'].lower() in time_names]
        if not time_points:
            logger.warning("No valid time points specified, using defaults")
            time_points = TIME_POINTS
    else:
        time_points = TIME_POINTS
    stats['total_time_points'] = len(time_points)

    # Calculate total expected files
    stats['total_files'] = stats['total_locations'] * stats['total_days'] * stats['total_time_points']
    logger.info(f"Processing {stats['total_files']} files")

    # Load and process ERA5 data
    try:
        logger.info(f"Loading ERA5 dataset from {args.input}")
        
        # Load dataset and keep it open for processing
        raw_ds = xr.open_dataset(args.input, chunks={'valid_time': 'auto'})
        ds = raw_ds.rename({'valid_time': 'time'})

        if not validate_era5_dataset(ds):
            logger.error("ERA5 dataset validation failed")
            ds.close()  # Clean up
            return 1

        # Filter time range if days are specified
        if args.days:
            day_timestamps = [pd.to_datetime(day["date"]) for day in days]
            min_date = min(day_timestamps) - pd.Timedelta(days=1)
            max_date = max(day_timestamps) + pd.Timedelta(days=1)
            logger.info(f"Filtering dataset to time range: {min_date.date()} to {max_date.date()}")
            ds = ds.sel(time=slice(min_date, max_date))

        # Process locations and times
        success_count, error_count, skipped_count, successful_files, failed_list = batch_process_locations(
            locations, days, time_points, ds, args, stats, elevation_df
        )

        # Update statistics
        stats['successful'] = success_count
        stats['errors'] = error_count
        stats['skipped'] = skipped_count
        failed_files.extend(failed_list)
        
        # Close dataset when done
        ds.close()

    except Exception as e:
        logger.error(f"Error processing ERA5 data: {e}")
        if 'ds' in locals():
            ds.close()
        return 1
    finally:
        stats['processing_time'] = time.time() - start_time
        
    # Write error log if there are failures
    if failed_files:
        try:
            error_df = pd.DataFrame({'error': failed_files})
            error_df.to_csv(error_log_path, index=False)
            logger.warning(f"Error log saved: {error_log_path}")
        except Exception as e:
            logger.error(f"Error saving error log: {e}")

    # Log detailed statistics
    logger.info("=" * 50)
    logger.info("Processing Summary")
    logger.info("=" * 50)
    logger.info(f"Total locations: {stats['total_locations']}")
    logger.info(f"Total days: {stats['total_days']}")
    logger.info(f"Total time points: {stats['total_time_points']}")
    logger.info(f"Total files expected: {stats['total_files']}")
    logger.info(f"Successfully generated: {stats['successful']} ({stats['successful']/stats['total_files']*100:.1f}%)")
    logger.info(f"Errors: {stats['errors']} ({stats['errors']/stats['total_files']*100:.1f}%)")
    logger.info(f"Skipped (resume mode): {stats['skipped']}")
    if stats.get('graceful_degradation', 0) > 0:
        logger.info(f"Files created with graceful degradation: {stats['graceful_degradation']}")
    logger.info(f"Processing time: {stats['processing_time']:.1f} seconds")
    logger.info(f"Average time per file: {stats['processing_time']/max(1, stats['successful'] + stats['errors']):.2f} seconds")
    
    # Log most common problems
    if stats['missing_critical']:
        logger.info("\nMost common missing critical parameters:")
        for param, count in sorted(stats['missing_critical'].items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"  {param}: {count} files")
            
    if stats['extraction_errors']:
        logger.info("\nMost common extraction errors:")
        for var, count in sorted(stats['extraction_errors'].items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"  {var}: {count} errors")
    
    if stats['time_warnings'] > 0:
        logger.warning(f"Time index warnings: {stats['time_warnings']} (requested times not closely matched in ERA5 data)")
        
    # Return status code
    if stats['errors'] > 0:
        return 2
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        traceback.print_exc()
        sys.exit(1)
