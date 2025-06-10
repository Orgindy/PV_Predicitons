#!/usr/bin/env python3
"""
ERA5 Data Processor

A clean, well-structured program for processing ERA5 climate reanalysis data:
- Loads ERA5 NetCDF files
- Calculates derived climate variables
- Performs temporal aggregation (hourly → daily → monthly → yearly)
- Computes climate indices
- Exports processed data to NetCDF files

Usage:
    python era5_processor.py --input INPUT_FILE --output OUTPUT_DIR [options]

Author: [Your Name]
Date: May 2025
"""
import os
import argparse
import logging
import gc
import sys  # Add this import
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
import xarray as xr
import time 

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("era5_processor")


# ======================================================================
# DATA LOADING AND PREPROCESSING
# ======================================================================

def load_dataset(input_file: str, 
                 spatial_subset: Optional[Dict] = None,
                 time_subset: Optional[Dict] = None,
                 chunks: Optional[Dict] = None) -> xr.Dataset:
    """
    Load an ERA5 dataset with optional subsetting and chunking.
    
    Parameters:
    -----------
    input_file : str
        Path to ERA5 NetCDF file
    spatial_subset : dict, optional
        Spatial subset to apply: {'lat_min', 'lat_max', 'lon_min', 'lon_max'}
    time_subset : dict, optional
        Time range to apply: {'start', 'end'}
    chunks : dict, optional
        Chunk sizes for each dimension
        
    Returns:
    --------
    xr.Dataset
        Loaded and subsetted dataset
    """
    try:
        # Load dataset with chunking
        logger.info(f"Loading dataset from {input_file}...")
        ds = xr.open_dataset(input_file, chunks=chunks)
        
        # Fix time dimension if named 'valid_time'
        if 'valid_time' in ds.coords and 'time' not in ds.coords:
            ds = ds.rename({'valid_time': 'time'})
            logger.info("Renamed 'valid_time' to 'time'")
        
        # Apply spatial subset if requested
        if spatial_subset:
            logger.info(f"Applying spatial subset: {spatial_subset}")
            if 'latitude' in ds.dims and 'longitude' in ds.dims:
                lat_min = spatial_subset.get('lat_min', ds.latitude.min().item())
                lat_max = spatial_subset.get('lat_max', ds.latitude.max().item())
                lon_min = spatial_subset.get('lon_min', ds.longitude.min().item())
                lon_max = spatial_subset.get('lon_max', ds.longitude.max().item())
                
                # Handle descending latitudes (ERA5 is usually 90 to -90)
                if ds.latitude[0] > ds.latitude[-1]:
                    ds = ds.sel(latitude=slice(lat_max, lat_min))
                else:
                    ds = ds.sel(latitude=slice(lat_min, lat_max))
                
                ds = ds.sel(longitude=slice(lon_min, lon_max))
                logger.info(f"Spatial subset applied: lat [{lat_min}, {lat_max}], lon [{lon_min}, {lon_max}]")
        
        # Apply time subset if requested
        if time_subset:
            logger.info(f"Applying time subset: {time_subset}")
            if 'time' in ds.dims:
                start_time = time_subset.get('start', None)
                end_time = time_subset.get('end', None)
                
                if start_time and end_time:
                    ds = ds.sel(time=slice(start_time, end_time))
                    logger.info(f"Time subset applied: {start_time} to {end_time}")
                elif start_time:
                    ds = ds.sel(time=slice(start_time, None))
                    logger.info(f"Time subset applied: {start_time} to end")
                elif end_time:
                    ds = ds.sel(time=slice(None, end_time))
                    logger.info(f"Time subset applied: beginning to {end_time}")
        
        logger.info(f"Dataset loaded successfully with dimensions: {dict(ds.dims)}")
        return ds
    
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
        
def detect_time_step(ds: xr.Dataset) -> int:
    """
    Detect the time step in seconds from the dataset.
    
    Parameters:
    -----------
    ds : xr.Dataset
        Input dataset with a time dimension
        
    Returns:
    --------
    int
        Time step in seconds
    """
    if len(ds.time) > 1:
        # Calculate time difference between consecutive points
        time_diffs = np.diff(ds.time.values)
        
        # Convert to seconds
        time_steps_seconds = time_diffs.astype('timedelta64[s]').astype(np.int64)
        
        # Check if time steps are consistent
        if len(time_steps_seconds) > 0 and np.all(time_steps_seconds == time_steps_seconds[0]):
            time_step_seconds = time_steps_seconds[0]
            logger.info(f"Detected uniform time step: {time_step_seconds} seconds")
        else:
            # Use the most common time step if varying
            unique_steps, counts = np.unique(time_steps_seconds, return_counts=True)
            time_step_seconds = unique_steps[np.argmax(counts)]
            logger.warning(f"Variable time steps detected. Using most common: {time_step_seconds} seconds")
    else:
        # Default to hourly if we can't determine
        time_step_seconds = 3600
        logger.warning("Could not determine time step, assuming hourly (3600 seconds)")
    
    return time_step_seconds


# ======================================================================
# VARIABLE PROCESSING
# ======================================================================

def convert_kelvin_to_celsius(da: xr.DataArray) -> xr.DataArray:
    """
    Convert temperature from Kelvin to Celsius.
    
    Parameters:
    -----------
    da : xr.DataArray
        Temperature in Kelvin
        
    Returns:
    --------
    xr.DataArray
        Temperature in Celsius
    """
    # Check if already in Celsius (approximation based on values)
    if da.min() < 100:
        logger.info("Temperature appears to be already in Celsius, no conversion needed")
        result = da.copy()
    else:
        logger.info("Converting temperature from Kelvin to Celsius")
        result = da - 273.15
    
    # Update attributes
    result.attrs['units'] = '°C'
    if 'long_name' in result.attrs:
        result.attrs['long_name'] = result.attrs['long_name'] + ' (°C)'
    
    return result


def calculate_relative_humidity(temp_c: xr.DataArray, dewp_c: xr.DataArray) -> xr.DataArray:
    """
    Calculate relative humidity from temperature and dewpoint.
    
    Parameters:
    -----------
    temp_c : xr.DataArray
        Temperature in Celsius
    dewp_c : xr.DataArray
        Dewpoint temperature in Celsius
        
    Returns:
    --------
    xr.DataArray
        Relative humidity in percentage
    """
    # Ensure dewpoint <= temperature (physical constraint)
    dewp_c = xr.where(dewp_c > temp_c, temp_c, dewp_c)
    
    # Use August-Roche-Magnus approximation
    num = np.exp(17.625 * dewp_c / (243.04 + dewp_c))
    den = np.exp(17.625 * temp_c / (243.04 + temp_c))
    
    rh = 100 * (num / den).clip(min=0, max=100)
    rh.attrs['units'] = '%'
    rh.attrs['long_name'] = 'Relative humidity'
    rh.attrs['standard_name'] = 'relative_humidity'
    
    return rh


def calculate_derived_variables(ds: xr.Dataset, time_step_seconds: int) -> xr.Dataset:
    """
    Calculate derived hourly variables with improved memory management.
    
    Parameters:
    -----------
    ds : xr.Dataset
        Input ERA5 dataset
    time_step_seconds : int
        Time step in seconds for energy to power conversions
        
    Returns:
    --------
    xr.Dataset
        Dataset with derived variables
    """
    logger.info("Calculating derived variables...")
    
    # Initialize output dataset with coordinates only
    ds_derived = xr.Dataset(
        coords={
            'time': ds.time,
            'latitude': ds.latitude,
            'longitude': ds.longitude
        }
    )
    
    # Process basic temperature conversions
    if 't2m' in ds:
        logger.info("Processing temperature...")
        
        # Check if we need to convert from Kelvin
        if hasattr(ds.t2m, 'units') and ds.t2m.units == 'K':
            ds_derived['T2m'] = ds.t2m - 273.15
            logger.info("Converted t2m from Kelvin to Celsius")
        else:
            # Check values to guess if Kelvin
            if ds.t2m.mean() > 100:
                ds_derived['T2m'] = ds.t2m - 273.15
                logger.info("Guessed t2m is in Kelvin based on values, converted to Celsius")
            else:
                ds_derived['T2m'] = ds.t2m
                logger.info("Assuming t2m is already in Celsius")
        
        # Set attributes
        ds_derived.T2m.attrs['units'] = '°C'
        ds_derived.T2m.attrs['long_name'] = '2-meter temperature'
    
    # Process dewpoint if available
    if 'd2m' in ds:
        logger.info("Processing dewpoint...")
        
        # Check if we need to convert from Kelvin
        if hasattr(ds.d2m, 'units') and ds.d2m.units == 'K':
            ds_derived['Td2m'] = ds.d2m - 273.15
        else:
            # Check values to guess if Kelvin
            if ds.d2m.mean() > 100:
                ds_derived['Td2m'] = ds.d2m - 273.15
            else:
                ds_derived['Td2m'] = ds.d2m
        
        # Set attributes
        ds_derived.Td2m.attrs['units'] = '°C'
        ds_derived.Td2m.attrs['long_name'] = '2-meter dewpoint temperature'
    
    # Calculate relative humidity if we have both temperature and dewpoint
    if 'T2m' in ds_derived and 'Td2m' in ds_derived:
        logger.info("Calculating relative humidity...")
        
        # Calculate RH using August-Roche-Magnus approximation
        # Ensure dewpoint <= temperature (physical constraint)
        dewp_c = xr.where(ds_derived.Td2m > ds_derived.T2m, ds_derived.T2m, ds_derived.Td2m)
        
        # Calculate RH
        num = np.exp(17.625 * dewp_c / (243.04 + dewp_c))
        den = np.exp(17.625 * ds_derived.T2m / (243.04 + ds_derived.T2m))
        
        ds_derived['RH'] = 100 * (num / den).clip(min=0, max=100)
        ds_derived.RH.attrs['units'] = '%'
        ds_derived.RH.attrs['long_name'] = 'Relative humidity'
    
    # Calculate wind speed if we have u and v components
    if 'u10' in ds and 'v10' in ds:
        logger.info("Calculating wind speed...")
        
        ds_derived['WS'] = np.sqrt(ds.u10**2 + ds.v10**2)
        ds_derived.WS.attrs['units'] = 'm/s'
        ds_derived.WS.attrs['long_name'] = '10-meter wind speed'
    
    # Convert precipitation from m to mm if needed
    if 'tp' in ds:
        logger.info("Processing precipitation...")
        
        # Check units attribute first
        if hasattr(ds.tp, 'units') and ds.tp.units in ['m', 'metres']:
            ds_derived['Precip'] = ds.tp * 1000
            logger.info("Converted precipitation from m to mm")
        else:
            # Check values to guess if in meters
            if ds.tp.mean() < 0.1:
                ds_derived['Precip'] = ds.tp * 1000
                logger.info("Guessed precipitation is in m based on values, converted to mm")
            else:
                ds_derived['Precip'] = ds.tp
                logger.info("Assuming precipitation is already in mm")
        
        # Ensure non-negative values
        ds_derived['Precip'] = ds_derived['Precip'].clip(min=0)
        
        # Set attributes
        ds_derived.Precip.attrs['units'] = 'mm'
        ds_derived.Precip.attrs['long_name'] = 'Total precipitation'
    
    # Preserve global attributes
    ds_derived.attrs = ds.attrs.copy()
    ds_derived.attrs['processing_history'] = f"Derived variables created at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    logger.info(f"Calculated {len(ds_derived.data_vars)} derived variables")
    return ds_derived

def calculate_radiation_variables(ds: xr.Dataset, time_step_seconds: int) -> xr.Dataset:
    """
    Process radiation variables separately to avoid memory issues.
    
    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing radiation variables
    time_step_seconds : int
        Time step in seconds for energy to power conversions
        
    Returns:
    --------
    xr.Dataset
        Dataset with processed radiation variables
    """
    logger.info("Processing radiation variables...")
    
    # Initialize output dataset with coordinates
    ds_rad = xr.Dataset(
        coords={
            'time': ds.time,
            'latitude': ds.latitude,
            'longitude': ds.longitude
        }
    )
    
    # List of radiation variables to check
    rad_vars = ['ssrd', 'strd', 'ssru', 'stru', 'ssr', 'str', 'fdir', 'tisr']
    
    # Process each radiation variable if present
    for var in rad_vars:
        if var in ds:
            logger.info(f"Processing radiation variable: {var}")
            
            # Check units - if energy (J/m²), convert to power (W/m²)
            is_energy_units = True
            
            # Check units attribute if available
            if hasattr(ds[var], 'units'):
                is_energy_units = ds[var].units in ['J m**-2', 'J/m2', 'J/m^2', 'J/m²']
            else:
                # Guess based on magnitude - if large values, likely energy
                is_energy_units = ds[var].mean() > 1000
            
            # Keep original in energy units
            var_name_energy = f"{var.upper()}_energy"
            ds_rad[var_name_energy] = ds[var]
            ds_rad[var_name_energy].attrs = ds[var].attrs.copy() if hasattr(ds[var], 'attrs') else {}
            ds_rad[var_name_energy].attrs['units'] = 'J/m²'
            
            # Calculate power version
            var_name_power = f"{var.upper()}_power"
            if is_energy_units:
                ds_rad[var_name_power] = ds[var] / time_step_seconds
            else:
                ds_rad[var_name_power] = ds[var]
                # Also update energy version if originally in power
                ds_rad[var_name_energy] = ds[var] * time_step_seconds
            
            ds_rad[var_name_power].attrs = ds[var].attrs.copy() if hasattr(ds[var], 'attrs') else {}
            ds_rad[var_name_power].attrs['units'] = 'W/m²'
    
    # Calculate net radiation components if available
    if 'SSRD_power' in ds_rad and 'STRD_power' in ds_rad:
        logger.info("Calculating net radiation components...")
        
        # Net shortwave radiation (if upward component available)
        if 'SSRU_power' in ds_rad:
            ds_rad['NET_SW'] = ds_rad['SSRD_power'] - ds_rad['SSRU_power']
        else:
            # Estimate using albedo or just use downward component
            ds_rad['NET_SW'] = ds_rad['SSRD_power']
        
        ds_rad['NET_SW'].attrs['units'] = 'W/m²'
        ds_rad['NET_SW'].attrs['long_name'] = 'Net shortwave radiation'
        
        # Net longwave radiation (if upward component available)
        if 'STRU_power' in ds_rad:
            ds_rad['NET_LW'] = ds_rad['STRD_power'] - ds_rad['STRU_power']
        else:
            # Just use net thermal radiation if available
            if 'STR_power' in ds_rad:
                ds_rad['NET_LW'] = ds_rad['STR_power']
        
        if 'NET_LW' in ds_rad:
            ds_rad['NET_LW'].attrs['units'] = 'W/m²'
            ds_rad['NET_LW'].attrs['long_name'] = 'Net longwave radiation'
        
        # Total net radiation
        if 'NET_SW' in ds_rad and 'NET_LW' in ds_rad:
            ds_rad['NET_RAD'] = ds_rad['NET_SW'] + ds_rad['NET_LW']
            ds_rad['NET_RAD'].attrs['units'] = 'W/m²'
            ds_rad['NET_RAD'].attrs['long_name'] = 'Net radiation balance'
    
    logger.info(f"Processed {len(ds_rad.data_vars)} radiation variables")
    return ds_rad

def determine_aggregation_method(var_name: str) -> str:
    """
    Determine the appropriate aggregation method for a variable.
    
    Parameters:
    -----------
    var_name : str
        Variable name
        
    Returns:
    --------
    str
        Aggregation method ('sum' or 'mean')
    """
    # Variables that should be summed
    sum_variables = [
        'Precip', 'GHI_energy', 'SSRD', 'STRD', 'SSR', 'STR',
        'HDD', 'CDD', 'GDD'
    ]
    
    # Check if variable should be summed
    for sum_var in sum_variables:
        if sum_var in var_name:
            return 'sum'
    
    # Default to mean
    return 'mean'


def aggregate_to_daily(ds: xr.Dataset) -> xr.Dataset:
    """
    Aggregate hourly data to daily timescale with improved memory handling
    and progress reporting.
    
    Parameters:
    -----------
    ds : xr.Dataset
        Hourly dataset
        
    Returns:
    --------
    xr.Dataset
        Daily aggregated dataset
    """
    logger.info("Aggregating hourly data to daily...")
    start_time = time.time()
    
    # Create output dataset with coordinates only
    time_range = pd.date_range(
        start=ds.time.values[0].astype('datetime64[D]'),
        end=ds.time.values[-1].astype('datetime64[D]'),
        freq='D'
    )
    
    ds_daily = xr.Dataset(
        coords={
            'time': time_range,
            'latitude': ds.latitude,
            'longitude': ds.longitude
        }
    )
    
    # Group variables by aggregation method to reduce operations
    sum_vars = []
    mean_vars = []
    max_vars = []
    min_vars = []
    
    for var_name in ds.data_vars:
        if determine_aggregation_method(var_name) == 'sum':
            sum_vars.append(var_name)
        elif var_name.lower().endswith('max') or 'maximum' in var_name.lower():
            max_vars.append(var_name)
        elif var_name.lower().endswith('min') or 'minimum' in var_name.lower():
            min_vars.append(var_name)
        else:
            mean_vars.append(var_name)
    
    # Process variables in batches to reduce memory usage
    batch_size = 3  # Process 3 variables at a time
    
    # Setup progress reporting
    try:
        from tqdm.auto import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
    
    # Process sum variables
    logger.info(f"Processing {len(sum_vars)} sum variables in batches...")
    
    if use_tqdm and sum_vars:
        batch_iterator = tqdm(range(0, len(sum_vars), batch_size), desc="Sum variables")
    else:
        batch_iterator = range(0, len(sum_vars), batch_size)
    
    for i in batch_iterator:
        batch = sum_vars[i:i+batch_size]
        if batch:
            logger.debug(f"Processing batch of sum variables: {batch}")
            
            # Only load the variables we need
            batch_ds = ds[batch]
            
            # Perform aggregation
            daily_batch = batch_ds.resample(time='D').sum(skipna=True)
            
            # Add to result dataset
            for var in daily_batch.data_vars:
                ds_daily[var] = daily_batch[var]
                
                # Preserve attributes
                if var in ds:
                    ds_daily[var].attrs = ds[var].attrs.copy()
            
            # Clean up
            del batch_ds, daily_batch
            gc.collect()
    
    # Process mean variables
    logger.info(f"Processing {len(mean_vars)} mean variables in batches...")
    
    if use_tqdm and mean_vars:
        batch_iterator = tqdm(range(0, len(mean_vars), batch_size), desc="Mean variables")
    else:
        batch_iterator = range(0, len(mean_vars), batch_size)
    
    for i in batch_iterator:
        batch = mean_vars[i:i+batch_size]
        if batch:
            logger.debug(f"Processing batch of mean variables: {batch}")
            
            # Only load the variables we need
            batch_ds = ds[batch]
            
            # Perform aggregation
            daily_batch = batch_ds.resample(time='D').mean(skipna=True)
            
            # Add to result dataset
            for var in daily_batch.data_vars:
                ds_daily[var] = daily_batch[var]
                
                # Preserve attributes
                if var in ds:
                    ds_daily[var].attrs = ds[var].attrs.copy()
            
            # Clean up
            del batch_ds, daily_batch
            gc.collect()
    
    # Process max variables
    if max_vars:
        logger.info(f"Processing {len(max_vars)} max variables in batches...")
        
        if use_tqdm:
            batch_iterator = tqdm(range(0, len(max_vars), batch_size), desc="Max variables")
        else:
            batch_iterator = range(0, len(max_vars), batch_size)
        
        for i in batch_iterator:
            batch = max_vars[i:i+batch_size]
            if batch:
                logger.debug(f"Processing batch of max variables: {batch}")
                
                # Only load the variables we need
                batch_ds = ds[batch]
                
                # Perform aggregation
                daily_batch = batch_ds.resample(time='D').max(skipna=True)
                
                # Add to result dataset
                for var in daily_batch.data_vars:
                    ds_daily[var] = daily_batch[var]
                    
                    # Preserve attributes
                    if var in ds:
                        ds_daily[var].attrs = ds[var].attrs.copy()
                
                # Clean up
                del batch_ds, daily_batch
                gc.collect()
    
    # Process min variables
    if min_vars:
        logger.info(f"Processing {len(min_vars)} min variables in batches...")
        
        if use_tqdm:
            batch_iterator = tqdm(range(0, len(min_vars), batch_size), desc="Min variables")
        else:
            batch_iterator = range(0, len(min_vars), batch_size)
        
        for i in batch_iterator:
            batch = min_vars[i:i+batch_size]
            if batch:
                logger.debug(f"Processing batch of min variables: {batch}")
                
                # Only load the variables we need
                batch_ds = ds[batch]
                
                # Perform aggregation
                daily_batch = batch_ds.resample(time='D').min(skipna=True)
                
                # Add to result dataset
                for var in daily_batch.data_vars:
                    ds_daily[var] = daily_batch[var]
                    
                    # Preserve attributes
                    if var in ds:
                        ds_daily[var].attrs = ds[var].attrs.copy()
                
                # Clean up
                del batch_ds, daily_batch
                gc.collect()
    
    # Calculate daily min/max temperature if T2m is available
    if 'T2m' in ds:
        logger.info("Calculating daily min/max temperature...")
        
        # Load temperature data
        t2m = ds['T2m']
        
        # Calculate min/max
        ds_daily['Tmax'] = t2m.resample(time='D').max(skipna=True)
        ds_daily['Tmin'] = t2m.resample(time='D').min(skipna=True)
        
        # Add attributes
        if 'T2m' in ds:
            ds_daily.Tmax.attrs = ds.T2m.attrs.copy()
            ds_daily.Tmax.attrs['long_name'] = 'Daily maximum temperature'
            
            ds_daily.Tmin.attrs = ds.T2m.attrs.copy()
            ds_daily.Tmin.attrs['long_name'] = 'Daily minimum temperature'
        
        # Calculate diurnal temperature range
        ds_daily['DTR'] = ds_daily.Tmax - ds_daily.Tmin
        ds_daily.DTR.attrs['units'] = '°C'
        ds_daily.DTR.attrs['long_name'] = 'Diurnal temperature range'
        
        # Clean up
        del t2m
        gc.collect()
    
    # Preserve global attributes
    ds_daily.attrs = ds.attrs.copy()
    ds_daily.attrs['processing_history'] = ds.attrs.get('processing_history', '') + f"; Daily aggregation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    logger.info(f"Daily aggregation completed in {time.time() - start_time:.2f} seconds")
    return ds_daily

def aggregate_to_monthly(ds_daily: xr.Dataset) -> xr.Dataset:
    """
    Aggregate daily data to monthly timescale.
    
    Parameters:
    -----------
    ds_daily : xr.Dataset
        Daily dataset
        
    Returns:
    --------
    xr.Dataset
        Monthly aggregated dataset
    """
    logger.info("Aggregating daily data to monthly...")
    
    # Create output dataset
    start_year_month = pd.Timestamp(ds_daily.time.values[0]).replace(day=1)
    end_year_month = pd.Timestamp(ds_daily.time.values[-1]).replace(day=1)
    
    ds_monthly = xr.Dataset(
        coords={
            'time': pd.date_range(
                start=start_year_month,
                end=end_year_month,
                freq='MS'  # Month start
            ),
            'latitude': ds_daily.latitude,
            'longitude': ds_daily.longitude
        }
    )
    
    # Process each variable
    for var_name in ds_daily.data_vars:
        # Determine aggregation method
        method = determine_aggregation_method(var_name)
        
        try:
            # Apply aggregation
            if method == 'sum':
                ds_monthly[var_name] = ds_daily[var_name].resample(time='MS').sum(skipna=True)
            else:  # mean
                ds_monthly[var_name] = ds_daily[var_name].resample(time='MS').mean(skipna=True)
            
            # Preserve attributes
            ds_monthly[var_name].attrs = ds_daily[var_name].attrs.copy()
            logger.info(f"Aggregated {var_name} using {method}")
        
        except Exception as e:
            logger.warning(f"Error aggregating {var_name}: {e}")
    
    # Calculate De Martonne Aridity Index if possible
    if 'Precip' in ds_monthly and 'T2m' in ds_monthly:
        # AI = P / (T + 10) where P is in mm and T is in °C
        ds_monthly['ArIndex'] = ds_monthly.Precip / (ds_monthly.T2m + 10)
        ds_monthly.ArIndex.attrs['units'] = 'mm/°C'
        ds_monthly.ArIndex.attrs['long_name'] = 'De Martonne Aridity Index'
        logger.info("Added Aridity Index")
    
    # Preserve global attributes
    ds_monthly.attrs = ds_daily.attrs.copy()
    ds_monthly.attrs['processing_history'] = ds_daily.attrs.get('processing_history', '') + f"; Monthly aggregation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    logger.info("Monthly aggregation complete")
    return ds_monthly


def aggregate_to_yearly(ds_monthly: xr.Dataset) -> xr.Dataset:
    """
    Aggregate monthly data to yearly timescale.
    
    Parameters:
    -----------
    ds_monthly : xr.Dataset
        Monthly dataset
        
    Returns:
    --------
    xr.Dataset
        Yearly aggregated dataset
    """
    logger.info("Aggregating monthly data to yearly...")
    
    # Create output dataset
    start_year = pd.Timestamp(ds_monthly.time.values[0]).year
    end_year = pd.Timestamp(ds_monthly.time.values[-1]).year
    
    ds_yearly = xr.Dataset(
        coords={
            'time': pd.date_range(
                start=f"{start_year}-01-01",
                end=f"{end_year}-01-01",
                freq='YS'  # Year start
            ),
            'latitude': ds_monthly.latitude,
            'longitude': ds_monthly.longitude
        }
    )
    
    # Process each variable
    for var_name in ds_monthly.data_vars:
        # Determine aggregation method
        method = determine_aggregation_method(var_name)
        
        try:
            # Apply aggregation
            if method == 'sum':
                ds_yearly[var_name] = ds_monthly[var_name].resample(time='YS').sum(skipna=True)
            else:  # mean
                ds_yearly[var_name] = ds_monthly[var_name].resample(time='YS').mean(skipna=True)
            
            # Preserve attributes
            ds_yearly[var_name].attrs = ds_monthly[var_name].attrs.copy()
            logger.info(f"Aggregated {var_name} using {method}")
        
        except Exception as e:
            logger.warning(f"Error aggregating {var_name}: {e}")
    
    # Add clearer names for annual variables
    if 'Precip' in ds_yearly:
        ds_yearly['AnnualPrecip'] = ds_yearly.Precip.copy()
        ds_yearly.AnnualPrecip.attrs = ds_yearly.Precip.attrs.copy()
        ds_yearly.AnnualPrecip.attrs['long_name'] = 'Annual Precipitation'
    
    if 'T2m' in ds_yearly:
        ds_yearly['AnnualTemp'] = ds_yearly.T2m.copy()
        ds_yearly.AnnualTemp.attrs = ds_yearly.T2m.attrs.copy()
        ds_yearly.AnnualTemp.attrs['long_name'] = 'Annual Mean Temperature'
    
    # Preserve global attributes
    ds_yearly.attrs = ds_monthly.attrs.copy()
    ds_yearly.attrs['processing_history'] = ds_monthly.attrs.get('processing_history', '') + f"; Yearly aggregation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    logger.info("Yearly aggregation complete")
    return ds_yearly


# ======================================================================
# CLIMATE INDICES
# ======================================================================

def calculate_degree_days(ds_daily: xr.Dataset) -> xr.Dataset:
    """
    Calculate heating, cooling, and growing degree days.
    
    Parameters:
    -----------
    ds_daily : xr.Dataset
        Daily dataset with temperature variables
        
    Returns:
    --------
    xr.Dataset
        Daily dataset with degree days added
    """
    if 'T2m' not in ds_daily:
        logger.warning("Temperature data not available for degree day calculation")
        return ds_daily
    
    logger.info("Calculating degree days...")
    
    # Base temperatures
    base_temp_hdd = 18.0  # °C
    base_temp_cdd = 18.0  # °C
    base_temp_gdd = 10.0  # °C
    
    # Heating Degree Days (HDD)
    ds_daily['HDD'] = xr.where(ds_daily.T2m < base_temp_hdd, 
                               base_temp_hdd - ds_daily.T2m, 
                               0).clip(min=0)
    ds_daily.HDD.attrs['units'] = '°C-day'
    ds_daily.HDD.attrs['long_name'] = 'Heating Degree Days'
    ds_daily.HDD.attrs['base_temperature'] = f'{base_temp_hdd}°C'
    
    # Cooling Degree Days (CDD)
    ds_daily['CDD'] = xr.where(ds_daily.T2m > base_temp_cdd, 
                               ds_daily.T2m - base_temp_cdd, 
                               0).clip(min=0)
    ds_daily.CDD.attrs['units'] = '°C-day'
    ds_daily.CDD.attrs['long_name'] = 'Cooling Degree Days'
    ds_daily.CDD.attrs['base_temperature'] = f'{base_temp_cdd}°C'
    
    # Growing Degree Days (GDD) if min/max temp available
    if 'Tmax' in ds_daily and 'Tmin' in ds_daily:
        # GDD using (Tmax + Tmin)/2 with base temperature
        ds_daily['GDD'] = (((ds_daily.Tmax + ds_daily.Tmin) / 2) - base_temp_gdd).clip(min=0)
        ds_daily.GDD.attrs['units'] = '°C-day'
        ds_daily.GDD.attrs['long_name'] = 'Growing Degree Days'
        ds_daily.GDD.attrs['base_temperature'] = f'{base_temp_gdd}°C'
    
    logger.info("Degree day calculations complete")
    return ds_daily


def calculate_consecutive_dry_days(ds_daily: xr.Dataset) -> xr.Dataset:
    """
    Calculate maximum consecutive dry days for each year using vectorized operations.
    
    Parameters:
    -----------
    ds_daily : xr.Dataset
        Daily dataset with precipitation
        
    Returns:
    --------
    xr.Dataset
        Dataset with consecutive dry days information
    """
    if 'Precip' not in ds_daily:
        logger.warning("Precipitation data not available for consecutive dry days calculation")
        return xr.Dataset()
    
    logger.info("Calculating consecutive dry days...")
    start_time = time.time()
    
    # Create dry day mask (precipitation < 1mm)
    dry_mask = (ds_daily.Precip < 1).astype(int)
    
    # Group by year
    years = ds_daily.time.dt.year
    unique_years = np.unique(years.values)
    
    # Create output dataset
    ds_yearly = xr.Dataset(
        coords={
            'time': pd.date_range(start=f"{unique_years[0]}-01-01", periods=len(unique_years), freq='YS'),
            'latitude': ds_daily.latitude,
            'longitude': ds_daily.longitude
        }
    )
    
    # Define vectorized function to compute max consecutive dry days
    def max_consecutive_dry(x):
        """Calculate maximum consecutive 1s in an array"""
        # Convert to numpy array and handle NaN values
        x_np = np.asarray(x)
        if np.all(np.isnan(x_np)):
            return 0
        
        # Replace NaN with 0 (not a dry day)
        x_np = np.nan_to_num(x_np, nan=0).astype(int)
        
        # Add sentinel values at beginning and end
        x_padded = np.hstack([[0], x_np, [0]])
        
        # Find run starts and ends
        run_starts = np.where(np.diff(x_padded) == 1)[0]
        run_ends = np.where(np.diff(x_padded) == -1)[0]
        
        # Calculate run lengths
        run_lengths = run_ends - run_starts
        
        # Return maximum run length or 0 if no runs
        return np.max(run_lengths) if len(run_lengths) > 0 else 0
    
    try:
        # Process each year with vectorized operations
        logger.info(f"Processing consecutive dry days for {len(unique_years)} years...")
        
        cdd_data = []
        
        # Process years, possibly with progress bar if tqdm is available
        try:
            from tqdm.auto import tqdm
            year_iter = tqdm(unique_years, desc="Processing years")
        except ImportError:
            year_iter = unique_years
        
        for year in year_iter:
            # Get dry days for this year
            year_mask = years == year
            if not np.any(year_mask):
                # No data for this year, use zeros
                cdd_year = xr.zeros_like(ds_daily.isel(time=0).Precip)
            else:
                year_dry_days = dry_mask.sel(time=year_mask)
                
                if year_dry_days.time.size > 0:
                    # Apply vectorized function along time dimension
                    cdd_year = xr.apply_ufunc(
                        max_consecutive_dry,
                        year_dry_days,
                        input_core_dims=[['time']],
                        vectorize=True,
                        output_dtypes=[int]
                    )
                else:
                    # No valid data, use zeros
                    cdd_year = xr.zeros_like(ds_daily.isel(time=0).Precip)
            
            # Add time coordinate for this year
            cdd_year = cdd_year.expand_dims(
                time=[pd.Timestamp(f"{year}-01-01")]
            )
            
            cdd_data.append(cdd_year)
        
        # Combine all years
        if cdd_data:
            ds_yearly['ConsecutiveDryDays'] = xr.concat(cdd_data, dim='time')
            ds_yearly.ConsecutiveDryDays.attrs = {
                'units': 'days',
                'long_name': 'Maximum consecutive dry days',
                'description': 'Maximum number of consecutive days with precipitation < 1mm'
            }
        
        logger.info(f"Consecutive dry days calculation completed in {time.time() - start_time:.2f} seconds")
        return ds_yearly
    
    except Exception as e:
        logger.error(f"Error calculating consecutive dry days: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Return empty dataset on failure
        return xr.Dataset()
    
def calculate_day_night_rc_split(ds, albedo=0.3, emissivity=0.95, ghi_threshold=20):
    """
    Calculate day, night, and total RC power at hourly resolution.

    Parameters:
    - ds: xarray.Dataset with ['GHI', 'T_air']
    - albedo: surface reflectivity
    - emissivity: material emissivity
    - ghi_threshold: W/m² threshold to define daylight

    Returns:
    - ds: Dataset with added P_rc_net, P_rc_day, P_rc_night
    """
    T_air = ds['T_air']
    T_sky = T_air - 20
    T_air_K = T_air + 273.15
    T_sky_K = T_sky + 273.15

    Q_rad = emissivity * 5.67e-8 * (T_air_K**4 - T_sky_K**4)
    Q_solar = (1 - albedo) * ds['GHI']
    P_rc_net = Q_rad - Q_solar

    is_day = ds['GHI'] > ghi_threshold
    is_night = ~is_day

    ds['P_rc_net'] = P_rc_net
    ds['P_rc_day'] = P_rc_net.where(is_day, 0.0)
    ds['P_rc_night'] = P_rc_net.where(is_night, 0.0)
    
    return ds

def aggregate_to_daily_day_and_night(ds):
    ds_daily = ds.resample(time='1D').mean()

    ds_daily = ds_daily.rename({
        'P_rc_net': 'RC_total',
        'P_rc_day': 'RC_day',
        'P_rc_night': 'RC_night'
    })

    return ds_daily

    # ======================================================================
# FILE SAVING
# ======================================================================

def save_dataset(ds: xr.Dataset, output_path: str, frequency: str) -> str:
    """
    Save a dataset to a NetCDF file with optimized settings.
    
    Parameters:
    -----------
    ds : xr.Dataset
        Dataset to save
    output_path : str
        Directory to save the file
    frequency : str
        Frequency identifier ('daily', 'monthly', 'yearly')
        
    Returns:
    --------
    str
        Path to the saved file
    """
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output filename
    filename = f"ERA5_{frequency}_{timestamp}.nc"
    output_file = os.path.join(output_path, filename)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save the dataset with optimized settings
    logger.info(f"Saving {frequency} dataset to {output_file}...")
    
    try:
        # Set up compression for efficient storage
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in ds.data_vars}
        
        # Save with chunking
        ds.to_netcdf(
            output_file,
            encoding=encoding,
            compute=True,  # Force computation now
            format='NETCDF4',
            engine='netcdf4'
        )
        logger.info(f"✓ {frequency.capitalize()} dataset saved successfully")
        
        return output_file
    except Exception as e:
        logger.error(f"Error saving dataset: {e}")
        # Try without compression as fallback
        logger.info("Attempting to save without compression...")
        ds.to_netcdf(output_file)
        return output_file
    
    # ======================================================================
# MAIN PROCESSING FUNCTION
# ======================================================================

def process_era5(
    input_file: str,
    output_dir: str,
    spatial_subset: Optional[Dict] = None,
    time_subset: Optional[Dict] = None,
    chunks: Optional[Dict] = None
) -> Dict[str, str]:
    """
    Process ERA5 data with optimized memory management and timing metrics.
    
    Parameters:
    -----------
    input_file : str
        Path to ERA5 NetCDF file
    output_dir : str
        Directory to save output files
    spatial_subset : dict, optional
        Spatial subset to apply: {'lat_min', 'lat_max', 'lon_min', 'lon_max'}
    time_subset : dict, optional
        Time range to apply: {'start', 'end'}
    chunks : dict, optional
        Chunk sizes for each dimension
        
    Returns:
    --------
    Dict[str, str]
        Dictionary with paths to saved files
    """
    total_start_time = time.time()
    logger.info(f"Processing ERA5 data from {input_file}")
    saved_files = {}
    
    # Default chunking if not specified
    if chunks is None:
        chunks = {'time': 100, 'latitude': 50, 'longitude': 50}
    
    try:
        # STEP 1: Load dataset with chunking
        logger.info("Step 1: Loading dataset...")
        step_start = time.time()
        ds = load_dataset(input_file, spatial_subset, time_subset, chunks=chunks)
        time_step_seconds = detect_time_step(ds)
        logger.info(f"Dataset loaded in {time.time() - step_start:.2f} seconds")
        
        # Print dataset info
        logger.info(f"Dataset dimensions: {dict(ds.dims)}")
        logger.info(f"Dataset variables: {list(ds.data_vars)}")
        
        # STEP 2: Calculate derived variables
        logger.info("Step 2: Calculating derived variables...")
        step_start = time.time()
        
        # Process essential variables first (temperature, humidity, etc.)
        essential_vars = ['t2m', 'd2m', 'tp', 'u10', 'v10']
        present_essential = [var for var in essential_vars if var in ds]
        
        # Load only essential variables to reduce memory usage
        if present_essential:
            logger.info(f"Processing essential variables: {present_essential}")
            ds_essential = ds[present_essential].compute()
            ds_derived = calculate_derived_variables(ds_essential, time_step_seconds)
            
            # Clean up
            del ds_essential
            gc.collect()
        else:
            # Initialize empty dataset with coordinates
            ds_derived = xr.Dataset(
                coords={
                    'time': ds.time,
                    'latitude': ds.latitude,
                    'longitude': ds.longitude
                }
            )
        
        # Process radiation variables separately
        radiation_vars = ['ssrd', 'strd', 'ssru', 'stru', 'ssr', 'str', 'fdir', 'tisr']
        present_radiation = [var for var in radiation_vars if var in ds]
        
        if present_radiation:
            logger.info(f"Processing radiation variables: {present_radiation}")
            ds_rad = ds[present_radiation].compute()
            ds_rad_derived = calculate_radiation_variables(ds_rad, time_step_seconds)
            
            # Merge with main derived dataset
            ds_derived = xr.merge([ds_derived, ds_rad_derived])
            
            # Clean up
            del ds_rad, ds_rad_derived
            gc.collect()
        
        # Free original dataset memory
        del ds
        gc.collect()
        
        logger.info(f"Derived variables calculated in {time.time() - step_start:.2f} seconds")
        
        # STEP 3: Daily aggregation
        logger.info("Step 3: Performing daily aggregation...")
        step_start = time.time()
        
        # Re-chunk for aggregation
        ds_derived = ds_derived.chunk({'time': min(100, len(ds_derived.time))})
        ds_daily = aggregate_to_daily(ds_derived)
        
        # Calculate degree days
        ds_daily = calculate_degree_days(ds_daily)
        
        # Free memory
        del ds_derived
        gc.collect()
        
        logger.info(f"Daily aggregation completed in {time.time() - step_start:.2f} seconds")
        
        # STEP 4: Save daily data before proceeding
        logger.info("Saving daily dataset...")
        save_start = time.time()
        saved_files['daily'] = save_dataset(ds_daily, output_dir, 'daily')
        logger.info(f"Daily dataset saved in {time.time() - save_start:.2f} seconds")
        
        # STEP 5: Monthly aggregation
        logger.info("Step 5: Performing monthly aggregation...")
        step_start = time.time()
        ds_monthly = aggregate_to_monthly(ds_daily)
        logger.info(f"Monthly aggregation completed in {time.time() - step_start:.2f} seconds")
        
        # Save monthly data
        logger.info("Saving monthly dataset...")
        save_start = time.time()
        saved_files['monthly'] = save_dataset(ds_monthly, output_dir, 'monthly')
        logger.info(f"Monthly dataset saved in {time.time() - save_start:.2f} seconds")
        
        # STEP 6: Yearly aggregation
        logger.info("Step 6: Performing yearly aggregation...")
        step_start = time.time()
        ds_yearly = aggregate_to_yearly(ds_monthly)
        
        # Free monthly data
        del ds_monthly
        gc.collect()
        
        logger.info(f"Yearly aggregation completed in {time.time() - step_start:.2f} seconds")
        
        # STEP 7: Calculate additional climate indices
        logger.info("Step 7: Calculating climate indices...")
        step_start = time.time()
        ds_consecutive = calculate_consecutive_dry_days(ds_daily)
        
        # Free daily data
        del ds_daily
        gc.collect()
        
        # Add consecutive dry days to yearly dataset
        if 'ConsecutiveDryDays' in ds_consecutive:
            ds_yearly['ConsecutiveDryDays'] = ds_consecutive.ConsecutiveDryDays
            
        # Clean up
        del ds_consecutive
        gc.collect()
        
        logger.info(f"Climate indices calculated in {time.time() - step_start:.2f} seconds")
        
        # STEP 8: Save yearly dataset
        logger.info("Saving yearly dataset...")
        save_start = time.time()
        saved_files['yearly'] = save_dataset(ds_yearly, output_dir, 'yearly')
        logger.info(f"Yearly dataset saved in {time.time() - save_start:.2f} seconds")
        
        # Final clean-up
        del ds_yearly
        gc.collect()
        
        total_time = time.time() - total_start_time
        logger.info(f"ERA5 processing completed successfully in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        return saved_files
    
    except Exception as e:
        logger.error(f"Error processing ERA5 data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    finally:
        # Final memory cleanup
        for var_name in list(locals().keys()):
            if var_name.startswith('ds_') and locals()[var_name] is not None:
                locals()[var_name] = None
        gc.collect()

# ======================================================================
# COMMAND LINE INTERFACE
# ======================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process ERA5 data: load, calculate derived variables, aggregate, and save.'
    )
    
    parser.add_argument('--input', required=True, help='Path to ERA5 NetCDF file')
    parser.add_argument('--output', required=True, help='Directory to save output files')
    
    parser.add_argument('--lat-min', type=float, help='Minimum latitude')
    parser.add_argument('--lat-max', type=float, help='Maximum latitude')
    parser.add_argument('--lon-min', type=float, help='Minimum longitude')
    parser.add_argument('--lon-max', type=float, help='Maximum longitude')
    
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    return parser.parse_args()


def main():
    """Main entry point with hardcoded file paths."""
    # Record start time
    start_time = time.time()
    
    # HARDCODED PATHS - MODIFY THESE TO YOUR ACTUAL FILE LOCATIONS
    input_file = os.getenv("ERA5_INPUT_FILE", "era5_2023_merged.nc")
    output_dir = os.getenv("ERA5_OUTPUT_DIR", "processed_era5")
    
    # Optional: Hardcode spatial subset if needed
    spatial_subset = None
    # spatial_subset = {
    #     'lat_min': 30.0,
    #     'lat_max': 60.0,
    #     'lon_min': -10.0,
    #     'lon_max': 30.0
    # }
    
    # Optional: Hardcode time subset if needed
    time_subset = None
    # time_subset = {
    #     'start': '2023-01-01',
    #     'end': '2023-12-31'
    # }
    
    # Configure logging
    logger.setLevel(logging.INFO)
    
    # Print system information
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        logger.info(f"System memory: {memory_gb:.1f} GB")
        logger.info(f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
    except ImportError:
        logger.info("psutil not available, skipping system information")
    
    # Determine appropriate chunking based on available memory
    chunks = {'time': 100, 'latitude': 50, 'longitude': 50}
    
    # Process ERA5 data
    try:
        logger.info(f"Starting ERA5 processing: {input_file} -> {output_dir}")
        
        if spatial_subset:
            logger.info(f"Using spatial subset: {spatial_subset}")
        if time_subset:
            logger.info(f"Using time subset: {time_subset}")
            
        logger.info(f"Using chunks: {chunks}")
        
        result = process_era5(
            input_file=input_file,
            output_dir=output_dir,
            spatial_subset=spatial_subset,
            time_subset=time_subset,
            chunks=chunks
        )
        
        logger.info("Processing complete. Output files:")
        for freq, path in result.items():
            logger.info(f"  - {freq}: {path}")
        
        # Report total execution time
        total_time = time.time() - start_time
        logger.info(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        return 0
    
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
