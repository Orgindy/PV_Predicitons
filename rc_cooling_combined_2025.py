import os
import glob
import numpy as np
import pandas as pd
import gc
import logging
from typing import Tuple
from sklearn.model_selection import train_test_split
from pykrige.ok import OrdinaryKriging
from config import get_nc_dir
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import geopandas as gpd
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
from humidity import compute_relative_humidity

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

##############################################################################
#                               CONFIGURATION
##############################################################################
STEFAN_BOLTZMANN = 5.67e-8      # W/m²/K^4
DEFAULT_RHO       = 0.2         # Default solar reflectivity (used if GRIB effective albedo not available)
DEFAULT_EPS_COAT  = 0.95        # Assumed IR emissivity of the coating
CHUNK_SIZE        = 10000       # Number of rows per CSV chunk
KRIGING_SAMPLE_SIZE = 500       # Maximum number of points for training kriging

# Variogram model configuration for kriging
VARIOGRAM_MODEL = 'spherical'   # Options: 'spherical', 'exponential', 'gaussian', etc.

# Europe grid boundaries and resolution (min_lon, max_lon, step)
GRID_LON_RANGE = (-30, 40, 2.0)
# Europe grid boundaries and resolution (min_lat, max_lat, step)
GRID_LAT_RANGE = (35, 70, 2.0)

# Path to your CSV files folder (containing the NASA CSV files with climate data for RC cooling)
DATA_FOLDER = os.getenv("RC_DATA_FOLDER", "qnet")

# (Optional) Path to GRIB data folder; if provided, effective albedo will be extracted from GRIB files.
GRIB_PATH = os.getenv("GRIB_PATH")  # Set to None if not available

##############################################################################
#         GRIB HELPER FUNCTION
##############################################################################

def calculate_sky_temperature_improved(T_air, RH=50, cloud_cover=0):
    """
    Calculate sky temperature using proper atmospheric physics.
    
    Parameters:
    - T_air: Air temperature in °C or array
    - RH: Relative humidity in % (default 50)
    - cloud_cover: Cloud fraction 0-1 (default 0)
    
    Returns:
    - T_sky in °C
    """
    import numpy as np
    
    T_air_K = np.array(T_air) + 273.15
    
    # Swinbank's formula for clear sky emissivity
    eps_clear = 0.741 + 0.0062 * np.array(RH)
    
    # Cloud correction (Duffie & Beckman)
    eps_sky = eps_clear + (1 - eps_clear) * np.array(cloud_cover)
    
    # Clip emissivity to physical range
    eps_sky = np.clip(eps_sky, 0.7, 1.0)
    
    # Sky temperature from Stefan-Boltzmann
    T_sky_K = T_air_K * (eps_sky ** 0.25)
    
    return T_sky_K - 273.15
def get_effective_albedo(grib_path: str, timestamp: pd.Timestamp, lat: float, lon: float) -> float:
    """
    Retrieve effective broadband albedo from GRIB data. Searches for a monthly GRIB file
    (assumed to include the year and month in the filename) and interpolates the 'fal'
    (forecast albedo) variable to the given time and location.
    
    If no file is found or an error occurs, returns DEFAULT_RHO.
    """
    if not grib_path:
        logging.info("GRIB path not provided; using default albedo.")
        return DEFAULT_RHO

    # Construct file search pattern (assumes filenames contain YYYYMM)
    year = timestamp.year
    month = timestamp.month
    pattern = os.path.join(grib_path, f"*{year}{month:02d}*.grib")
    grib_files = glob.glob(pattern)
    if not grib_files:
        logging.warning(f"No GRIB file found for {year}-{month:02d}; using default albedo.")
        return DEFAULT_RHO

    grib_file = grib_files[0]
    try:
        ds = xr.open_dataset(grib_file, engine='cfgrib')
        if 'fal' not in ds:
            logging.warning("GRIB file does not contain 'fal' (forecast albedo); using default albedo.")
            return DEFAULT_RHO
        ds = ds.sortby('time')
        albedo_interp = ds['fal'].interp(time=[timestamp], latitude=[lat], longitude=[lon], method="nearest")
        effective_albedo = float(albedo_interp.values[0])
        logging.info(f"Retrieved effective albedo {effective_albedo:.3f} from GRIB.")
        return effective_albedo
    except Exception as e:
        logging.error(f"Error retrieving albedo from GRIB: {e}")
        return DEFAULT_RHO

##############################################################################
#         QNET CALCULATION FUNCTIONS (Vectorized Implementation)
##############################################################################


def calculate_qnet_vectorized(df: pd.DataFrame,
                              sigma: float = STEFAN_BOLTZMANN,
                              eps_coat: float = DEFAULT_EPS_COAT) -> pd.Series:
    """
    Calculate QNET (net radiative cooling potential) in a vectorized way.
    Uses proper sky temperature calculation.
    
    Assumes T_air is in Celsius (standardized variable name).
    """
    # Determine effective reflectivity from GRIB or use default
    if 'effective_albedo' in df.columns:
        rho = df['effective_albedo'].fillna(DEFAULT_RHO)
    else:
        rho = DEFAULT_RHO

    # Standardize temperature variable name
    if 'T2M' in df.columns:
        T_air = df['T2M']
    elif 'T_air' in df.columns:
        T_air = df['T_air']
    else:
        raise ValueError("No temperature column found (expected 'T_air' or 'T2M')")
    
    T_air_K = T_air + 273.15
    SW = df.get('ALLSKY_SFC_SW_DWN', pd.Series(0.0, index=df.index))
    
    # Use proper sky temperature calculation
    RH = df.get('RH', 50)  # Default RH if not available
    TCC = df.get('TCC', 0)  # Total cloud cover if available
    T_sky = calculate_sky_temperature_improved(T_air, RH, TCC)
    T_sky_K = T_sky + 273.15
    
    # Calculate QNET with proper physics
    Q_rad_out = eps_coat * sigma * (T_air_K ** 4)
    Q_rad_in = eps_coat * sigma * (T_sky_K ** 4)
    Q_solar_abs = (1 - rho) * SW
    
    qnet = Q_rad_out - Q_rad_in - Q_solar_abs
    
    return qnet

##############################################################################
#             2. LOADING AND COMBINING CSV FILES (WITH GRIB INTEGRATION)
##############################################################################
def add_effective_albedo_optimized(chunk, grib_path):
    """Optimized GRIB albedo processing"""
    if not grib_path or not os.path.exists(grib_path):
        chunk['effective_albedo'] = DEFAULT_RHO
        return chunk
    
    # Group by unique time values to reduce GRIB file operations
    unique_times = chunk['time'].dropna().dt.floor('H').unique()  # Round to hourly
    
    if len(unique_times) == 0:
        chunk['effective_albedo'] = DEFAULT_RHO
        return chunk
    
    # Cache albedo values for each time
    albedo_cache = {}
    for time_val in unique_times:
        try:
            # Load GRIB data once per time step
            pattern = os.path.join(grib_path, f"*{time_val.year}{time_val.month:02d}*.grib")
            grib_files = glob.glob(pattern)
            if grib_files:
                with xr.open_dataset(grib_files[0], engine='cfgrib') as ds:
                    if 'fal' in ds:
                        albedo_cache[time_val] = ds['fal'].sel(time=time_val, method='nearest').load()
        except Exception as e:
            logging.warning(f"Could not load GRIB for {time_val}: {e}")
            albedo_cache[time_val] = None
    
    # Apply cached values efficiently
    def get_albedo(row):
        time_key = pd.to_datetime(row['time']).floor('H') if pd.notnull(row['time']) else None
        if time_key in albedo_cache and albedo_cache[time_key] is not None:
            try:
                albedo_data = albedo_cache[time_key]
                return float(albedo_data.interp(latitude=row['LAT'], longitude=row['LON'], method='nearest').values)
            except:
                return DEFAULT_RHO
        return DEFAULT_RHO
    
    chunk['effective_albedo'] = chunk.apply(get_albedo, axis=1)
    return chunk

def load_and_process_csv_files(folder_path: str,
                               chunk_size: int = CHUNK_SIZE,
                               grib_path: str = GRIB_PATH) -> pd.DataFrame:
    """
    Loads all CSV files in `folder_path`, constructs a time column if needed,
    calculates QNET using a vectorized approach, and (if grib_path is provided)
    extracts effective albedo for each row.
    
    Uses chunked reading for memory efficiency and returns a concatenated DataFrame.
    """
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")

    possible_columns = [
        'YEAR', 'MO', 'DY', 'HR', 'LAT', 'LON', 'T2M',
        'ALLSKY_SFC_LW_DWN', 'ALLSKY_SFC_SW_DWN',  # for QNET calculation
        'TCC'  # optional for improved calculation
    ]
    
    df_list = []
    for file in csv_files:
        logging.info(f"Processing CSV file: {file}")
        for chunk in pd.read_csv(file, chunksize=chunk_size):
            use_cols = [col for col in possible_columns if col in chunk.columns]
            chunk = chunk[use_cols].copy()

            # Convert selected columns to numeric
            for col in ['T2M', 'LAT', 'LON', 'ALLSKY_SFC_LW_DWN', 'ALLSKY_SFC_SW_DWN', 'TCC']:
                if col in chunk.columns:
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce', downcast='float')
            
            # Construct time column if not already present
            if 'time' in chunk.columns:
                chunk['time'] = pd.to_datetime(chunk['time'], errors='coerce')
            elif all(c in chunk.columns for c in ['YEAR','MO','DY','HR']):
                try:
                    chunk['time'] = pd.to_datetime({
                        'year':  chunk['YEAR'].astype(int),
                        'month': chunk['MO'].astype(int),
                        'day':   chunk['DY'].astype(int),
                        'hour':  chunk['HR'].astype(int)
                    }, errors='coerce')
                except Exception as ex:
                    logging.error("Error constructing time column from YEAR, MO, DY, HR", exc_info=True)
                    chunk['time'] = pd.NaT
            else:
                chunk['time'] = pd.NaT

            # If GRIB path is provided, add effective_albedo column using row-wise lookup.
            if grib_path:
                chunk = add_effective_albedo_optimized(chunk, grib_path)
            # Calculate QNET for this chunk.
            chunk['QNET'] = calculate_qnet_vectorized(chunk)
            df_list.append(chunk)

    data = pd.concat(df_list, ignore_index=True)
    del df_list
    gc.collect()
    logging.info(f"Total rows loaded: {len(data)}")
    return data

def add_cluster_labels(rc_file, cluster_file, output_file, lat_col='latitude', lon_col='longitude', cluster_col='Cluster_ID'):
    """
    Adds cluster labels to the radiative cooling dataset.
    
    Parameters:
    - rc_file (str): Path to the RC potential CSV file.
    - cluster_file (str): Path to the cluster file with spatial zones.
    - output_file (str): Path to the final output file.
    - lat_col (str): Name of the latitude column.
    - lon_col (str): Name of the longitude column.
    - cluster_col (str): Name of the cluster ID column.
    
    Returns:
    - None
    """
    # Load RC and cluster data
    rc_df = pd.read_csv(rc_file)
    cluster_df = pd.read_csv(cluster_file)
    
    # Convert to GeoDataFrames
    rc_gdf = gpd.GeoDataFrame(
        rc_df, geometry=gpd.points_from_xy(rc_df[lon_col], rc_df[lat_col]), crs="EPSG:4326"
    )
    cluster_gdf = gpd.GeoDataFrame(
        cluster_df, geometry=gpd.points_from_xy(cluster_df[lon_col], cluster_df[lat_col]), crs="EPSG:4326"
    )
    
    # Reproject to match
    rc_gdf = rc_gdf.to_crs(epsg=3857)
    cluster_gdf = cluster_gdf.to_crs(epsg=3857)
    
    # Spatial join to add cluster labels
    rc_gdf_with_labels = gpd.sjoin(rc_gdf, cluster_gdf[[cluster_col, 'geometry']], how='left', predicate='intersects')
    
    # Remove geometry before saving
    final_df = pd.DataFrame(rc_gdf_with_labels.drop(columns='geometry'))
    
    # Save the final dataset
    final_df.to_csv(output_file, index=False)
    print(f"✅ Cluster labels added and saved to {output_file}")

def load_and_merge_rc_netcdf_years(folder_path=None, var_name='QNET', time_dim='time'):
    """
    Load and merge multiple yearly RC NetCDF files into a single xarray.Dataset.

    Parameters:
    - folder_path: directory containing per-year NetCDF files
    - var_name: name of the RC variable (e.g., QNET)
    - time_dim: name of the time dimension (default: 'time')

    Returns:
    - ds_merged: combined xarray.Dataset with stacked time
    """
    import glob
    if folder_path is None:
        folder_path = get_nc_dir()
    elif not os.path.isabs(folder_path):
        folder_path = os.path.join(get_nc_dir(), folder_path)

    files = sorted(glob.glob(os.path.join(folder_path, "*.nc")))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {folder_path}")

    datasets = []
    for file in files:
        print(f"📥 Loading {file}")
        with xr.open_dataset(file) as ds:
            if var_name not in ds:
                raise ValueError(f"Variable '{var_name}' not found in {file}")
            datasets.append(ds.load())  # Load data into memory, then close file

    ds_merged = xr.concat(datasets, dim=time_dim)
    print(f"✅ Merged {len(datasets)} NetCDF files into one dataset")
    return ds_merged

def multi_year_kriging(rc_folder=None, output_file=None, cluster_file=None, lat_col='latitude', lon_col='longitude'):
    """
    Performs multi-year kriging to generate high-resolution RC maps.
    
    Parameters:
    - rc_folder (str or None): Directory containing yearly RC NetCDF files. If
      None, uses ``get_nc_dir()``.
    - output_file (str or None): Path to the final merged output file.
    - cluster_file (str): Optional path to cluster file for spatial weighting.
    - lat_col (str): Name of the latitude column.
    - lon_col (str): Name of the longitude column.
    
    Returns:
    - None
    """
    # Load all RC data
    ds = load_and_merge_rc_netcdf_years(rc_folder, var_name='RC_total')
    df = ds.to_dataframe().reset_index()
    
    # If cluster file is provided, add cluster labels
    if cluster_file:
         cluster_df = pd.read_csv(cluster_file)
         df = pd.merge(df, cluster_df[[lat_col, lon_col, 'Cluster_ID']],
                  on=[lat_col, lon_col], how='left')
    
    # Perform Kriging for each cluster
    if 'Cluster_ID' in df.columns:
        kriged_data = []
        for cluster_id in df['Cluster_ID'].unique():
            cluster_df = df[df['Cluster_ID'] == cluster_id]
            lats = cluster_df[lat_col].values
            lons = cluster_df[lon_col].values
            rc_values = cluster_df['RC_Potential'].values
            
            # Standardize coordinates
            coords = np.vstack((lats, lons)).T
            scaler = StandardScaler()
            scaled_coords = scaler.fit_transform(coords)
            
            # Kriging
            try:
                OK = OrdinaryKriging(
                    scaled_coords[:, 0], scaled_coords[:, 1], rc_values,
                    variogram_model='spherical', verbose=True, enable_plotting=False
                )
                
                # Predict on the same points (could be a grid for higher resolution)
                z, ss = OK.execute('points', scaled_coords[:, 0], scaled_coords[:, 1])
                
                # Save results
                cluster_df['RC_Kriged'] = z
                kriged_data.append(cluster_df)
                print(f"✅ Kriging completed for Cluster {cluster_id}")
            
            except Exception as e:
                print(f"❌ Kriging failed for Cluster {cluster_id}: {e}")
        
        # Combine all kriged data
        final_df = pd.concat(kriged_data, ignore_index=True)
    else:
        print("⚠️ No Cluster_ID found. Performing global kriging.")
        
        # Global kriging
        lats = df[lat_col].values
        lons = df[lon_col].values
        rc_values = df['RC_Potential'].values
        
        coords = np.vstack((lats, lons)).T
        scaler = StandardScaler()
        scaled_coords = scaler.fit_transform(coords)
        
        OK = OrdinaryKriging(
            scaled_coords[:, 0], scaled_coords[:, 1], rc_values,
            variogram_model='spherical', verbose=True, enable_plotting=False
        )
        
        z, ss = OK.execute('points', scaled_coords[:, 0], scaled_coords[:, 1])
        df['RC_Kriged'] = z
        final_df = df
    
    # Save final kriged data once
    final_df.to_csv(output_file, index=False)
    print(f"✅ Multi-year kriged data saved to {output_file}")

def aggregate_rc_metrics(kriged_file, output_file, cluster_col='Cluster_ID'):
    """
    Aggregates multi-year RC metrics for each cluster.
    
    Parameters:
    - kriged_file (str): Path to the kriged RC potential file.
    - output_file (str): Path to the final aggregated metrics file.
    - cluster_col (str): Name of the cluster ID column (default 'Cluster_ID').
    
    Returns:
    - None
    """
    # Load kriged data
    df = pd.read_csv(kriged_file)
    
    if cluster_col not in df.columns:
        print(f"❌ Cluster column '{cluster_col}' not found in the file.")
        return
    
    # Aggregate metrics for each cluster
    metrics_df = df.groupby(cluster_col).agg({
        'RC_Kriged': ['mean', 'median', 'std', 'min', 'max', 'sum', 'count']
    }).reset_index()
    
    # Flatten multi-level columns
    metrics_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in metrics_df.columns]
    
    # Add cluster ID as a standalone column
    metrics_df.columns = [
        'Cluster_ID',
        'RC_mean',
        'RC_median',
        'RC_std',
        'RC_min',
        'RC_max',
        'RC_sum',
        'RC_count',
    ]
    
    # Save the aggregated metrics
    metrics_df.to_csv(output_file, index=False)
    print(f"✅ Aggregated RC metrics saved to {output_file}")


##############################################################################
#             3. VALIDATION & PREPROCESSING (ANNUAL AGGREGATION)
##############################################################################

def validate_data(df: pd.DataFrame) -> None:
    """Ensure that required columns exist and that LAT/LON are within plausible ranges."""
    required = ['LAT', 'LON', 'QNET', 'time']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    if not df['LAT'].between(GRID_LAT_RANGE[0], GRID_LAT_RANGE[1]).all():
        logging.warning("Some latitude values are outside the normal EU range.")
    if not df['LON'].between(GRID_LON_RANGE[0], GRID_LON_RANGE[1]).all():
        logging.warning("Some longitude values are outside the normal EU range.")

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert QNET to numeric, parse the time column, round location values,
    and group by (LAT, LON, year) to compute the annual mean and total QNET.
    Total QNET is converted from Wh to kWh/m²·year.
    """
    validate_data(df)
    df['QNET'] = pd.to_numeric(df['QNET'], errors='coerce')
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    
    df['year'] = df['time'].dt.year
    df['LAT'] = df['LAT'].round(2)
    df['LON'] = df['LON'].round(2)
    
    pos_only = df[df['QNET'] > 0].copy()
    
    annual_stats = (
        pos_only
        .groupby(['LAT', 'LON', 'year'])
        .agg({
            'QNET': [
                ('annual_mean', 'mean'),
                ('annual_total', lambda x: x.sum() / 1000.0)  # converting Wh to kWh
            ]
        })
        .reset_index()
    )
    annual_stats.columns = ['LAT', 'LON', 'year', 'QNET_annual_mean', 'QNET_annual_total']
    return annual_stats

# def compute_relative_humidity(T_air_K, T_dew_K):
#     """Compute relative humidity (%) using the Magnus formula."""
#     # Parameters:
#     #     T_air_K (float or np.ndarray or pd.Series): Air temperature in Kelvin
#     #     T_dew_K (float or np.ndarray or pd.Series): Dew point temperature in Kelvin
#     # Returns:
#     #     Relative humidity in percentage (0–100%)
#     # Example implementation (superseded by humidity.compute_relative_humidity):
#     # T_air = np.array(T_air_K) - 273.15
#     # T_dew = np.array(T_dew_K) - 273.15
#     # a = 17.625
#     # b = 243.04
#     # e_s = np.exp((a * T_air) / (b + T_air))
#     # e_d = np.exp((a * T_dew) / (b + T_dew))
#     # RH = 100.0 * (e_d / e_s)
#     # return np.clip(RH, 0, 100)

def calculate_rc_with_albedo(df, albedo_values=[0.3, 0.6, 1.0]):
    results = []
    for alb in albedo_values:
        df_copy = df.copy()
        df_copy['Albedo'] = alb
        df_copy = calculate_day_night_rc_power(df_copy, albedo=alb)
        results.append(df_copy)
    return pd.concat(results)

def estimate_sky_temperature_hybrid(df):
    """
    Estimate T_sky using RH and TCC (cloud cover) with improved sky emissivity.

    Requires:
    - T_air in Celsius
    - Dew_Point in Celsius
    - TCC (cloud fraction 0–1)
    - ALLSKY_SFC_LW_DWN in W/m²

    Returns:
    - T_sky in Celsius
    """
    
    
    required_cols = ['T2M', 'Dew_Point', 'TCC', 'ALLSKY_SFC_LW_DWN']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    T_air_K = df['T2M'] + 273.15
    T_dew_K = df['Dew_Point'] + 273.15
    RH = compute_relative_humidity(T_air_K, T_dew_K)  # Now imported from humidity.py
    logging.info("Using hybrid RH + cloud model for T_sky")

    # Empirical emissivity model
    e_sky = 0.6 + 0.2 * df['TCC'] + 0.002 * RH
    e_sky = np.clip(e_sky, 0.7, 1.0)

    sigma = 5.670374419e-8
    IR_down = df['ALLSKY_SFC_LW_DWN']
    T_sky_K = (IR_down / (e_sky * sigma)) ** 0.25
    return T_sky_K - 273.15  # Return in Celsius

def calculate_day_night_rc_power(df, albedo=0.3, emissivity=0.95, zenith_col='solar_zenith'):
    """
    Calculate total, daytime, and nighttime RC power with albedo adjustment.

    Parameters:
    - df: DataFrame with GHI, T_air, solar_zenith
    - albedo: solar reflectivity (0–1)
    - emissivity: material emissivity (0–1)

    Returns:
    - df_out: DataFrame with columns:
        - P_rc_net (total)
        - P_rc_day
        - P_rc_night
    """
    from scipy.constants import sigma as σ
    df_out = df.copy()

    # Ensure required columns exist
    assert all(col in df for col in ['T_air', 'GHI', zenith_col])

    # Estimate sky temperature
    try:
        T_sky = estimate_sky_temperature_hybrid(df)
    except Exception as e:
        logging.warning(f"Hybrid sky temperature estimate failed: {e}. Falling back to default.")
        T_sky = df['T_air'] - 20  # Fallback

    # Stefan-Boltzmann emission term
    Q_rad = emissivity * σ * ((df['T_air'] + 273.15) ** 4 - (T_sky + 273.15) ** 4)

    # Absorbed solar energy
    Q_solar = (1 - albedo) * df['GHI']

    # Total net RC
    df_out['P_rc_net'] = Q_rad - Q_solar

    # Identify night/day
    df_out['is_night'] = df[zenith_col] > 90

    # Split day/night
    df_out['P_rc_day'] = df_out['P_rc_net'].where(~df_out['is_night'], 0)
    df_out['P_rc_night'] = df_out['P_rc_net'].where(df_out['is_night'], 0)

    return df_out

##############################################################################
#       4. SPLIT DATA, TRAIN ORDINARY KRIGING, INTERPOLATE & EVALUATION
##############################################################################
def split_data(annual_df: pd.DataFrame) -> Tuple:
    """Split annual data into training and testing sets using location (LON, LAT)."""
    if len(annual_df) < 10:
        raise ValueError("Insufficient data points for train/test split.")
    X = annual_df[['LON', 'LAT']]
    y = annual_df['QNET_annual_total']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_kriging(X_train: pd.DataFrame, y_train: pd.Series) -> OrdinaryKriging:
    """Train an Ordinary Kriging model using a random subset of the training data."""
    if len(X_train) < 3:
        raise ValueError("Insufficient unique points for Kriging.")
    
    sample_size = min(len(X_train), KRIGING_SAMPLE_SIZE)
    sampled_indices = np.random.choice(len(X_train), sample_size, replace=False)
    
    OK = OrdinaryKriging(
        x=X_train.iloc[sampled_indices]['LON'].values.astype(float),
        y=X_train.iloc[sampled_indices]['LAT'].values.astype(float),
        z=y_train.iloc[sampled_indices].values.astype(float),
        variogram_model=VARIOGRAM_MODEL,
        verbose=False,
        enable_plotting=False
    )
    return OK

def evaluate_model(OK: OrdinaryKriging, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Evaluate the kriging model on the test set and log the RMSE."""
    pred, _ = OK.execute('points', X_test['LON'].values.astype(float), X_test['LAT'].values.astype(float))
    pred = np.array(pred)
    y_true = y_test.values
    rmse = np.sqrt(np.mean((pred - y_true) ** 2))
    logging.info(f"Test RMSE: {rmse:.4f} kWh/m²·year")

def interpolate_grid(OK: OrdinaryKriging) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate QNET on a predefined lat/lon grid across Europe.
    Returns 2D arrays for grid longitudes, latitudes, and predicted QNET.
    """
    grid_lon = np.arange(*GRID_LON_RANGE).astype(float)
    grid_lat = np.arange(*GRID_LAT_RANGE).astype(float)
    lon2d, lat2d = np.meshgrid(grid_lon, grid_lat)
    
    z_pred = np.full(lon2d.shape, np.nan, dtype=np.float32)
    chunk_size = 10
    num_rows = lon2d.shape[0]
    
    for i in range(0, num_rows, chunk_size):
        chunk_end = min(i + chunk_size, num_rows)
        lon_chunk = lon2d[i:chunk_end, :].ravel()
        lat_chunk = lat2d[i:chunk_end, :].ravel()
        
        z_chunk, _ = OK.execute(style='points', xpoints=lon_chunk, ypoints=lat_chunk)
        z_pred[i:chunk_end, :] = z_chunk.reshape(chunk_end - i, lon2d.shape[1])
    
    return lon2d, lat2d, z_pred

##############################################################################
#                       5. PLOTTING THE RESULT
##############################################################################

def plot_rc_split_maps(df, lat_col='LAT', lon_col='LON',
                       rc_day_col='RC_day', rc_night_col='RC_night',
                       save_day='rc_day_map.png', save_night='rc_night_map.png'):
    """
    Plot separate RC_day and RC_night maps using Cartopy.
    """

    def plot_map(data_col, title, save_path):
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())

        sc = ax.scatter(df[lon_col], df[lat_col], c=df[data_col],
                        cmap='coolwarm', s=30, edgecolor='k',
                        transform=ccrs.PlateCarree())

        ax.coastlines(resolution='50m')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.set_extent([GRID_LON_RANGE[0], GRID_LON_RANGE[1],
                       GRID_LAT_RANGE[0], GRID_LAT_RANGE[1]], crs=ccrs.PlateCarree())

        cbar = plt.colorbar(sc, ax=ax, orientation='vertical')
        cbar.set_label(f"{data_col} (W/m²)")

        ax.set_title(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"✅ Saved {title} to {save_path}")
        plt.close()

    plot_map(rc_day_col, "Daytime RC Potential", save_day)
    plot_map(rc_night_col, "Nighttime RC Potential", save_night)


def plot_qnet_map(grid_lon: np.ndarray, 
                  grid_lat: np.ndarray, 
                  z_pred: np.ndarray,
                  save_path: str = None) -> None:
    """
    Plot the kriged annual radiative cooling potential on a map.
    If save_path is provided, the plot is saved to that file.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    cs = ax.pcolormesh(grid_lon, grid_lat, z_pred, cmap='YlOrRd',
                       shading='auto', transform=ccrs.PlateCarree())
    cbar = plt.colorbar(cs, orientation='vertical', pad=0.02)
    cbar.set_label("Annual RC Potential (kWh/m²·year)")
    
    ax.coastlines(resolution='50m', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_extent([GRID_LON_RANGE[0], GRID_LON_RANGE[1],
                   GRID_LAT_RANGE[0], GRID_LAT_RANGE[1]], crs=ccrs.PlateCarree())
    
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title("Annual Radiative Cooling Potential (kWh/m²·year)",
                 fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Map saved to {save_path}")
    
    plt.show()

##############################################################################
#                             6. MAIN EXECUTION
##############################################################################
def main(db_url=None, db_table="rc_data"):
    try:
        if db_url:
            from database_utils import read_table, write_dataframe
            logging.info(f"Loading data from table {db_table}")
            data = read_table(db_table, db_url=db_url)
        else:
            logging.info("Loading and processing CSV data for RC cooling potential...")
            data = load_and_process_csv_files(DATA_FOLDER, grib_path=GRIB_PATH)
        
        logging.info("Preprocessing annual data (aggregating by LAT, LON, year)...")
        annual_df = preprocess_data(data)
        
        # Optionally, filter to a single year (e.g., annual_df = annual_df[annual_df['year'] == 2019])
        
        logging.info("Splitting data into train and test sets for Kriging...")
        X_train, X_test, y_train, y_test = split_data(annual_df)
        
        logging.info("Training Ordinary Kriging model...")
        OK = train_kriging(X_train, y_train)
        
        logging.info("Evaluating Kriging model on test data...")
        evaluate_model(OK, X_test, y_test)
        
        logging.info("Interpolating QNET on grid...")
        grid_lon, grid_lat, z_pred = interpolate_grid(OK)
        
        logging.info("Plotting final RC potential map...")
        # Optionally specify a save path for the map, e.g., "rc_potential_map.png"
        plot_qnet_map(grid_lon, grid_lat, z_pred, save_path=None)
        
        logging.info("RC cooling model execution completed successfully.")
        if db_url:
            write_dataframe(annual_df, db_table, db_url=db_url, if_exists="replace")
        
    except Exception as e:
        logging.error("An error occurred during execution.", exc_info=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RC cooling model")
    parser.add_argument("--db-url", default=os.getenv("PV_DB_URL"))
    parser.add_argument("--db-table", default="rc_data")
    args = parser.parse_args()
    main(db_url=args.db_url, db_table=args.db_table)

