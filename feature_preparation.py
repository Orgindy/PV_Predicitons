import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import mutual_info_regression
import os
import argparse
from config import get_nc_dir, get_path
from utils.feature_utils import filter_valid_columns, compute_band_ratios


def parse_args():
    """Parse command line arguments for file paths."""
    parser = argparse.ArgumentParser(description="Prepare features for PV model")
    parser.add_argument(
        "--input-file",
        default=get_path("merged_data_path"),
        help="Path to merged dataset CSV",
    )
    parser.add_argument(
        "--validated-file",
        default=os.path.join(get_path("results_path"), "validated_dataset.csv"),
        help="Path to save validated CSV",
    )
    parser.add_argument(
        "--physics-file",
        default=os.path.join(get_path("results_path"), "physics_dataset.csv"),
        help="Path to save dataset with physics-based PV potential",
    )
    parser.add_argument(
        "--netcdf-file",
        default=os.path.join(get_nc_dir(), "ERA5_daily.nc"),
        help="Path to processed ERA5 NetCDF file",
    )
    parser.add_argument(
        "--results-dir",
        default=get_path("results_path"),
        help="Directory where results will be written",
    )
    parser.add_argument(
        "--db-url",
        default=os.getenv("PV_DB_URL"),
        help="Optional database URL for reading/writing data",
    )
    parser.add_argument(
        "--db-table",
        default=os.getenv("PV_DB_TABLE", "pv_data"),
        help="Database table name if --db-url is provided",
    )
    return parser.parse_args()

# --------------------------------------
# 1. Physics-based PV potential function
# --------------------------------------
from pv_potential import calculate_pv_potential


def map_netcdf_variables(df):
    """Map NetCDF variable names to standardized column names."""
    variable_mapping = {
        'T2m': 'T_air',
        'SSRD_power': 'GHI',
        'NET_RAD': 'RC_potential',
        'Red_band': 'Red_band',
        'Total_band': 'Total_band',
    }
    for src, dest in variable_mapping.items():
        if src in df.columns:
            df[dest] = df[src]


def has_required_columns(df):
    """Return True if DataFrame has all columns needed for PV calculation."""
    required = ['GHI', 'T_air', 'RC_potential', 'Red_band', 'Total_band']
    return all(col in df.columns for col in required)


def validate_parameters(input_file, output_file, drop_invalid=True):
    """
    Validates climate, RC, and spectral parameters to ensure physical accuracy.
    
    Parameters:
    - input_file (str): Path to the input CSV file.
    - output_file (str): Path to the cleaned output file.
    - drop_invalid (bool): Whether to drop rows with invalid values (default True).
    
    Returns:
    - None
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)
    
    # Define valid ranges (based on physical limits)
    valid_ranges = {
        "GHI": (0, 1361),        # Max solar constant at TOA
        "T_air": (-90, 60),      # Realistic surface temperature range (°C)
        "RC_potential": (0, 300),  # Reasonable cooling range (W/m²)
        "Wind_Speed": (0, 150),  # Typical wind speeds (m/s)
        "Dew_Point": (-90, 60),  # Realistic dew point temperature (°C)
        "Blue_band": (0, 1500),  # Reasonable range for spectral bands (W/m²)
        "Green_band": (0, 1500),
        "Red_band": (0, 1500),
        "NIR_band": (0, 1500),
        "IR_band": (0, 1500),
        "Total_band": (0, 5000)  # Full spectral range (W/m²)
    }
    
    # Check each parameter
    invalid_rows = []
    for col, (min_val, max_val) in valid_ranges.items():
        if col in df.columns:
            invalid_rows += df[(df[col] < min_val) | (df[col] > max_val)].index.tolist()
    
    # Drop or flag invalid rows
    if drop_invalid:
        df.drop(index=invalid_rows, inplace=True)
        df.reset_index(drop=True, inplace=True)
        logging.info(f"✅ Dropped {len(invalid_rows)} rows with invalid values.")
    else:
        df["Invalid_Row"] = 0
        df.loc[invalid_rows, "Invalid_Row"] = 1
        logging.warning(f"⚠️ Flagged {len(invalid_rows)} rows as invalid.")
    
    # Save the cleaned file
    df.to_csv(output_file, index=False)
    logging.info(f"✅ Validated data saved to {output_file}")

def auto_generate_physics_pv(input_file, output_file, temp_coeff=-0.0045):
    """
    Automatically generates physics-based PV potential for each row.
    
    Parameters:
    - input_file (str): Path to the input CSV file.
    - output_file (str): Path to the output file with PV potential.
    - temp_coeff (float): Temperature coefficient for PV efficiency (default -0.0045).
    
    Returns:
    - None
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)
    
    # Check for required columns
    required_columns = ["GHI", "T_air", "RC_potential", "Total_band", "Red_band"]
    if not all(col in df.columns for col in required_columns):
        logging.error(f"❌ Missing required columns: {required_columns}")
        return
    
    # Calculate PV potential using centralized implementation
    df['PV_Potential_physics'] = calculate_pv_potential(
        df['GHI'].values,
        df['T_air'].values,
        df['RC_potential'].values,
        df['Red_band'].values,
        df['Total_band'].values,
    )
    
    # Save the updated file
    df.to_csv(output_file, index=False)
    logging.info(f"✅ Physics-based PV potential added and saved to {output_file}")


def compute_feature_weights(df, target_col='PV_Potential_physics'):
    """
    Computes feature weights using mutual information.
    
    Parameters:
    - df (pd.DataFrame): Input data with all features.
    - target_col (str): The target variable for weighting.
    
    Returns:
    - weights (dict): Feature weights.
    """
    # Select only numerical features for weighting
    features = df.drop(columns=[target_col]).select_dtypes(include=['float64', 'int64'])
    target = df[target_col]
    
    # Compute mutual information
    mi_scores = mutual_info_regression(features, target)
    
    # Normalize scores
    weights = {feature: score for feature, score in zip(features.columns, mi_scores)}
    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}
    
    logging.info("\n=== Feature Weights (Normalized) ===")
    for k, v in normalized_weights.items():
        logging.info(f"{k}: {v:.4f}")
    
    return normalized_weights

# --------------------------------------
# NEW: NetCDF Processing Functions
# --------------------------------------
def load_netcdf_data(netcdf_file, sample_fraction=0.1):
    """
    Load NetCDF data and convert to format suitable for ML.
    
    Parameters:
    - netcdf_file: Path to NetCDF file
    - sample_fraction: Fraction of data to sample (to manage memory)
    
    Returns:
    - DataFrame with flattened spatial-temporal data
    """
    logging.info(f"Loading NetCDF data from {netcdf_file}...")
    
    # Load dataset
    with xr.open_dataset(netcdf_file) as ds:
        # Convert to DataFrame (this flattens all dimensions)
        df = ds.load().to_dataframe().reset_index()
    
    # Remove rows with all NaN values
    df = df.dropna(how='all')
    
    # Sample data if too large
    if sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=42)
        logging.info(f"Sampled {len(df)} rows ({sample_fraction*100}% of data)")
    
    logging.info(f"Loaded {len(df)} data points")
    logging.info(f"Available columns: {list(df.columns)}")
    
    return df

def calculate_pv_potential_netcdf(df):
    """Calculate PV potential directly from NetCDF data."""
    logging.info("Calculating PV potential from NetCDF data...")

    # Map variables from NetCDF-specific names
    map_netcdf_variables(df)

    # Use centralized implementation when possible
    if has_required_columns(df):
        df['PV_Potential'] = calculate_pv_potential(
            df['GHI'].values,
            df['T_air'].values,
            df['RC_potential'].values,
            df['Red_band'].values,
            df['Total_band'].values,
        )
        logging.info("✅ PV potential calculated")
    else:
        missing = [c for c in ['GHI', 'T_air', 'RC_potential', 'Red_band', 'Total_band'] if c not in df.columns]
        logging.error("❌ Missing required variables for PV calculation")
        logging.error(f"Missing: {missing}")

    return df

def main_csv_workflow(input_file, validated_file, physics_file, results_dir):
    """CSV-based workflow."""
    import joblib
    import os

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
    
    # Load initial data
    df = pd.read_csv(input_file)
    logging.info(f"Loaded {len(df)} rows from {input_file}")

    # --- Data Validation ---
    # Save to temp file, validate, then reload
    temp_input = "temp_input.csv"
    df.to_csv(temp_input, index=False)
    validate_parameters(temp_input, validated_file, drop_invalid=True)
    df = pd.read_csv(validated_file)
    logging.info(f"After validation: {len(df)} rows")

    # --- Add physics-based PV potential ---
    auto_generate_physics_pv(validated_file, physics_file)
    df = pd.read_csv(physics_file)
    logging.info("Added physics-based PV potential")
    df, _ = compute_band_ratios(
        df,
        ['Blue_band', 'Green_band', 'Red_band', 'IR_band'],
        total_col='Total_band'
    )

    # Save final processed dataset
    final_output = os.path.join(results_dir, "merged_with_physics_pv.csv")
    df.to_csv(final_output, index=False)
    logging.info(f"✅ Final dataset saved to {final_output}")

    # --- Feature Selection ---
    features = [
        'GHI', 'T_air', 'RH', 'Wind_Speed',
        'Albedo', 'Cloud_Cover', 'Dew_Point',
        'Red_band', 'RC_total', 'RC_day', 'RC_night'
    ]

    # Check which features are actually available
    available_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]

    if missing_features:
        logging.warning(f"⚠️ Missing features: {missing_features}")

    logging.info(f"Using features: {available_features}")
    
    # Use only available features
    X = filter_valid_columns(df, available_features)
    missing_count = X.isnull().sum().sum()
    if missing_count > 0:
        logging.info(f"Filling {missing_count} missing values with median")
        X = X.fillna(X.median())
    y = df['PV_Potential_physics']

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Model ---
    model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
    model.fit(X_train, y_train)

    # --- Evaluation ---
    y_pred = model.predict(X_test)
    logging.info("\n=== Model Performance ===")
    logging.info(f"R²: {r2_score(y_test, y_pred):.4f}")
    logging.info(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
    logging.info(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")

    # --- Save model ---
    model_path = os.path.join(results_dir, "models", "rf_pv_model.joblib")
    joblib.dump(model, model_path)
    logging.info(f"💾 Random Forest model saved to: {model_path}")

    # --- Feature Importance Plot ---
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': available_features, 
        'Importance': importances
    })
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "feature_importance_plot.png"), dpi=300)
    plt.close()
    logging.info(f"📊 Feature importance plot saved to {os.path.join(results_dir, 'feature_importance_plot.png')}")

    # Clean up temporary files
    for temp_file in [temp_input]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    logging.info("\n✅ CSV-based feature preparation completed successfully!")

def main_netcdf_workflow(netcdf_file, results_dir):
    """NetCDF-based workflow."""
    import joblib

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
    
    # Load NetCDF data
    df = load_netcdf_data(netcdf_file, sample_fraction=0.1)  # Use 10% sample for speed
    
    # Calculate PV potential
    df = calculate_pv_potential_netcdf(df)
    df, _ = compute_band_ratios(
        df,
        ['Blue_band', 'Green_band', 'Red_band', 'IR_band'],
        total_col='Total_band'
    )
    
    # Save processed data for reference
    processed_csv = os.path.join(results_dir, "processed_netcdf_data.csv")
    df.to_csv(processed_csv, index=False)
    logging.info(f"✅ Processed data saved to {processed_csv}")
    
    # Feature selection - use available columns
    potential_features = [
        'T_air', 'GHI', 'RH', 'Wind_Speed', 'latitude', 'longitude'
        # Add more based on what's available in your NetCDF
    ]
    
    # Check which features are actually available
    available_features = [f for f in potential_features if f in df.columns]
    logging.info(f"Available features: {available_features}")
    
    if 'PV_Potential' not in df.columns:
        logging.error("❌ PV_Potential not calculated - cannot proceed with ML")
        return
    
    # Prepare for ML
    X = filter_valid_columns(df, available_features).dropna()
    y = df.loc[X.index, 'PV_Potential']
    
    logging.info(f"ML dataset: {len(X)} samples, {len(available_features)} features")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    logging.info("\n=== Model Performance ===")
    logging.info(f"R²: {r2_score(y_test, y_pred):.4f}")
    logging.info(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
    logging.info(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    
    # Save model
    model_path = os.path.join(results_dir, "models", "rf_pv_model_netcdf.joblib")
    joblib.dump(model, model_path)
    logging.info(f"💾 Model saved to: {model_path}")
    
    # Feature importance plot
    if len(available_features) > 0:
        importances = model.feature_importances_
        feature_df = pd.DataFrame({
            'Feature': available_features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_df, x='Importance', y='Feature')
        plt.title("Feature Importance - NetCDF Based Model")
        plt.tight_layout()
        plot_path = os.path.join(results_dir, "feature_importance_netcdf.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"📊 Feature importance plot saved to {plot_path}")
    
    logging.info("\n✅ NetCDF-based feature preparation completed!")

def main():
    """Entry point for command line execution."""
    args = parse_args()

    if args.db_url:
        from database_utils import read_table, write_dataframe
        try:
            df_db = read_table(args.db_table, db_url=args.db_url)
        except Exception as e:
            logging.error(f"❌ Failed to read table {args.db_table}: {e}")
            return
        temp_path = "db_input.csv"
        df_db.to_csv(temp_path, index=False)
        main_csv_workflow(temp_path, args.validated_file, args.physics_file, args.results_dir)
        final_csv = os.path.join(args.results_dir, "merged_with_physics_pv.csv")
        if os.path.exists(final_csv):
            processed = pd.read_csv(final_csv)
            write_dataframe(processed, args.db_table, db_url=args.db_url, if_exists="replace")
        if os.path.exists(temp_path):
            os.remove(temp_path)
    elif os.path.exists(args.netcdf_file):
        logging.info("🌐 NetCDF file found - running NetCDF workflow")
        main_netcdf_workflow(args.netcdf_file, args.results_dir)
    elif os.path.exists(args.input_file):
        logging.info("📊 CSV file found - running CSV workflow")
        main_csv_workflow(args.input_file, args.validated_file, args.physics_file, args.results_dir)
    else:
        logging.error("❌ Neither NetCDF nor CSV file found")
        logging.info(f"Looking for NetCDF: {args.netcdf_file}")
        logging.info(f"Looking for CSV: {args.input_file}")

if __name__ == "__main__":
    main()
