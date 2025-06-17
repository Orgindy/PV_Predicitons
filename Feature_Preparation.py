import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import mutual_info_regression
import os
import argparse
from config import get_nc_dir


def parse_args():
    """Parse command line arguments for file paths."""
    parser = argparse.ArgumentParser(description="Prepare features for PV model")
    parser.add_argument(
        "--input-file",
        default="data/merged_dataset.csv",
        help="Path to merged dataset CSV",
    )
    parser.add_argument(
        "--validated-file",
        default="data/validated_dataset.csv",
        help="Path to save validated CSV",
    )
    parser.add_argument(
        "--physics-file",
        default="data/physics_dataset.csv",
        help="Path to save dataset with physics-based PV potential",
    )
    parser.add_argument(
        "--netcdf-file",
        default=os.path.join(get_nc_dir(), "ERA5_daily.nc"),
        help="Path to processed ERA5 NetCDF file",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
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
def calculate_pv_potential(GHI, T_air, RC_potential, Red_band, Total_band):
    """
    Calculate PV potential corrected by temperature loss, radiative cooling gain, and spectral adjustment.
    
    Parameters:
    - GHI: Global Horizontal Irradiance [W/m²]
    - T_air: Air Temperature at 2m [°C]
    - RC_potential: Radiative Cooling Potential [W/m²]
    - Red_band: Red band irradiance [W/m²]
    - Total_band: Total band irradiance (Blue+Green+Red+IR) [W/m²]
    
    Returns:
    - PV_Potential [W/m²]
    """
    # Constants
    NOCT = 45  # Nominal Operating Cell Temperature [°C]
    Reference_Red_Fraction = 0.42  # From AM1.5 standard
    PR_ref = 0.80  # Reference performance ratio (typical PV system)

    # 1. Estimate PV Cell Temperature
    T_cell = T_air + (NOCT - 20) / 800 * GHI

    # 2. Temperature Loss
    Temp_Loss = -0.0045 * (T_cell - 25)

    # 3. Radiative Cooling Gain
    RC_Gain = 0.01 * (RC_potential / 50)

    # 4. Spectral Adjustment
    with np.errstate(divide='ignore', invalid='ignore'):
        Actual_Red_Fraction = np.divide(Red_band, Total_band, out=np.zeros_like(Red_band), where=Total_band != 0)
    Spectral_Adjust = (Actual_Red_Fraction - Reference_Red_Fraction)

    # 5. Corrected PR
    PR_corrected = PR_ref + Temp_Loss + RC_Gain + Spectral_Adjust
    PR_corrected = np.clip(PR_corrected, 0.7, 0.9)

    # 6. Final PV Potential
    PV_Potential = GHI * PR_corrected  # [W/m²]
    
    return PV_Potential


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
    # Load input data
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
        print(f"✅ Dropped {len(invalid_rows)} rows with invalid values.")
    else:
        df["Invalid_Row"] = 0
        df.loc[invalid_rows, "Invalid_Row"] = 1
        print(f"⚠️ Flagged {len(invalid_rows)} rows as invalid.")
    
    # Save the cleaned file
    df.to_csv(output_file, index=False)
    print(f"✅ Validated data saved to {output_file}")

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
    # Load input data
    df = pd.read_csv(input_file)
    
    # Check for required columns
    required_columns = ["GHI", "T_air", "RC_potential", "Total_band", "Red_band"]
    if not all(col in df.columns for col in required_columns):
        print(f"❌ Missing required columns: {required_columns}")
        return
    
    # Constants
    NOCT = 45  # Nominal Operating Cell Temperature (°C)
    Reference_Red_Fraction = 0.42  # AM1.5 standard red fraction
    PR_ref = 0.80  # Reference performance ratio
    
    # 1. Estimate PV Cell Temperature
    df["T_cell"] = df["T_air"] + (NOCT - 20) / 800 * df["GHI"]
    
    # 2. Temperature Loss
    df["Temp_Loss"] = temp_coeff * (df["T_cell"] - 25)
    
    # 3. Radiative Cooling Gain
    df["RC_Gain"] = 0.01 * (df["RC_potential"] / 50)
    
    # 4. Spectral Adjustment
    df["Red_Fraction"] = df["Red_band"] / df["Total_band"]
    df["Spectral_Adjust"] = (df["Red_Fraction"] - Reference_Red_Fraction)
    
    # 5. Corrected PR
    df["PR_corrected"] = PR_ref + df["Temp_Loss"] + df["RC_Gain"] + df["Spectral_Adjust"]
    df["PR_corrected"] = np.clip(df["PR_corrected"], 0.7, 0.9)  # Clip to realistic range
    
    # 6. Final PV Potential
    df["PV_Potential_physics"] = df["GHI"] * df["PR_corrected"]
    
    # Save the updated file
    df.to_csv(output_file, index=False)
    print(f"✅ Physics-based PV potential added and saved to {output_file}")


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
    
    print("\n=== Feature Weights (Normalized) ===")
    for k, v in normalized_weights.items():
        print(f"{k}: {v:.4f}")
    
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
    print(f"Loading NetCDF data from {netcdf_file}...")
    
    # Load dataset
    with xr.open_dataset(netcdf_file) as ds:
        # Convert to DataFrame (this flattens all dimensions)
        df = ds.load().to_dataframe().reset_index()
    
    # Remove rows with all NaN values
    df = df.dropna(how='all')
    
    # Sample data if too large
    if sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=42)
        print(f"Sampled {len(df)} rows ({sample_fraction*100}% of data)")
    
    print(f"Loaded {len(df)} data points")
    print(f"Available columns: {list(df.columns)}")
    
    return df

def calculate_pv_potential_netcdf(df):
    """
    Calculate PV potential directly from NetCDF data.
    Assumes your NetCDF has variables like T2m, GHI, etc.
    """
    print("Calculating PV potential from NetCDF data...")
    
    # Map NetCDF variable names to expected names
    # Adjust these based on your actual NetCDF variable names
    variable_mapping = {
        'T2m': 'T_air',                    # Temperature
        'SSRD_power': 'GHI',               # Global Horizontal Irradiance  
        'RH': 'RH',                        # Relative Humidity
        'WS': 'Wind_Speed',                # Wind Speed
        'NET_RAD': 'RC_potential',         # Use net radiation as proxy for RC potential
        # Add more mappings based on your NetCDF variables
    }
    
    # Create mapped columns
    for netcdf_var, standard_var in variable_mapping.items():
        if netcdf_var in df.columns:
            df[standard_var] = df[netcdf_var]
    
    # Calculate PV potential using available variables
    if 'GHI' in df.columns and 'T_air' in df.columns:
        # Constants
        NOCT = 45  # Nominal Operating Cell Temperature [°C]
        PR_ref = 0.80  # Reference performance ratio
        
        # Basic PV potential calculation
        df['T_cell'] = df['T_air'] + (NOCT - 20) / 800 * df['GHI']
        df['Temp_Loss'] = -0.0045 * (df['T_cell'] - 25)
        
        # Add RC gain if available
        if 'RC_potential' in df.columns:
            df['RC_Gain'] = 0.01 * (df['RC_potential'] / 50)
        else:
            df['RC_Gain'] = 0  # No RC gain if not available
        
        # Calculate corrected PR
        df['PR_corrected'] = (PR_ref + df['Temp_Loss'] + df['RC_Gain']).clip(0.7, 0.9)
        df['PV_Potential'] = df['GHI'] * df['PR_corrected']
        
        print("✅ PV potential calculated")
    else:
        print("❌ Missing required variables for PV calculation")
        print(f"Available: {[col for col in ['GHI', 'T_air'] if col in df.columns]}")
        print(f"Missing: {[col for col in ['GHI', 'T_air'] if col not in df.columns]}")
    
    return df

def main_csv_workflow(input_file, validated_file, physics_file, results_dir):
    """CSV-based workflow."""
    import joblib
    import os

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
    
    # Load initial data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")

    # --- Data Validation ---
    # Save to temp file, validate, then reload
    temp_input = "temp_input.csv"
    df.to_csv(temp_input, index=False)
    validate_parameters(temp_input, validated_file, drop_invalid=True)
    df = pd.read_csv(validated_file)
    print(f"After validation: {len(df)} rows")

    # --- Add physics-based PV potential ---
    auto_generate_physics_pv(validated_file, physics_file)
    df = pd.read_csv(physics_file)
    print(f"Added physics-based PV potential")

    # Save final processed dataset
    final_output = os.path.join(results_dir, "merged_with_physics_pv.csv")
    df.to_csv(final_output, index=False)
    print(f"✅ Final dataset saved to {final_output}")

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
        print(f"⚠️ Missing features: {missing_features}")
        print(f"✅ Available features: {available_features}")
    
    # Use only available features
    X = df[available_features]
    y = df['PV_Potential_physics']

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Model ---
    model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
    model.fit(X_train, y_train)

    # --- Evaluation ---
    y_pred = model.predict(X_test)
    print("\n=== Model Performance ===")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")

    # --- Save model ---
    model_path = os.path.join(results_dir, "models", "rf_pv_model.joblib")
    joblib.dump(model, model_path)
    print(f"💾 Random Forest model saved to: {model_path}")

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
    print(f"📊 Feature importance plot saved to {os.path.join(results_dir, 'feature_importance_plot.png')}")

    # Clean up temporary files
    for temp_file in [temp_input]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print("\n✅ CSV-based feature preparation completed successfully!")

def main_netcdf_workflow(netcdf_file, results_dir):
    """NetCDF-based workflow."""
    import joblib

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
    
    # Load NetCDF data
    df = load_netcdf_data(netcdf_file, sample_fraction=0.1)  # Use 10% sample for speed
    
    # Calculate PV potential
    df = calculate_pv_potential_netcdf(df)
    
    # Save processed data for reference
    processed_csv = os.path.join(results_dir, "processed_netcdf_data.csv")
    df.to_csv(processed_csv, index=False)
    print(f"✅ Processed data saved to {processed_csv}")
    
    # Feature selection - use available columns
    potential_features = [
        'T_air', 'GHI', 'RH', 'Wind_Speed', 'latitude', 'longitude'
        # Add more based on what's available in your NetCDF
    ]
    
    # Check which features are actually available
    available_features = [f for f in potential_features if f in df.columns]
    print(f"Available features: {available_features}")
    
    if 'PV_Potential' not in df.columns:
        print("❌ PV_Potential not calculated - cannot proceed with ML")
        return
    
    # Prepare for ML
    X = df[available_features].dropna()
    y = df.loc[X.index, 'PV_Potential']
    
    print(f"ML dataset: {len(X)} samples, {len(available_features)} features")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    print("\n=== Model Performance ===")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    
    # Save model
    model_path = os.path.join(results_dir, "models", "rf_pv_model_netcdf.joblib")
    joblib.dump(model, model_path)
    print(f"💾 Model saved to: {model_path}")
    
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
        print(f"📊 Feature importance plot saved to {plot_path}")
    
    print("\n✅ NetCDF-based feature preparation completed!")

def main():
    """Entry point for command line execution."""
    args = parse_args()

    if args.db_url:
        from database_utils import read_table, write_dataframe
        try:
            df_db = read_table(args.db_table, db_url=args.db_url)
        except Exception as e:
            print(f"❌ Failed to read table {args.db_table}: {e}")
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
        print("🌐 NetCDF file found - running NetCDF workflow")
        main_netcdf_workflow(args.netcdf_file, args.results_dir)
    elif os.path.exists(args.input_file):
        print("📊 CSV file found - running CSV workflow")
        main_csv_workflow(args.input_file, args.validated_file, args.physics_file, args.results_dir)
    else:
        print("❌ Neither NetCDF nor CSV file found")
        print(f"Looking for NetCDF: {args.netcdf_file}")
        print(f"Looking for CSV: {args.input_file}")

if __name__ == "__main__":
    main()
