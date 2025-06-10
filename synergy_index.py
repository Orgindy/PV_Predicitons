# -*- coding: utf-8 -*-
"""
Created on Mon May 26 12:21:06 2025

@author: Gindi002
"""
import os
import pandas as pd
import numpy as np

def calculate_synergy_index(
    T_pv,
    T_rc,
    GHI,
    gamma_pv=-0.004,
    rc_cooling_energy=None,
    normalize_to=None
):
    """
    Calculate PV–RC synergy index.

    Parameters:
        T_pv (np.ndarray): Baseline PV cell temperatures [°C]
        T_rc (np.ndarray): RC-enhanced surface temperatures [°C]
        GHI (np.ndarray): Global horizontal irradiance [W/m²]
        gamma_pv (float): PV temperature coefficient (e.g. -0.004 / °C)
        rc_cooling_energy (np.ndarray or None): Optional RC energy benefit [W/m²]
        normalize_to (float or None): Total insolation or energy for normalization [Wh/m²]

    Returns:
        float: Synergy index [%]
    """
    # Convert to numpy arrays for safe operations
    T_pv = np.array(T_pv)
    T_rc = np.array(T_rc)
    GHI = np.array(GHI)
    
    # Validate inputs
    if len(T_pv) != len(T_rc) or len(T_pv) != len(GHI):
        raise ValueError("Input arrays must have the same length")
    
    delta_T = T_pv - T_rc  # Cooling benefit [°C]
    # PV temperature coefficient is typically negative (efficiency drops with
    # higher cell temperature). The benefit from radiative cooling should be
    # positive, so we use the absolute value here.
    delta_P = abs(gamma_pv) * delta_T * GHI  # Instantaneous PV power gain [W/m²]

    if rc_cooling_energy is None:
        rc_cooling_energy = np.zeros_like(delta_P)
    else:
        rc_cooling_energy = np.array(rc_cooling_energy)

    synergy = delta_P + rc_cooling_energy  # Total benefit in watts

    if normalize_to is None:
        normalize_to = GHI.sum()  # Total GHI over the period

    # Avoid division by zero
    if normalize_to == 0:
        return 0.0
        
    synergy_index = (synergy.sum() / normalize_to) * 100  # %

    return synergy_index


def add_synergy_index_to_dataset_vectorized(csv_path, output_path=None, gamma_pv=-0.004, rc_energy_col=None):
    """
    Load a dataset, compute synergy index for each row using vectorized operations, and save updated CSV.

    Parameters:
        csv_path (str): Path to input CSV
        output_path (str): Path to output CSV (default: overwrite input)
        gamma_pv (float): PV temperature coefficient [°C⁻¹]
        rc_energy_col (str or None): Column name for RC energy benefit, if included

    Returns:
        pd.DataFrame: Updated dataframe with 'Synergy_Index' column
    """
    print(f"📥 Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = ["T_PV", "T_RC", "GHI"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print("🧮 Computing synergy index using vectorized operations...")
    
    # Vectorized calculation - much faster than row-by-row
    delta_T = df["T_PV"] - df["T_RC"]  # Temperature reduction benefit
    # Use absolute value so a negative PV temperature coefficient results in a
    # positive power gain when cooling lowers the cell temperature.
    delta_P = abs(gamma_pv) * delta_T * df["GHI"]  # PV power gain
    
    # Add RC cooling energy if available
    if rc_energy_col and rc_energy_col in df.columns:
        rc_energy = df[rc_energy_col]
    else:
        rc_energy = 0
    
    # Total synergy benefit
    synergy_benefit = delta_P + rc_energy
    
    # Normalize by GHI (simple approach - can be customized)
    synergy_index = np.where(df["GHI"] > 0, 
                            (synergy_benefit / df["GHI"]) * 100, 
                            0)
    
    df["Synergy_Index"] = synergy_index

    if output_path is None:
        output_path = csv_path

    # Ensure output directory exists if a directory component is provided
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"✅ Synergy index added and saved to: {output_path}")

    return df


def calculate_synergy_metrics_summary(df, group_by_cols=None):
    """
    Calculate summary statistics for synergy index across different groups.
    
    Parameters:
        df (pd.DataFrame): DataFrame with Synergy_Index column
        group_by_cols (list): Columns to group by (e.g., ['Cluster_ID', 'season'])
    
    Returns:
        pd.DataFrame: Summary statistics
    """
    if 'Synergy_Index' not in df.columns:
        raise ValueError("DataFrame must contain 'Synergy_Index' column")
    
    if group_by_cols:
        summary = df.groupby(group_by_cols)['Synergy_Index'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(3)
    else:
        summary = df['Synergy_Index'].describe()
    
    return summary


if __name__ == "__main__":
    # Example usage - only runs when script is executed directly
    try:
        # Test with a sample dataset
        result_df = add_synergy_index_to_dataset_vectorized("clustered_dataset.csv")
        print(f"📊 Processed {len(result_df)} rows")
        
        # Show summary statistics
        summary = calculate_synergy_metrics_summary(result_df)
        print("\n📈 Synergy Index Summary:")
        print(summary)
        
    except FileNotFoundError:
        print("⚠️ clustered_dataset.csv not found - skipping example")
    except Exception as e:
        print(f"❌ Error: {e}")
