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


def add_synergy_index(df, gamma_pv=-0.004, rc_energy_col=None):
    """Return DataFrame with a new ``Synergy_Index`` column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing ``T_PV``, ``T_RC`` and ``GHI`` columns.
    gamma_pv : float, optional
        PV temperature coefficient. Default ``-0.004``.
    rc_energy_col : str, optional
        Name of a column with additional RC cooling energy.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with the new column appended.
    """
    required_cols = ["T_PV", "T_RC", "GHI"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.copy()
    delta_T = df["T_PV"] - df["T_RC"]
    delta_P = abs(gamma_pv) * delta_T * df["GHI"]

    if rc_energy_col and rc_energy_col in df.columns:
        rc_energy = df[rc_energy_col]
    else:
        rc_energy = 0

    synergy_benefit = delta_P + rc_energy
    df["Synergy_Index"] = np.where(
        df["GHI"] > 0,
        (synergy_benefit / df["GHI"]) * 100,
        0,
    )
    return df


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
    df = add_synergy_index(df, gamma_pv=gamma_pv, rc_energy_col=rc_energy_col)

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
    import argparse
    from database_utils import read_table, write_dataframe

    parser = argparse.ArgumentParser(description="Add Synergy_Index to a dataset")
    parser.add_argument("--input", default="clustered_dataset.csv", help="Input CSV path")
    parser.add_argument("--output", default="clustered_dataset_synergy.csv", help="Output CSV path")
    parser.add_argument("--db-url", default=os.getenv("PV_DB_URL"), help="Database URL")
    parser.add_argument("--db-table", default=os.getenv("PV_DB_TABLE", "pv_data"), help="Table name for DB operations")
    args = parser.parse_args()

    if args.db_url:
        df = read_table(args.db_table, db_url=args.db_url)
        df = add_synergy_index(df)
        write_dataframe(df, args.db_table, db_url=args.db_url, if_exists="replace")
        df.to_csv(args.output, index=False)
        print(f"✅ Results written to DB table {args.db_table}")
    else:
        add_synergy_index_to_dataset_vectorized(args.input, args.output)
