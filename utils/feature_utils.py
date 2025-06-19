import os
import logging
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml


def filter_valid_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Return a copy of df with only the columns that exist."""
    valid = [c for c in columns if c in df.columns]
    missing = set(columns) - set(valid)
    if missing:
        logging.warning(f"Missing columns: {sorted(missing)}")
    return df[valid].copy()


def compute_band_ratios(
    df: pd.DataFrame, band_cols: List[str], total_col: str = "Total_band"
) -> Tuple[pd.DataFrame, List[str]]:
    """Add ratio columns for each band relative to ``total_col``."""
    ratio_cols: List[str] = []
    if total_col not in df.columns:
        logging.warning(f"'{total_col}' not found for band ratio computation")
        return df, ratio_cols
    total = df[total_col].replace(0, np.nan)
    for col in band_cols:
        if col in df.columns:
            ratio_name = f"{col}_ratio"
            df[ratio_name] = df[col] / total
            ratio_cols.append(ratio_name)
    return df, ratio_cols


def spectral_summary(df: pd.DataFrame, band_cols: List[str]) -> pd.DataFrame:
    """Return basic statistics for the provided spectral columns."""
    return df[band_cols].describe().T


def compute_cluster_spectra(df_clustered: pd.DataFrame, cluster_col: str = "Cluster_ID") -> pd.DataFrame:
    """Compute normalized mean spectra and temperature per cluster."""
    spectral_col_options = {
        "Blue": ["Blue_band", "Blue_Band", "Blue"],
        "Green": ["Green_band", "Green_Band", "Green"],
        "Red": ["Red_band", "Red_Band", "Red"],
        "IR": ["IR_band", "NIR_band", "IR_Band", "IR"],
    }
    spectral_cols = []
    spectral_mapping = {}
    for band, options in spectral_col_options.items():
        for col in options:
            if col in df_clustered.columns:
                spectral_cols.append(col)
                spectral_mapping[band] = col
                break
    if not spectral_cols:
        logging.warning("No spectral columns found for cluster analysis")
        return pd.DataFrame()

    temp_col = None
    for col in ["T_air", "T2M", "Temperature"]:
        if col in df_clustered.columns:
            temp_col = col
            break
    if not temp_col:
        logging.warning("No temperature column found")
        return pd.DataFrame()

    grouped = df_clustered.groupby(cluster_col)
    cluster_spectra_df = grouped[spectral_cols + [temp_col]].mean().reset_index()

    if len(spectral_cols) > 1:
        spectrum_sum = cluster_spectra_df[spectral_cols].sum(axis=1)
        for col in spectral_cols:
            cluster_spectra_df[col] = cluster_spectra_df[col] / spectrum_sum

    for band, col in spectral_mapping.items():
        if col in cluster_spectra_df.columns:
            cluster_spectra_df[f"{band}_band"] = cluster_spectra_df[col]
    return cluster_spectra_df


def save_config(config: dict, output_dir: str) -> str:
    """Save configuration dictionary to a YAML file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"config_dump_{timestamp}.yaml")
    with open(path, "w") as f:
        yaml.dump(config, f)
    logging.info(f"Saved configuration to {path}")
    return path
