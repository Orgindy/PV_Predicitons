#!/usr/bin/env python3
"""
Spectral Data Analysis Tool

This script provides tools for analyzing and visualizing the spectral data
output from SMARTS simulations.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import glob
from scipy import integrate
from config import get_path
import logging
from plot_utils import apply_standard_plot_style, save_figure

logging.basicConfig(level=logging.INFO)


class DataProcessor:
    """Utility class for loading and storing spectral data files."""

    def __init__(self, input_folder: str, output_folder: str):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def load_data(self, file_path: Path) -> pd.DataFrame:
        return pd.read_csv(file_path, delim_whitespace=True)


def load_spectral_data(file_path):
    """
    Load spectral data from a SMARTS output file.

    Args:
        file_path: Path to the .ext.txt file

    Returns:
        DataFrame with the spectral data
    """
    try:
        # Detect header line first
        header_line = None
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if line.strip().startswith("Wvlgth"):
                    header_line = i
                    header = line.strip().split()
                    break

        if header_line is None:
            logging.info(f"❌ 'Wvlgth' not found in {file_path}")
            return None

        chunks = []
        for chunk in pd.read_csv(
            file_path,
            delim_whitespace=True,
            header=None,
            names=header,
            skiprows=header_line + 1,
            chunksize=50000,
            engine="python",
        ):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True).astype(float)
        return df
    except Exception as e:
        logging.info(f"Error loading {file_path}: {e}")
        return None


def load_combined_data(file_path):
    """
    Load combined spectral data from a CSV file.

    Args:
        file_path: Path to the combined CSV file

    Returns:
        DataFrame with the combined spectral data
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logging.info(f"Error loading {file_path}: {e}")
        return None


def batch_process_smarts_outputs(input_folder, output_file):
    """
    Processes all SMARTS .ext.txt files in a directory and extracts spectral data.

    Parameters:
    - input_folder (str): Directory containing SMARTS .ext.txt files.
    - output_file (str): Path to the combined CSV output file.

    Returns:
    - None
    """
    # Find all .ext.txt files
    ext_files = [f for f in os.listdir(input_folder) if f.endswith(".ext.txt")]

    if not ext_files:
        logging.info("❌ No .ext.txt files found in the directory.")
        return

    # Initialize empty list for storing processed data
    all_data = []

    for ext_file in ext_files:
        file_path = os.path.join(input_folder, ext_file)

        try:
            # Extract location from filename
            parts = ext_file.replace(".ext.txt", "").split("_")
            try:
                latitude = float(parts[0])
                longitude = float(parts[1])
            except (IndexError, ValueError):
                logging.info(
                    f"⚠️ Could not parse lat/lon from filename {ext_file}. Skipping."
                )
                continue

            # Read the .ext.txt file
            df = pd.read_csv(file_path, delim_whitespace=True, header=None, skiprows=3)

            # Extract relevant bands (UV, Blue, Green, Red, IR, SWIR, FIR)
            band_names = ["UV", "Blue", "Green", "Red", "NIR", "SWIR", "FIR"]
            band_ranges = [
                (0.3, 0.4),
                (0.4, 0.5),
                (0.5, 0.6),
                (0.6, 0.7),
                (0.7, 1.0),
                (1.0, 2.5),
                (2.5, 50.0),
            ]

            band_data = {}
            for band_name, (low, high) in zip(band_names, band_ranges):
                band_df = df[(df[0] >= low) & (df[0] <= high)]
                band_data[band_name] = band_df[1].sum()

            # Add coordinates
            band_data["latitude"] = latitude
            band_data["longitude"] = longitude

            # Append to the final dataset
            all_data.append(band_data)

        except Exception as e:
            logging.info(f"❌ Failed to process {ext_file}: {e}")

    # Save combined data
    final_df = pd.DataFrame(all_data)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_df.to_csv(output_file, index=False)
    logging.info(f"✅ Combined spectral data saved to {output_file}")


def combine_band_data(input_folder, output_file):
    """
    Combines spectral band data from multiple SMARTS .ext.txt files.

    Parameters:
    - input_folder (str): Directory containing processed band data files.
    - output_file (str): Path to the combined CSV output file.

    Returns:
    - None
    """
    # Find all processed band data files
    csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

    if not csv_files:
        logging.info("❌ No band data files found in the directory.")
        return

    # Initialize empty DataFrame for combined data
    combined_df = pd.DataFrame()

    for csv_file in csv_files:
        file_path = os.path.join(input_folder, csv_file)

        try:
            # Read the band data
            df = pd.read_csv(file_path)

            # Check for required columns
            required_columns = [
                "latitude",
                "longitude",
                "UV",
                "Blue",
                "Green",
                "Red",
                "NIR",
                "SWIR",
                "FIR",
            ]
            missing_cols = [c for c in required_columns if c not in df.columns]
            assert not missing_cols, f"Missing columns {missing_cols} in {csv_file}"

            # Merge with the main combined dataset
            combined_df = pd.concat([combined_df, df], ignore_index=True)

        except Exception as e:
            logging.info(f"❌ Failed to process {csv_file}: {e}")

    # Save the final combined dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    logging.info(f"✅ Combined band data saved to {output_file}")


def add_spectral_ratios(input_file, output_file):
    """
    Adds spectral band ratios to the combined band data file.

    Parameters:
    - input_file (str): Path to the combined band data file.
    - output_file (str): Path to the final output file with ratios.

    Returns:
    - None
    """
    # Load combined band data
    df = pd.read_csv(input_file)

    required_cols = ["UV", "Blue", "Green", "Red", "NIR", "SWIR", "FIR"]
    missing = [c for c in required_cols if c not in df.columns]
    assert not missing, f"Missing spectral columns: {missing}"

    # Calculate total irradiance
    df["Total_Irradiance"] = df[
        ["UV", "Blue", "Green", "Red", "NIR", "SWIR", "FIR"]
    ].sum(axis=1)

    # Calculate band ratios
    for band in ["UV", "Blue", "Green", "Red", "NIR", "SWIR", "FIR"]:
        df[f"{band}_Ratio"] = df[band] / df["Total_Irradiance"]

    # Save the updated file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    logging.info(f"✅ Spectral ratios added and saved to {output_file}")


def calculate_band_metrics(
    df, wavelength_col="Wvlgth", irradiance_cols=None, bands=None
):
    """
    Calculate integrated irradiance for specified bands.

    Args:
        df: DataFrame with spectral data
        wavelength_col: Name of the wavelength column
        irradiance_cols: List of irradiance column names to integrate
        bands: Dictionary of band name to (min_nm, max_nm) tuples

    Returns:
        DataFrame with band metrics
    """
    if bands is None:
        bands = {
            "UV": (280, 400),
            "Blue": (400, 500),
            "Green": (500, 600),
            "Red": (600, 700),
            "NIR": (700, 1100),
            "SWIR1": (1100, 1700),
            "SWIR2": (1700, 2500),
            "FIR": (2500, 4000),
            "PAR": (400, 700),
            "Total": (280, 4000),
        }

    if irradiance_cols is None:
        # Try to find common irradiance columns
        standard_cols = ["Global_Tilt", "Beam_Normal", "Difuse_Horiz", "Global_Horiz"]
        irradiance_cols = [col for col in standard_cols if col in df.columns]

        if not irradiance_cols:
            # Try to guess based on column name patterns
            irradiance_cols = [
                col
                for col in df.columns
                if any(
                    term in col.lower()
                    for term in ["irrad", "tilt", "beam", "difuse", "global", "horiz"]
                )
            ]

    # Make sure wavelength is numeric
    df[wavelength_col] = pd.to_numeric(df[wavelength_col])

    # Sort by wavelength
    df = df.sort_values(by=wavelength_col)

    results = []

    # Process each location/time combination if present
    for location_id in (
        df["location_id"].unique() if "location_id" in df.columns else [None]
    ):
        for date_time in (
            df["date_time"].unique() if "date_time" in df.columns else [None]
        ):
            # Filter data if needed
            if location_id is not None and date_time is not None:
                filtered_df = df[
                    (df["location_id"] == location_id) & (df["date_time"] == date_time)
                ]
            elif location_id is not None:
                filtered_df = df[df["location_id"] == location_id]
            elif date_time is not None:
                filtered_df = df[df["date_time"] == date_time]
            else:
                filtered_df = df

            # Calculate band metrics for each irradiance column
            for irradiance_col in irradiance_cols:
                for band_name, (band_min, band_max) in bands.items():
                    # Filter to the band wavelength range
                    band_df = filtered_df[
                        (filtered_df[wavelength_col] >= band_min)
                        & (filtered_df[wavelength_col] <= band_max)
                    ]

                    if len(band_df) < 2:
                        continue

                    # Integrate using trapezoidal rule
                    try:
                        band_df = band_df.sort_values(by=wavelength_col)
                        band_irradiance = integrate.trapz(
                            band_df[irradiance_col], band_df[wavelength_col]
                        )

                        # Calculate normalized metrics
                        if band_name != "Total":
                            total_df = filtered_df[
                                (filtered_df[wavelength_col] >= 280)
                                & (filtered_df[wavelength_col] <= 4000)
                            ]
                            if len(total_df) >= 2:
                                total_df = total_df.sort_values(by=wavelength_col)
                                total_irradiance = integrate.trapz(
                                    total_df[irradiance_col], total_df[wavelength_col]
                                )
                                normalized = (
                                    band_irradiance / total_irradiance
                                    if total_irradiance > 0
                                    else 0
                                )
                            else:
                                total_irradiance = 0
                                normalized = 0
                        else:
                            total_irradiance = band_irradiance
                            normalized = 1.0

                        # Store results
                        result = {
                            "location_id": location_id,
                            "date_time": date_time,
                            "irradiance_type": irradiance_col,
                            "band_name": band_name,
                            "band_min_nm": band_min,
                            "band_max_nm": band_max,
                            "irradiance_W_m2": band_irradiance,
                            "total_irradiance_W_m2": total_irradiance,
                            "normalized_fraction": normalized,
                        }
                        results.append(result)
                    except Exception as e:
                        logging.info(
                            f"Error calculating {band_name} for {irradiance_col}: {e}"
                        )

    return pd.DataFrame(results)


def plot_spectral_irradiance(
    df,
    wavelength_col="Wvlgth",
    irradiance_cols=None,
    title=None,
    log_scale=False,
    show_bands=True,
):
    """
    Plot spectral irradiance.

    Args:
        df: DataFrame with spectral data
        wavelength_col: Name of the wavelength column
        irradiance_cols: List of irradiance column names to plot
        title: Plot title
        log_scale: Whether to use logarithmic scale for y-axis
        show_bands: Whether to show spectral bands as background colors

    Returns:
        Matplotlib figure
    """
    if irradiance_cols is None:
        # Try to find common irradiance columns
        standard_cols = ["Global_Tilt", "Beam_Normal", "Difuse_Horiz", "Global_Horiz"]
        irradiance_cols = [col for col in standard_cols if col in df.columns]

        if not irradiance_cols:
            # Try to guess based on column name patterns
            irradiance_cols = [
                col
                for col in df.columns
                if any(
                    term in col.lower()
                    for term in ["irrad", "tilt", "beam", "difuse", "global", "horiz"]
                )
            ]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Show spectral bands if requested
    if show_bands:
        bands = {
            "UV": ((280, 400), "lavender"),
            "Blue": ((400, 500), "lightblue"),
            "Green": ((500, 600), "lightgreen"),
            "Red": ((600, 700), "mistyrose"),
            "NIR": ((700, 1100), "bisque"),
            "SWIR": ((1100, 2500), "wheat"),
            "FIR": ((2500, 4000), "lightgray"),
        }

        for band_name, ((band_min, band_max), color) in bands.items():
            if band_min >= min(df[wavelength_col]) and band_max <= max(
                df[wavelength_col]
            ):
                ax.axvspan(band_min, band_max, alpha=0.2, color=color, label=band_name)

    # Plot each irradiance component
    for col in irradiance_cols:
        ax.plot(df[wavelength_col], df[col], label=col)

    if title:
        plot_title = title
    else:
        location = (
            df["location_id"].iloc[0] if "location_id" in df.columns else "Unknown"
        )
        date_time = df["date_time"].iloc[0] if "date_time" in df.columns else "Unknown"
        plot_title = f"Spectral Irradiance - {location} - {date_time}"

    if log_scale:
        ax.set_yscale("log")

    ax.legend(loc="upper right")

    # Set reasonable x-axis limits
    ax.set_xlim(min(df[wavelength_col]), min(4000, max(df[wavelength_col])))

    apply_standard_plot_style(
        ax,
        title=plot_title,
        xlabel="Wavelength (nm)",
        ylabel="Spectral Irradiance (W/m²/nm)",
    )
    return fig


def create_spectral_composition_plots(band_df, output_dir):
    """
    Create plots showing the spectral composition for different locations and conditions.

    Args:
        band_df: DataFrame with band metrics
        output_dir: Directory to save plots

    Returns:
        List of created plot file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    created_files = []

    # Extract the set of bands (excluding 'Total')
    bands = sorted([band for band in band_df["band_name"].unique() if band != "Total"])

    # Plot for each irradiance type
    for irradiance_type in band_df["irradiance_type"].unique():
        # Filter to this irradiance type and exclude 'Total' band
        df_filt = band_df[
            (band_df["irradiance_type"] == irradiance_type)
            & (band_df["band_name"] != "Total")
        ]

        # Skip if no data
        if len(df_filt) == 0:
            continue

        # Create a stacked bar chart by location/datetime
        locations = df_filt["location_id"].unique()
        datetimes = df_filt["date_time"].unique()

        # Group by location
        for location in locations:
            df_loc = df_filt[df_filt["location_id"] == location]

            # Create a plot of the spectral composition
            fig, ax = plt.subplots(figsize=(12, 8))

            # Extract data for all datetimes for this location
            plot_data = []

            for date_time in datetimes:
                df_time = df_loc[df_loc["date_time"] == date_time]

                # Skip if no data for this time
                if len(df_time) == 0:
                    continue

                # Extract band irradiances
                row = {"date_time": date_time}
                for band in bands:
                    band_value = df_time[df_time["band_name"] == band][
                        "normalized_fraction"
                    ].values
                    row[band] = band_value[0] if len(band_value) > 0 else 0

                plot_data.append(row)

            # Convert to DataFrame
            plot_df = pd.DataFrame(plot_data)

            # Skip if no data
            if len(plot_df) == 0:
                continue

            # Sort by date_time
            plot_df = plot_df.sort_values(by="date_time")

            # Create stacked bar chart
            bottom = np.zeros(len(plot_df))

            # Use a color palette suitable for spectral bands
            band_colors = {
                "UV": "purple",
                "Blue": "blue",
                "Green": "green",
                "Red": "red",
                "NIR": "orange",
                "SWIR1": "brown",
                "SWIR2": "sienna",
                "FIR": "gray",
                "PAR": "limegreen",
            }

            # Plot each band
            for band in bands:
                if band in plot_df.columns:
                    color = band_colors.get(
                        band, None
                    )  # Use predefined color or let matplotlib choose
                    ax.bar(
                        plot_df["date_time"],
                        plot_df[band],
                        bottom=bottom,
                        label=band,
                        color=color,
                    )
                    bottom += plot_df[band].values

            # Set plot properties
            ax.set_ylim(0, 1)

            # Format x-axis labels if they're datetime strings
            plt.xticks(rotation=45, ha="right")

            ax.legend(title="Spectral Band", bbox_to_anchor=(1.05, 1), loc="upper left")

            apply_standard_plot_style(
                ax,
                title=f"Spectral Composition - {location} - {irradiance_type}",
                xlabel="Date/Time",
                ylabel="Normalized Fraction",
            )

            filename = f"spectral_composition_{location}_{irradiance_type}.png"
            filepath = os.path.join(output_dir, filename)
            save_figure(fig, filename, folder=output_dir)

            created_files.append(filepath)

    return created_files


def plot_location_comparison(
    band_df, band_name, irradiance_type, output_dir, time_points=None, normalized=True
):
    """
    Create a plot comparing the specified band irradiance across locations.

    Args:
        band_df: DataFrame with band metrics
        band_name: Name of the band to compare
        irradiance_type: Type of irradiance to compare
        output_dir: Directory to save plots
        time_points: Specific time points to include (if None, use all)
        normalized: Whether to use normalized values

    Returns:
        Path to created plot file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filter data
    df_filt = band_df[
        (band_df["band_name"] == band_name)
        & (band_df["irradiance_type"] == irradiance_type)
    ]

    # Filter to specific time points if requested
    if time_points is not None:
        df_filt = df_filt[df_filt["date_time"].isin(time_points)]

    # Skip if no data
    if len(df_filt) == 0:
        logging.info(f"No data for band {band_name}, irradiance {irradiance_type}")
        return None

    # Create a pivot table with locations as rows and times as columns
    if normalized:
        pivot = df_filt.pivot(
            index="location_id", columns="date_time", values="normalized_fraction"
        )
        value_label = "Normalized Fraction"
    else:
        pivot = df_filt.pivot(
            index="location_id", columns="date_time", values="irradiance_W_m2"
        )
        value_label = "Irradiance (W/m²)"

    # Sort by mean value
    pivot["mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values(by="mean", ascending=False)
    pivot = pivot.drop(columns=["mean"])

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.3)))

    # Create a heatmap
    cmap = "YlOrRd" if band_name in ["UV", "FIR"] else "viridis"
    sns.heatmap(pivot, cmap=cmap, annot=True, fmt=".3f", linewidths=0.5, ax=ax)

    apply_standard_plot_style(
        ax,
        title=f"{band_name} Band - {irradiance_type}",
        xlabel="Date/Time",
        ylabel="Location",
    )

    filename = f"{band_name}_{irradiance_type}_location_comparison.png"
    filepath = os.path.join(output_dir, filename)
    save_figure(fig, filename, folder=output_dir)

    return filepath


def plot_temporal_variation(band_df, location_id, band_name, output_dir):
    """
    Plot the temporal variation of spectral bands for a specific location.

    Args:
        band_df: DataFrame with band metrics
        location_id: Location to plot
        band_name: Band to plot (or 'all' for all bands)
        output_dir: Directory to save plots

    Returns:
        Path to created plot file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filter to the specified location
    df_loc = band_df[band_df["location_id"] == location_id]

    # Filter to the specified band or get all bands except 'Total'
    if band_name.lower() == "all":
        bands = [b for b in df_loc["band_name"].unique() if b != "Total"]
    else:
        bands = [band_name]
        df_loc = df_loc[df_loc["band_name"].isin(bands)]

    # Skip if no data
    if len(df_loc) == 0:
        logging.info(f"No data for location {location_id}, band {band_name}")
        return None

    # Get unique irradiance types
    irradiance_types = df_loc["irradiance_type"].unique()

    # Create a subplot for each irradiance type
    fig, axes = plt.subplots(
        len(irradiance_types), 1, figsize=(12, 4 * len(irradiance_types)), sharex=True
    )

    # If only one irradiance type, wrap in a list for consistent indexing
    if len(irradiance_types) == 1:
        axes = [axes]

    for i, irradiance_type in enumerate(irradiance_types):
        ax = axes[i]

        # Filter to this irradiance type
        df_irr = df_loc[df_loc["irradiance_type"] == irradiance_type]

        # Sort by date_time
        df_irr = df_irr.sort_values(by="date_time")

        # Plot each band
        for band in bands:
            df_band = df_irr[df_irr["band_name"] == band]

            # Skip if no data for this band
            if len(df_band) == 0:
                continue

            # Plot absolute irradiance
            ax.plot(
                df_band["date_time"],
                df_band["irradiance_W_m2"],
                marker="o",
                label=f"{band}",
            )

        # Format x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        ax.legend(title="Band")

        apply_standard_plot_style(
            ax,
            title=f"{irradiance_type} - {location_id}",
            ylabel="Irradiance (W/m²)",
        )

    axes[-1].set_xlabel("Date/Time")
    apply_standard_plot_style(axes[-1])

    # Save the plot
    band_str = "all_bands" if band_name.lower() == "all" else band_name
    filename = f"temporal_variation_{location_id}_{band_str}.png"
    filepath = os.path.join(output_dir, filename)
    save_figure(fig, filename, folder=output_dir)

    return filepath


def plot_band_ratios(band_df, output_dir, location_id=None, date_time=None):
    """
    Plot ratios between different spectral bands.

    Args:
        band_df: DataFrame with band metrics
        output_dir: Directory to save plots
        location_id: Specific location to plot (if None, plot all)
        date_time: Specific time to plot (if None, plot all)

    Returns:
        List of created plot file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    created_files = []

    # Define interesting band ratios
    ratios = [
        ("Blue", "Red", "Blue/Red"),
        ("NIR", "Red", "NIR/Red (NDVI-related)"),
        ("UV", "PAR", "UV/PAR"),
        ("NIR", "SWIR1", "NIR/SWIR1"),
    ]

    # Filter data if needed
    if location_id is not None:
        band_df = band_df[band_df["location_id"] == location_id]

    if date_time is not None:
        band_df = band_df[band_df["date_time"] == date_time]

    # Process each irradiance type
    for irradiance_type in band_df["irradiance_type"].unique():
        df_irr = band_df[band_df["irradiance_type"] == irradiance_type]

        # Create ratios for each location/time combination
        results = []

        for loc in df_irr["location_id"].unique():
            for dt in df_irr[df_irr["location_id"] == loc]["date_time"].unique():
                df_lt = df_irr[
                    (df_irr["location_id"] == loc) & (df_irr["date_time"] == dt)
                ]

                # Calculate each ratio
                row = {"location_id": loc, "date_time": dt}

                for band1, band2, ratio_name in ratios:
                    # Get band values
                    band1_val = df_lt[df_lt["band_name"] == band1][
                        "irradiance_W_m2"
                    ].values
                    band2_val = df_lt[df_lt["band_name"] == band2][
                        "irradiance_W_m2"
                    ].values

                    # Calculate ratio if both bands have data
                    if len(band1_val) > 0 and len(band2_val) > 0 and band2_val[0] > 0:
                        row[ratio_name] = band1_val[0] / band2_val[0]
                    else:
                        row[ratio_name] = None

                results.append(row)

        # Skip if no data
        if len(results) == 0:
            continue

        # Convert to DataFrame
        ratio_df = pd.DataFrame(results)

        # Create a plot for each ratio
        for _, _, ratio_name in ratios:
            # Skip if no data for this ratio
            if ratio_name not in ratio_df.columns or ratio_df[ratio_name].isna().all():
                continue

            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 6))

            # If we have multiple locations or times, create a boxplot
            if len(ratio_df["location_id"].unique()) > 1 and location_id is None:
                # Group by location
                sns.boxplot(x="location_id", y=ratio_name, data=ratio_df, ax=ax)
                plt.xticks(rotation=90)
                title = f"{ratio_name} Ratio by Location - {irradiance_type}"
                x_label = "Location"
            elif len(ratio_df["date_time"].unique()) > 1 and date_time is None:
                # Group by time
                sns.boxplot(x="date_time", y=ratio_name, data=ratio_df, ax=ax)
                plt.xticks(rotation=45, ha="right")
                title = f"{ratio_name} Ratio over Time - {irradiance_type}"
                x_label = "Date/Time"
            else:
                # Just plot individual points
                ax.bar(
                    (
                        ratio_df["location_id"]
                        if location_id is None
                        else ratio_df["date_time"]
                    ),
                    ratio_df[ratio_name],
                )
                plt.xticks(rotation=90 if location_id is None else 45, ha="right")
                title = f"{ratio_name} Ratio - {irradiance_type}"
                x_label = "Location" if location_id is None else "Date/Time"

            apply_standard_plot_style(
                ax, title=title, xlabel=x_label, ylabel=ratio_name
            )

            loc_str = f"{location_id}_" if location_id is not None else ""
            time_str = f"{date_time}_" if date_time is not None else ""
            filename = f"band_ratio_{loc_str}{time_str}{ratio_name.replace('/', '_')}_{irradiance_type}.png"
            filepath = os.path.join(output_dir, filename)
            save_figure(fig, filename, folder=output_dir)

            created_files.append(filepath)

    return created_files


def main():
    parser = argparse.ArgumentParser(description="Process SMARTS spectral output")
    parser.add_argument(
        "--input-folder",
        default=get_path("smarts_out_path"),
        help="Directory with SMARTS .ext.txt files",
    )
    parser.add_argument(
        "--output-folder",
        default=os.path.join(get_path("results_path"), "spectral_analysis_output"),
        help="Directory for analysis output",
    )
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)
    if not os.path.isdir(input_folder):
        logging.info(f"❌ Input folder not found: {input_folder}")
        return

    # Find all SMARTS .ext.txt files in the folder
    files = glob.glob(os.path.join(input_folder, "*.ext.txt"))
    if not files:
        logging.info(f"❌ No .ext.txt files found in {input_folder}")
        return

    logging.info(f"📂 Found {len(files)} SMARTS output files in {input_folder}")

    all_data = []
    for file in files:
        try:
            filename = os.path.basename(file)
            parts = filename.replace(".ext.txt", "").split("_")
            location_id = parts[1] if len(parts) >= 3 else "unknown"
            date_time = "_".join(parts[2:]) if len(parts) >= 3 else "unknown"

            df = load_spectral_data(file)
            if df is not None:
                df["location_id"] = location_id
                df["date_time"] = date_time
                all_data.append(df)
        except Exception as e:
            logging.info(f"Error loading {file}: {e}")

    if not all_data:
        logging.info("❌ No valid spectral data loaded.")
        return

    df = pd.concat(all_data, ignore_index=True)
    logging.info(f"✅ Loaded {len(df)} total data points from SMARTS outputs")

    # Calculate band metrics
    band_df = calculate_band_metrics(df)

    # Save the metrics
    metrics_file = os.path.join(output_folder, "band_metrics.csv")
    band_df.to_csv(metrics_file, index=False)
    logging.info(f"💾 Saved band metrics to {metrics_file}")

    # Generate spectral composition plots
    composition_plots = create_spectral_composition_plots(band_df, output_folder)
    logging.info(f"📊 Created {len(composition_plots)} spectral composition plots")

    # Create band ratio plots
    ratio_plots = plot_band_ratios(band_df, output_folder)
    logging.info(f"📈 Created {len(ratio_plots)} band ratio plots")


if __name__ == "__main__":
    main()
