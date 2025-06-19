#!/usr/bin/env python
"""
Standalone script to visualize radiative cooling potential maps for EU area.
Takes yearly and seasonal aggregated CSV files as input and creates maps.

Usage:
    python visualize_rc_maps.py yearly_file.csv seasonal_file.csv output_directory
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec

OUTPUT_FORMAT = os.getenv("MAP_FORMAT", "png")

def load_data(yearly_file, seasonal_file):
    """
    Load yearly and seasonal data from CSV files.
    
    Parameters:
        yearly_file (str): Path to yearly aggregated CSV file
        seasonal_file (str): Path to seasonal aggregated CSV file
        
    Returns:
        tuple: (yearly_df, seasonal_df) DataFrames with radiative cooling data
    """
    try:
        yearly_df = pd.read_csv(yearly_file)
        seasonal_df = pd.read_csv(seasonal_file)
        
        print(f"Loaded yearly data: {yearly_file}")
        print(f"Available years: {yearly_df['year'].unique()}")
        
        print(f"Loaded seasonal data: {seasonal_file}")
        print(f"Available seasons: {seasonal_df['season'].unique()}")
        
        return yearly_df, seasonal_df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)


def get_eu_boundaries(df):
    """
    Get latitude and longitude boundaries for the data, focused on EU region.
    
    Parameters:
        df (pandas.DataFrame): DataFrame with latitude and longitude columns
        
    Returns:
        tuple: (lon_min, lon_max, lat_min, lat_max) boundary coordinates
    """
    # Get the boundaries from the data
    lon_min = df['longitude'].min()
    lon_max = df['longitude'].max()
    lat_min = df['latitude'].min()
    lat_max = df['latitude'].max()
    
    # Restrict to EU region if data covers a wider area
    lon_min_eu = max(-20, lon_min)
    lon_max_eu = min(40, lon_max)
    lat_min_eu = max(35, lat_min)
    lat_max_eu = min(72, lat_max)
    
    # Use EU boundaries if they're within our data range
    lon_min = lon_min_eu if lon_min_eu >= lon_min else lon_min
    lon_max = lon_max_eu if lon_max_eu <= lon_max else lon_max
    lat_min = lat_min_eu if lat_min_eu >= lat_min else lat_min
    lat_max = lat_max_eu if lat_max_eu <= lat_max else lat_max
    
    return lon_min, lon_max, lat_min, lat_max


def create_yearly_maps(yearly_df, output_dir, boundaries):
    """
    Create yearly maps for P_rc_basic and P_rc_net.
    
    Parameters:
        yearly_df (pandas.DataFrame): DataFrame with yearly aggregated data
        output_dir (str): Directory to save output maps
        boundaries (tuple): (lon_min, lon_max, lat_min, lat_max) boundary coordinates
    """
    lon_min, lon_max, lat_min, lat_max = boundaries
    
    # Select the most recent year if multiple years are available
    latest_year = yearly_df['year'].max()
    df_year = yearly_df[yearly_df['year'] == latest_year]
    
    # Create a figure with two maps side by side
    plt.figure(figsize=(18, 8))
    
    # Setup for both plots
    projection = ccrs.PlateCarree()
    
    # Common colormap properties
    cmap = 'viridis'
    
    # First map: P_rc_basic
    ax1 = plt.subplot(1, 2, 1, projection=projection)
    
    # Get min/max values for consistent color scale
    vmin_basic = max(0, df_year['P_rc_basic'].min())
    vmax_basic = df_year['P_rc_basic'].max()
    
    # Create scatter plot with P_rc_basic data
    sc1 = ax1.scatter(df_year['longitude'], df_year['latitude'], 
                     c=df_year['P_rc_basic'], cmap=cmap,
                     transform=ccrs.PlateCarree(), s=20, alpha=0.7,
                     vmin=vmin_basic, vmax=vmax_basic, edgecolor='none')
    
    # Add coastlines and borders
    ax1.coastlines(resolution='50m')
    ax1.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Set map extent to EU
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    # Add gridlines
    gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Add title and colorbar
    ax1.set_title(f'Basic Radiative Cooling Potential (P_rc_basic) - {latest_year}', fontsize=12)
    cbar1 = plt.colorbar(sc1, ax=ax1, pad=0.01, shrink=0.8)
    cbar1.set_label('RC Potential (W/m²)', fontsize=10)
    
    # Second map: P_rc_net
    ax2 = plt.subplot(1, 2, 2, projection=projection)
    
    # Get min/max values for consistent color scale
    vmin_net = max(0, df_year['P_rc_net'].min())
    vmax_net = df_year['P_rc_net'].max()
    
    # Create scatter plot with P_rc_net data
    sc2 = ax2.scatter(df_year['longitude'], df_year['latitude'], 
                     c=df_year['P_rc_net'], cmap=cmap,
                     transform=ccrs.PlateCarree(), s=20, alpha=0.7,
                     vmin=vmin_net, vmax=vmax_net, edgecolor='none')
    
    # Add coastlines and borders
    ax2.coastlines(resolution='50m')
    ax2.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Set map extent to EU
    ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    # Add gridlines
    gl = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Add title and colorbar
    ax2.set_title(f'Net Radiative Cooling Potential (P_rc_net) - {latest_year}', fontsize=12)
    cbar2 = plt.colorbar(sc2, ax=ax2, pad=0.01, shrink=0.8)
    cbar2.set_label('RC Potential (W/m²)', fontsize=10)

    if 'Cluster_ID' in df_year.columns:
        unique_clusters = sorted(df_year['Cluster_ID'].unique())
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=f'C{i%10}', label=str(c), markersize=6)
                   for i, c in enumerate(unique_clusters)]
        ax2.legend(handles=handles, title='Cluster', loc='lower left')
    
    # Add overall title
    plt.suptitle(f'Annual Radiative Cooling Potential for Europe - {latest_year}', fontsize=16, y=0.98)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_file = os.path.join(output_dir, f'yearly_rc_potential_{latest_year}.{OUTPUT_FORMAT}')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved yearly map to: {output_file}")


def create_seasonal_maps(seasonal_df, output_dir, boundaries, variable):
    """
    Create seasonal maps for a specific variable (P_rc_basic or P_rc_net).
    
    Parameters:
        seasonal_df (pandas.DataFrame): DataFrame with seasonal aggregated data
        output_dir (str): Directory to save output maps
        boundaries (tuple): (lon_min, lon_max, lat_min, lat_max) boundary coordinates
        variable (str): Variable to plot ('P_rc_basic' or 'P_rc_net')
    """
    lon_min, lon_max, lat_min, lat_max = boundaries
    
    # Select the most recent year if multiple years are available
    latest_year = seasonal_df['year'].max()
    df_year = seasonal_df[seasonal_df['year'] == latest_year]
    
    # Check if the variable exists
    if variable not in df_year.columns:
        print(f"Warning: Variable {variable} not found in data. Available variables: {df_year.columns}")
        return
    
    # Get seasons in order
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    available_seasons = df_year['season'].unique()
    
    # Filter to only include seasons that exist in the data
    seasons = [s for s in seasons if s in available_seasons]
    
    if not seasons:
        print("No season data available")
        return
    
    # Create a figure with maps for each season
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # Setup for all plots
    projection = ccrs.PlateCarree()
    
    # Common colormap properties
    cmap = 'viridis'
    
    # Get min/max values for consistent color scale across all seasons
    vmin = max(0, df_year[variable].min())
    vmax = df_year[variable].max()
    
    # Create a map for each season
    for i, season in enumerate(seasons):
        if i >= 4:  # Only plot up to 4 seasons
            break
        
        # Get data for this season
        df_season = df_year[df_year['season'] == season]
        
        if df_season.empty:
            continue
        
        # Create subplot
        row, col = i // 2, i % 2
        ax = fig.add_subplot(gs[row, col], projection=projection)
        
        # Create scatter plot with data
        sc = ax.scatter(df_season['longitude'], df_season['latitude'], 
                         c=df_season[variable], cmap=cmap,
                         transform=ccrs.PlateCarree(), s=20, alpha=0.7,
                         vmin=vmin, vmax=vmax, edgecolor='none')
        
        # Add coastlines and borders
        ax.coastlines(resolution='50m')
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        
        # Set map extent to EU
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False

        # Add title
        ax.set_title(f'{season} {latest_year}', fontsize=12)

        if 'Cluster_ID' in df_season.columns:
            unique_clusters = sorted(df_season['Cluster_ID'].unique())
            handles = [plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor=f'C{i%10}', label=str(c), markersize=6)
                       for i, c in enumerate(unique_clusters)]
            ax.legend(handles=handles, title='Cluster', loc='lower left')
    
    # Add a colorbar that applies to all subplots
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label(f'{variable} (W/m²)', fontsize=12)
    
    # Add overall title
    variable_label = "Basic" if variable == "P_rc_basic" else "Net"
    plt.suptitle(f'Seasonal {variable_label} Radiative Cooling Potential for Europe - {latest_year}', 
                 fontsize=16, y=0.98)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    variable_str = "basic" if variable == "P_rc_basic" else "net"
    output_file = os.path.join(
        output_dir,
        f"seasonal_rc_potential_{variable_str}_{latest_year}.{OUTPUT_FORMAT}",
    )
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved seasonal {variable} map to: {output_file}")


def main():
    """Main execution function"""
    # Parse command line arguments
    if len(sys.argv) != 4:
        print(
            "Usage: python visualize_rc_maps.py "
            "yearly_file.csv seasonal_file.csv output_directory"
        )
        sys.exit(1)

    yearly_file = sys.argv[1]
    seasonal_file = sys.argv[2]
    output_dir = sys.argv[3]

    # Validate input files
    if not os.path.isfile(yearly_file):
        print(
            f"Error: Yearly file {yearly_file} does not exist or "
            "is not a file."
        )
        sys.exit(1)

    if not os.path.isfile(seasonal_file):
        print(
            f"Error: Seasonal file {seasonal_file} does not exist or "
            "is not a file."
        )
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")  # plain string
    yearly_df, seasonal_df = load_data(yearly_file, seasonal_file)

    # Get EU boundaries
    boundaries = get_eu_boundaries(yearly_df)
    print(
        f"Map boundaries: Longitude [{boundaries[0]:.2f}, "
        f"{boundaries[1]:.2f}], Latitude [{boundaries[2]:.2f}, "
        f"{boundaries[3]:.2f}]"
    )

    # Create yearly maps
    print("Creating yearly maps...")  # plain string
    create_yearly_maps(yearly_df, output_dir, boundaries)

    # Create seasonal maps for both variables
    print("Creating seasonal maps for P_rc_basic...")  # plain string
    create_seasonal_maps(seasonal_df, output_dir, boundaries, 'P_rc_basic')
    
    print("Creating seasonal maps for P_rc_net...")  # plain string
    create_seasonal_maps(seasonal_df, output_dir, boundaries, 'P_rc_net')
    
    print("All maps generated successfully!")


if __name__ == "__main__":
    main()
