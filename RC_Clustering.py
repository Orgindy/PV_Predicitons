import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids


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
    T_sky_K = T_air_K * (eps_sky**0.25)

    return T_sky_K - 273.15


def rc_only_clustering(df, features=None, n_clusters=5, cluster_col="RC_Cluster"):
    """
    Perform K-Medoids clustering based on RC potential and thermal features.

    Parameters:
    - df: DataFrame with relevant columns
    - features: list of feature columns to use (default: RC + T + wind + RH)
    - n_clusters: number of clusters
    - cluster_col: name of the column to store cluster labels

    Returns:
    - df_out: DataFrame with added cluster column
    - model: trained KMedoids model
    """
    if features is None:
        features = ["P_rc_net", "T_air", "RH", "Wind_Speed"]

    df_subset = df.dropna(subset=features).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_subset[features])

    model = KMedoids(
        n_clusters=n_clusters, random_state=42, metric="euclidean", init="k-medoids++"
    )
    labels = model.fit_predict(X_scaled)

    df_out = df.copy()
    df_out[cluster_col] = -1
    df_out.loc[df_subset.index, cluster_col] = labels

    return df_out, model


def plot_overlay_rc_pv_zones(
    df,
    rc_col="RC_Cluster",
    tech_col="Best_Technology",
    lat_col="latitude",
    lon_col="longitude",
    output_path="results/maps/overlay_rc_pv_map.png",
):
    """
    Overlay RC-only clusters and matched PV technologies on the same map.

    Parameters:
    - df: DataFrame with coordinates, RC cluster, and tech columns
    - rc_col: column for RC cluster IDs
    - tech_col: column for matched PV technology
    - lat_col, lon_col: coordinate columns
    - output_path: file path to save the output image
    """
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Background: RC Clusters (bigger circles, soft alpha)
    gdf.plot(
        ax=ax,
        column=rc_col,
        cmap="Pastel1",
        markersize=50,
        alpha=0.6,
        legend=True,
        edgecolor="none",
        label="RC Cluster",
    )

    # Foreground: PV Technologies (cross markers)
    gdf.plot(
        ax=ax,
        column=tech_col,
        cmap="tab10",
        markersize=20,
        marker="x",
        legend=True,
        edgecolor="black",
        label="Best Tech",
    )

    try:
        ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite)
    except BaseException:
        print("⚠️ Basemap could not be loaded — offline mode.")

    ax.set_title("Overlay of RC Climate Zones and Optimal PV Technologies", fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Overlay map saved to: {output_path}")


def generate_zone_descriptions(
    df,
    cluster_col="Cluster_ID",
    rc_col="RC_potential",
    temp_col="T_air",
    red_col="Red_band",
    tech_col="Best_Technology",
):
    """
    Generate human-readable zone descriptions based on cluster characteristics.

    Parameters:
    - df: DataFrame with clustering and key features
    - cluster_col: name of the clustering column
    - rc_col: column for radiative cooling potential
    - temp_col: air temperature column
    - red_col: red spectral band column
    - tech_col: best PV technology column

    Returns:
    - zone_df: DataFrame with cluster ID and description
    """
    grouped = (
        df.groupby(cluster_col)
        .agg(
            {
                temp_col: "mean",
                rc_col: "mean",
                red_col: "mean",
                tech_col: lambda x: (
                    x.mode().iloc[0] if not x.mode().empty else "Unknown"
                ),
            }
        )
        .reset_index()
    )

    def describe(row):
        t = row[temp_col]
        rc = row[rc_col]
        red = row[red_col]
        tech = row[tech_col]

        temp_str = "Hot" if t > 18 else "Cool"
        rc_str = "High RC" if rc > 40 else "Low RC"
        red_str = "Red-rich" if red > 0.35 else "Red-poor"

        return f"{temp_str}, {rc_str}, {red_str} → {tech}"

    grouped["Zone_Description"] = grouped.apply(describe, axis=1)
    return grouped[[cluster_col, "Zone_Description"]]


def calculate_rc_power_improved(df, albedo=0.3, emissivity=0.95):
    """
    Compute net radiative cooling power with proper sky temperature calculation.
    """
    df_out = df.copy()

    # Use proper sky temperature calculation
    RH = df.get("RH", 50)
    TCC = df.get("TCC", 0)  # Cloud cover if available
    T_sky = calculate_sky_temperature_improved(df["T_air"], RH, TCC)

    σ = 5.67e-8  # Stefan-Boltzmann constant
    T_air_K = df["T_air"] + 273.15
    T_sky_K = T_sky + 273.15

    # Proper RC power calculation
    df_out["Q_rad"] = emissivity * σ * (T_air_K**4 - T_sky_K**4)
    df_out["Q_solar"] = (1 - albedo) * df["GHI"]
    df_out["P_rc_net"] = df_out["Q_rad"] - df_out["Q_solar"]

    return df_out


def sweep_rc_with_reflectivity(df, albedo_values=[0.3, 0.6, 1.0]):
    """
    Run RC power calculations across multiple albedo values.

    Returns:
    - combined_df: DataFrame with one row per (location × albedo)
    """
    results = []
    for alb in albedo_values:
        df_alb = calculate_rc_power_improved(df, albedo=alb)
        df_alb["Albedo"] = alb
        results.append(df_alb)

    return pd.concat(results, ignore_index=True)
