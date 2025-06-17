# rc_climate_zoning.py

import os

import contextily as ctx
import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import sigma as σ
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids

# Constants
DEFAULT_ALBEDO = 0.3
DEFAULT_EMISSIVITY = 0.95
GHI_THRESHOLD = 20  # W/m²


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


def calculate_day_night_rc_split(
    df, albedo=DEFAULT_ALBEDO, emissivity=DEFAULT_EMISSIVITY
):
    """
    Compute total, daytime, and nighttime radiative cooling power for each row.
    Expects columns: T_air (°C), GHI (W/m²), RH (%), TCC (0-1, optional)
    Returns DataFrame with new columns: P_rc_net, P_rc_day, P_rc_night
    """
    df = df.copy()
    T_air = df["T_air"]

    # Use proper sky temperature calculation
    RH = df.get("RH", 50)  # Default RH if not available
    TCC = df.get("TCC", 0)  # Total cloud cover if available
    T_sky = calculate_sky_temperature_improved(T_air, RH, TCC)

    T_air_K = T_air + 273.15
    T_sky_K = T_sky + 273.15

    Q_rad = emissivity * σ * (T_air_K**4 - T_sky_K**4)
    Q_solar = (1 - albedo) * df["GHI"]

    df["P_rc_net"] = Q_rad - Q_solar
    df["is_night"] = df["GHI"] <= GHI_THRESHOLD
    df["P_rc_day"] = df["P_rc_net"].where(~df["is_night"], 0)
    df["P_rc_night"] = df["P_rc_net"].where(df["is_night"], 0)
    return df


def rc_only_clustering(df, features=None, n_clusters=5, cluster_col="RC_Cluster"):
    """
    Cluster based only on RC and thermal features using K-Medoids.
    """
    if features is None:
        features = ["P_rc_net", "T_air", "RH", "Wind_Speed"]

    df_subset = df.dropna(subset=features).copy()
    X_scaled = StandardScaler().fit_transform(df_subset[features])
    model = KMedoids(n_clusters=n_clusters, random_state=42).fit(X_scaled)

    df_out = df.copy()
    df_out[cluster_col] = -1
    df_out.loc[df_subset.index, cluster_col] = model.labels_
    return df_out, model


def generate_zone_descriptions(
    df,
    cluster_col="RC_Cluster",
    rc_col="P_rc_net",
    temp_col="T_air",
    red_col="Red_band",
    tech_col="Best_Technology",
):
    """
    Generate text labels like: 'Cool, High RC, Red-rich → Tandem'
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
        temp_str = "Hot" if row[temp_col] > 18 else "Cool"
        rc_str = "High RC" if row[rc_col] > 40 else "Low RC"
        red_str = "Red-rich" if row[red_col] > 0.35 else "Red-poor"
        return f"{temp_str}, {rc_str}, {red_str} → {row[tech_col]}"

    grouped["Zone_Description"] = grouped.apply(describe, axis=1)
    return grouped[[cluster_col, "Zone_Description"]]


def plot_overlay_rc_pv_zones(
    df,
    rc_col="RC_Cluster",
    tech_col="Best_Technology",
    lat_col="latitude",
    lon_col="longitude",
    output_path="results/maps/overlay_rc_pv_map.png",
):
    """
    Plot a dual-layer overlay map showing RC clusters (background) and best PV tech (foreground).
    """
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Background: RC cluster colors
    gdf.plot(
        ax=ax,
        column=rc_col,
        cmap="Pastel1",
        markersize=50,
        legend=True,
        alpha=0.6,
        edgecolor="none",
    )

    # Foreground: PV technology with crosses
    gdf.plot(
        ax=ax,
        column=tech_col,
        cmap="tab10",
        markersize=20,
        marker="x",
        legend=True,
        edgecolor="k",
    )

    try:
        ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite)
    except Exception as e:
        print(f"⚠️ Could not load basemap: {e}")

    ax.set_title("RC Climate Zones with PV Technology Matches", fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Overlay map saved: {output_path}")


def run_rc_zoning_pipeline(
    input_csv,
    output_csv="results/rc_zones_described.csv",
    overlay_map="results/maps/overlay_rc_pv_map.png",
    n_clusters=5,
    db_url=None,
    db_table=None,
):
    """
    Full pipeline to generate RC climate zones, zone descriptions,
    and overlay maps from a PV + RC dataset.

    Parameters:
    - input_csv: CSV file with RC, GHI, T_air, Wind, RH, Red_band, and Best_Technology
    - output_csv: Where to save enriched output with RC clusters and descriptions
    - overlay_map: Path to save the final map
    - n_clusters: Number of RC clusters to use
    """
    if db_url:
        from database_utils import read_table, write_dataframe

        print(f"📥 Loading table {db_table}")
        df = read_table(db_table, db_url=db_url)
    else:
        print(f"📥 Loading: {input_csv}")
        df = pd.read_csv(input_csv)

    # Calculate RC components
    print("⚙️ Calculating day vs. night RC power...")
    df = calculate_day_night_rc_split(df)

    # RC-only clustering
    print("🔗 Running RC-only clustering...")
    df, rc_model = rc_only_clustering(df, n_clusters=n_clusters)
    # Save KMedoids model to disk
    model_save_path = "results/models/rc_kmedoids_model.joblib"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(rc_model, model_save_path)
    print(f"💾 Saved RC clustering model to: {model_save_path}")

    # Directory for results
    model_dir = "results/models"
    os.makedirs(model_dir, exist_ok=True)

    # Compute silhouette score (only if >1 cluster)
    if df["RC_Cluster"].nunique() > 1:
        features = ["P_rc_net", "T_air", "RH", "Wind_Speed"]
        X = df[features].dropna()
        score = silhouette_score(
            StandardScaler().fit_transform(X), df.loc[X.index, "RC_Cluster"]
        )
        print(f"📊 Silhouette score for RC clustering: {score:.3f}")
    else:
        score = -1
        print("⚠️ Only one cluster found — silhouette score not applicable.")

    # Save model with versioned filename
    model_filename = f"rc_kmedoids_model_k{n_clusters}_s{int(score*1000)}.joblib"
    model_path = os.path.join(model_dir, model_filename)
    joblib.dump(rc_model, model_path)
    print(f"💾 Saved RC clustering model to: {model_path}")

    # Save labels CSV (optional but useful for overlay/debugging)
    label_csv_path = "results/clusters/rc_cluster_labels.csv"
    df[["latitude", "longitude", "RC_Cluster"]].dropna().to_csv(
        label_csv_path, index=False
    )
    print(f"📁 Cluster labels saved to: {label_csv_path}")

    # Zone descriptions
    print("🧠 Generating zone labels...")
    descriptions = generate_zone_descriptions(df, cluster_col="RC_Cluster")
    df = df.merge(descriptions, on="RC_Cluster", how="left")

    # Save enriched CSV
    df.to_csv(output_csv, index=False)
    if db_url:
        write_dataframe(df, db_table, db_url=db_url, if_exists="replace")
        print(f"💾 Output written to table {db_table}")
    print(f"💾 Output saved to: {output_csv}")

    # Overlay map
    print("🗺️ Generating overlay map...")
    plot_overlay_rc_pv_zones(df, output_path=overlay_map)

    print("✅ Pipeline complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RC climate zoning pipeline")
    parser.add_argument("--input", default="input.csv", help="Path to input CSV")
    parser.add_argument("--db-url", default=os.getenv("PV_DB_URL"))
    parser.add_argument("--db-table", default=os.getenv("PV_DB_TABLE", "pv_data"))
    args = parser.parse_args()
    run_rc_zoning_pipeline(args.input, db_url=args.db_url, db_table=args.db_table)
