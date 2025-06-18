import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os 
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import ticker

from shapely.geometry import Point
import rasterio
from rasterio.plot import show as rio_show
from pyproj import Transformer
# Import humidity function
try:
    from utils.humidity import compute_relative_humidity
except ImportError:
    print("⚠️ utils.humidity not available, using local fallback")
    def compute_relative_humidity(T_air_K, T_dew_K):
        """Fallback humidity calculation"""
        T_air = np.array(T_air_K) - 273.15
        T_dew = np.array(T_dew_K) - 273.15
        a, b = 17.625, 243.04
        e_s = np.exp((a * T_air) / (b + T_air))
        e_d = np.exp((a * T_dew) / (b + T_dew))
        RH = 100.0 * (e_d / e_s)
        return np.clip(RH, 0, 100)
        
def prepare_clustered_dataset(input_path="data/clustered_dataset_rh.csv",
                               output_path="data/clustered_dataset_rh_albedo.csv",
                               db_url=None,
                               db_table=None):
    """
    Load dataset, recompute RH if needed, apply clustering, and save enhanced file.
    """
    if db_url:
        from database_utils import read_table
        df = read_table(db_table, db_url=db_url)
        print(f"📥 Loaded {len(df)} rows from table {db_table}")
    else:
        print(f"📥 Loading dataset from {input_path}")

        # Check if file exists, try alternatives
        if not os.path.exists(input_path):
            alternative_paths = ["clustered_dataset.csv", "data/clustered_dataset.csv", "merged_dataset.csv"]
            input_path = None
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    input_path = alt_path
                    print(f"📥 Using alternative input: {input_path}")
                    break

            if input_path is None:
                print(f"❌ No input file found. Tried: {alternative_paths}")
                return None

        df = pd.read_csv(input_path)
        print(f"📊 Loaded {len(df)} rows with columns: {list(df.columns)}")

    # Add albedo if available from GRIB data
    if "fal" in df.columns and "Albedo" not in df.columns:
        df["Albedo"] = df["fal"]
        print("✅ Added Albedo from 'fal' column")

    # Compute RH if missing
    if "RH" not in df.columns:
        print("🧮 Computing RH...")
        # Look for temperature and dewpoint columns with flexible naming
        temp_cols = [col for col in ['T_air', 'T2M', 'Temperature'] if col in df.columns]
        dew_cols = [col for col in ['Dew_Point', 'TD2M', 'Dewpoint', 'd2m'] if col in df.columns]
        
        if temp_cols and dew_cols:
            T_air_K = df[temp_cols[0]].values
            T_dew_K = df[dew_cols[0]].values
            
            # Convert to Kelvin if values appear to be in Celsius
            if T_air_K.mean() < 100:  # Likely Celsius
                T_air_K = T_air_K + 273.15
                T_dew_K = T_dew_K + 273.15
            
            df["RH"] = compute_relative_humidity(T_air_K, T_dew_K)
            print(f"✅ Computed RH using {temp_cols[0]} and {dew_cols[0]}")
        else:
            df["RH"] = 50.0  # Default value
            print("⚠️ Could not compute RH, using default value of 50%")

    # Flexible feature selection - check what's actually available
    potential_features = [
        "GHI", "T_air", "Wind_Speed", "RC_Potential",
        "Cloud_Cover", "Dew_Point", "RH", "Albedo"
    ]
    
    # Map potential features to actual column names
    feature_mapping = {}
    feature_options = {
        "GHI": ["GHI", "SSRD_power", "Global_Horizontal_Irradiance"],
        "T_air": ["T_air", "T2M", "Temperature"], 
        "Wind_Speed": ["Wind_Speed", "WS", "WindSpeed"],
        "RC_Potential": ["RC_Potential", "P_rc_net", "QNET"],
        "Cloud_Cover": ["Cloud_Cover", "TCC", "CloudCover"],
        "Dew_Point": ["Dew_Point", "TD2M", "Dewpoint", "d2m"],
        "RH": ["RH", "Relative_Humidity"],
        "Albedo": ["Albedo", "fal", "effective_albedo"]
    }
    
    # Find available features
    for feature, options in feature_options.items():
        for col in options:
            if col in df.columns:
                feature_mapping[feature] = col
                break
    
    available_features = list(feature_mapping.keys())
    feature_columns = [feature_mapping[f] for f in available_features]
    
    print(f"🎯 Using {len(available_features)} features: {available_features}")
    
    if len(available_features) < 3:
        print(f"❌ Not enough features ({len(available_features)}), need at least 3")
        return None

    # Prepare data for clustering
    X = df[feature_columns].copy()
    
    # Handle missing values
    missing_count = X.isnull().sum().sum()
    if missing_count > 0:
        print(f"🔧 Filling {missing_count} missing values with median")
        X = X.fillna(X.median())
    
    # Scale and apply clustering
    X_scaled = StandardScaler().fit_transform(X)
    model = KMeans(n_clusters=5, random_state=42)
    df["Cluster_ID"] = model.fit_predict(X_scaled)

    # Save enhanced dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    if db_url:
        from database_utils import write_dataframe
        write_dataframe(df, db_table, db_url=db_url, if_exists="replace")
        print(f"✅ Saved enhanced dataset to table {db_table}")
    print(f"✅ Saved enhanced dataset to {output_path}")
    return df


def run_kmedoids_clustering(X_scaled, n_clusters=5, metric='euclidean', random_state=42):
    kmedoids = KMedoids(n_clusters=n_clusters, metric=metric, init='k-medoids++', random_state=random_state)
    kmedoids.fit(X_scaled)
    labels = kmedoids.labels_
    silhouette = silhouette_score(X_scaled, labels)
    print(f"K-Medoids Silhouette Score: {silhouette:.4f}")
    return kmedoids, labels, silhouette

def assign_clusters_to_dataframe(df, labels, column_name='Cluster_ID'):
    df_out = df.copy()
    df_out[column_name] = labels
    return df_out

def prepare_features_for_clustering(df, use_predicted_pv=True):
    """
    Prepare features for clustering with flexible column detection.
    """
    print("🔧 Preparing features for clustering...")
    
    # Map feature names to possible column names in your data
    feature_options = {
        'GHI': ['GHI', 'SSRD_power', 'Global_Horizontal_Irradiance'],
        'T_air': ['T_air', 'T2M', 'Temperature'],
        'RC_potential': ['RC_potential', 'P_rc_net', 'QNET'],
        'Wind_Speed': ['Wind_Speed', 'WS', 'WindSpeed'],
        'Dew_Point': ['Dew_Point', 'TD2M', 'Dewpoint', 'd2m'],
        'Cloud_Cover': ['Cloud_Cover', 'TCC', 'CloudCover'],
        'Red_band': ['Red_band', 'Red_Band', 'Red'],
        'Blue_band': ['Blue_band', 'Blue_Band', 'Blue'],
        'IR_band': ['IR_band', 'NIR_band', 'IR_Band', 'IR'],
        'Total_band': ['Total_band', 'Total_Band'],
        'PV_Potential_physics': ['PV_Potential_physics', 'PV_Potential']
    }
    
    # Find available features
    feature_names = []
    for feature_name, possible_cols in feature_options.items():
        for col in possible_cols:
            if col in df.columns:
                feature_names.append(col)
                print(f"✅ Found {feature_name} as '{col}'")
                break
        else:
            print(f"⚠️ {feature_name} not found")
    
    # Add predicted PV potential if requested and available
    if use_predicted_pv and 'Predicted_PV_Potential' in df.columns:
        feature_names.append('Predicted_PV_Potential')
        print("✅ Added Predicted_PV_Potential")
    
    if len(feature_names) < 3:
        print(f"❌ Not enough features found ({len(feature_names)}). Need at least 3.")
        return None, [], None
    
    print(f"🎯 Using {len(feature_names)} features: {feature_names}")
    
    X = df[feature_names].copy()
    
    # Handle missing values
    missing_count = X.isnull().sum().sum()
    if missing_count > 0:
        print(f"🔧 Handling {missing_count} missing values")
        X = X.fillna(X.median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, feature_names, scaler

def find_optimal_k(X_scaled, k_range=range(2, 11), random_state=42):
    """
    Evaluate silhouette scores for a range of cluster values.

    Parameters:
    - X_scaled: standardized feature matrix
    - k_range: iterable of k values to test (default: 2 to 10)
    - random_state: seed for reproducibility

    Returns:
    - scores: dict of silhouette scores keyed by k
    - best_k: the k value with highest silhouette score
    """
    scores = {}
    for k in k_range:
        try:
            model = KMedoids(n_clusters=k, init='k-medoids++', random_state=random_state)
            model.fit(X_scaled)
            score = silhouette_score(X_scaled, model.labels_)
            scores[k] = score
            print(f"k = {k}: silhouette = {score:.4f}")
        except Exception as e:
            print(f"Failed for k = {k}: {e}")
    
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(list(scores.keys()), list(scores.values()), marker='o', linestyle='-')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Optimal Number of Clusters")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    best_k = max(scores, key=scores.get)
    print(f"\n✅ Best k based on silhouette score: {best_k} (score = {scores[best_k]:.4f})")
    return scores, best_k


def get_pv_cell_profiles():
    """
    Returns a dictionary of predefined PV cell technologies,
    each with spectral response weights and temperature coefficients.

    Returns:
    - pv_profiles (dict): PV cell types and their properties
    """
    pv_profiles = {
        "Silicon": {
            "spectral_response": {
                "Blue": 0.3,
                "Green": 0.7,
                "Red": 1.0,
                "IR": 1.0
            },
            "temperature_coefficient": -0.0045  # per °C
        },
        "Perovskite": {
            "spectral_response": {
                "Blue": 1.0,
                "Green": 0.9,
                "Red": 0.4,
                "IR": 0.1
            },
            "temperature_coefficient": -0.0025
        },
        "Tandem": {
            "spectral_response": {
                "Blue": 1.0,
                "Green": 1.0,
                "Red": 1.0,
                "IR": 1.0
            },
            "temperature_coefficient": -0.0035
        },
        "CdTe": {
            "spectral_response": {
                "Blue": 0.8,
                "Green": 0.9,
                "Red": 0.7,
                "IR": 0.2
            },
            "temperature_coefficient": -0.0028
        }
    }
    return pv_profiles

def compute_cluster_spectra(df_clustered, cluster_col='Cluster_ID'):
    """
    Computes average spectral band values and temperature for each cluster.
    """
    print("📊 Computing cluster spectra...")
    
    # Check which spectral and temperature columns are available
    spectral_col_options = {
        'Blue': ['Blue_band', 'Blue_Band', 'Blue'],
        'Green': ['Green_band', 'Green_Band', 'Green'], 
        'Red': ['Red_band', 'Red_Band', 'Red'],
        'IR': ['IR_band', 'NIR_band', 'IR_Band', 'IR']
    }
    
    spectral_cols = []
    spectral_mapping = {}
    
    for band, options in spectral_col_options.items():
        for col in options:
            if col in df_clustered.columns:
                spectral_cols.append(col)
                spectral_mapping[band] = col
                print(f"✅ Found {band} band as '{col}'")
                break
        else:
            print(f"⚠️ {band} band not found")
    
    if not spectral_cols:
        print("❌ No spectral columns found for cluster analysis")
        return pd.DataFrame()
    
    # Find temperature column
    temp_col = None
    for col in ['T_air', 'T2M', 'Temperature']:
        if col in df_clustered.columns:
            temp_col = col
            print(f"✅ Using temperature column: {col}")
            break
    
    if not temp_col:
        print("❌ No temperature column found")
        return pd.DataFrame()

    # Compute cluster averages
    grouped = df_clustered.groupby(cluster_col)
    cluster_spectra_df = grouped[spectral_cols + [temp_col]].mean().reset_index()

    # Normalize spectra to sum = 1 if we have multiple bands
    if len(spectral_cols) > 1:
        spectrum_sum = cluster_spectra_df[spectral_cols].sum(axis=1)
        for col in spectral_cols:
            cluster_spectra_df[col] = cluster_spectra_df[col] / spectrum_sum

    # Add standardized column names for compatibility
    for band, col in spectral_mapping.items():
        if col in cluster_spectra_df.columns:
            cluster_spectra_df[f'{band}_band'] = cluster_spectra_df[col]

    print(f"✅ Computed spectra for {len(cluster_spectra_df)} clusters")
    return cluster_spectra_df

def compute_pv_potential_by_cluster_year(df, 
                                         cluster_col='Cluster_ID',
                                         year_col='year',
                                         pv_col='Predicted_PV_Potential',
                                         output_path='results/pv_potential_by_cluster_year.csv'):
    """
    Aggregate total and mean PV potential per cluster per year.

    Parameters:
    - df: DataFrame with PV predictions and cluster info
    - cluster_col: name of the cluster column
    - year_col: name of the year column
    - pv_col: name of the predicted PV potential column
    - output_path: where to save the output CSV
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Ensure year column exists
    if year_col not in df.columns:
        if 'time' in df.columns:
            df[year_col] = pd.to_datetime(df['time'], errors='coerce').dt.year
        else:
            raise ValueError(f"Column '{year_col}' not found and no 'time' column to derive from.")

    grouped = df.groupby([cluster_col, year_col]).agg(
        Total_PV_Potential=(pv_col, 'sum'),
        Mean_PV_Potential=(pv_col, 'mean'),
        Location_Count=(pv_col, 'count')
    ).reset_index()

    grouped.to_csv(output_path, index=False)
    print(f"✅ Saved PV potential by cluster and year to {output_path}")
    return grouped


def match_technology_to_clusters(cluster_spectra_df, pv_profiles, temp_col='T_air'):
    """
    Match each cluster to the best PV technology based on spectral fit and temperature adjustment.

    Parameters:
    - cluster_spectra_df: DataFrame with per-cluster spectral values and average temperature
    - pv_profiles: Dictionary of PV technologies with spectral_response and temperature_coefficient

    Returns:
    - match_df: DataFrame with scores and best match per cluster
    """
    spectral_bands = ['Blue', 'Green', 'Red', 'IR']
    results = []

    for _, row in cluster_spectra_df.iterrows():
        cluster_id = row['Cluster_ID']
        cluster_temp = row[temp_col]

        tech_scores = {}

        for tech, profile in pv_profiles.items():
            score = 0.0

            # Spectral matching (dot product)
            for band in spectral_bands:
                band_key = band + '_band'
                score += row[band_key] * profile['spectral_response'][band]

            # Apply temperature penalty
            temp_coeff = profile['temperature_coefficient']
            temp_penalty = temp_coeff * (cluster_temp - 25)  # assume PR loss below/above 25°C
            adjusted_score = score + temp_penalty

            tech_scores[tech] = adjusted_score

        # Pick best match
        best_match = max(tech_scores, key=tech_scores.get)

        results.append({
            'Cluster_ID': cluster_id,
            'Best_Technology': best_match,
            **tech_scores  # Unpack all scores
        })

    match_df = pd.DataFrame(results)
    return match_df

def compute_adjusted_yield_by_technology(df, pv_profiles):
    """
    Compute adjusted yield for each PV technology per location using:
      Adjusted_Yield = Predicted_PV_Potential × (1 + SpectralMatch + TempGain)

    Parameters:
    - df: DataFrame with spectral bands, temperature, and predicted PV potential
    - pv_profiles: dictionary of PV technologies (spectral_response + temp_coefficient)

    Returns:
    - DataFrame with ['Location_ID', 'Technology', 'Adjusted_Yield']
    """
    spectral_bands = ['Blue', 'Green', 'Red', 'IR']
    output = []

    for idx, row in df.iterrows():
        location_id = row.get('location_id', idx)
        T_air = row['T_air']
        pred_pv = row['Predicted_PV_Potential']
        total_band = row['Total_band']

        for tech, props in pv_profiles.items():
            spectral_match = 0
            for band in spectral_bands:
                band_col = f"{band}_band"
                spectral_frac = row[band_col] / total_band if total_band > 0 else 0
                spectral_match += spectral_frac * props['spectral_response'][band]

            temp_gain = props['temperature_coefficient'] * (T_air - 25)
            adjusted_yield = pred_pv * (1 + spectral_match + temp_gain)

            output.append({
                "Location_ID": location_id,
                "Technology": tech,
                "Adjusted_Yield": adjusted_yield
            })

    return pd.DataFrame(output)

def label_clusters(df, cluster_col='Cluster_ID'):
    """
    Map numeric clusters to human-readable climate zone labels.
    
    Parameters:
    - df: DataFrame with cluster IDs
    - cluster_col: name of column holding cluster labels
    
    Returns:
    - df with new column 'Cluster_Label'
    """
    cluster_label_map = {
        0: "Hot & Sunny",
        1: "Cool & Diffuse",
        2: "High RC & Low Temp",
        3: "Mountainous, Low IR",
        4: "Temperate High PV"
        # Add more based on silhouette results or feature means
    }
    
    df['Cluster_Label'] = df[cluster_col].map(cluster_label_map).fillna("Unlabeled")
    return df

def compute_cluster_summary(df, cluster_col='Cluster_ID', output_path='results/cluster_summary.csv'):
    """
    Compute summary statistics for each cluster and save to CSV.

    Parameters:
    - df: DataFrame with climate, PV, and cluster columns
    - cluster_col: column with cluster ID
    - output_path: where to save the CSV
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    summary = df.groupby(cluster_col).agg({
        'T_air': 'mean',
        'RH': 'mean',
        'Albedo': 'mean',
        'Predicted_PV_Potential': 'mean',
        'Cluster_Label': 'first'  # assuming label is consistent within each cluster
    }).reset_index()

    summary = summary.rename(columns={
        'T_air': 'Mean_T_air',
        'RH': 'Mean_RH',
        'Albedo': 'Mean_Albedo',
        'Predicted_PV_Potential': 'Mean_PV_Potential'
    })

    summary.to_csv(output_path, index=False)
    print(f"✅ Saved cluster summary to {output_path}")
    return summary

def plot_prediction_uncertainty_with_contours(
    df,
    lat_col='latitude',
    lon_col='longitude',
    value_col='Prediction_Uncertainty',
    output_path='results/maps/prediction_uncertainty_map_contours.png',
    contour_levels=10,
    use_hatching=False
):
    """
    Plot map of prediction uncertainty with optional contours or hatching.

    Parameters:
    - df: DataFrame with location and uncertainty data
    - lat_col, lon_col: name of coordinate columns
    - value_col: column with uncertainty values
    - output_path: path to save PNG
    - contour_levels: number of contour levels
    - use_hatching: if True, adds hatching overlay
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if value_col not in df.columns:
        print(f"⚠️ Missing column: {value_col}")
        return

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df[lon_col], df[lat_col])],
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(14, 10))
    scatter = gdf.plot(
        ax=ax,
        column=value_col,
        cmap='coolwarm',
        legend=True,
        markersize=35,
        edgecolor='black',
        alpha=0.8
    )

    # Add basemap
    try:
        ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite)
    except Exception as e:
        print("Basemap not available – offline mode.")

    ax.set_title("Prediction Uncertainty Map", fontsize=16)
    ax.set_axis_off()

    # Contours or hatching
    if use_hatching or contour_levels:
        from scipy.interpolate import griddata

        # Create grid for interpolation
        x = gdf.geometry.x
        y = gdf.geometry.y
        z = gdf[value_col]
        xi = np.linspace(x.min(), x.max(), 300)
        yi = np.linspace(y.min(), y.max(), 300)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x, y), z, (xi, yi), method='cubic')

        if use_hatching:
            ax.contour(xi, yi, zi, levels=contour_levels, linewidths=0, colors='none', hatches=['///'], alpha=0.4)
        else:
            cs = ax.contour(xi, yi, zi, levels=contour_levels, colors='black', linewidths=0.7)
            ax.clabel(cs, inline=1, fontsize=8, fmt=ticker.FuncFormatter(lambda x, _: f"{x:.2f}"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved uncertainty map with contours to {output_path}")

def plot_technology_matches(df_clustered, match_df, lat_col='latitude', lon_col='longitude', cluster_col='Cluster_ID'):
    """
    Plot best PV technology for each location on a geographic map.
    """
    print("🗺️ Creating technology matching map...")
    
    # Check if coordinate columns exist with flexible naming
    coord_mapping = {}
    
    # Find latitude column
    lat_options = ['latitude', 'lat', 'LAT']
    for col in lat_options:
        if col in df_clustered.columns:
            coord_mapping['lat'] = col
            break
    
    # Find longitude column  
    lon_options = ['longitude', 'lon', 'LON']
    for col in lon_options:
        if col in df_clustered.columns:
            coord_mapping['lon'] = col
            break
    
    if len(coord_mapping) < 2:
        print("⚠️ Cannot create map - missing coordinate columns")
        print(f"Available columns: {list(df_clustered.columns)}")
        return
    
    lat_col = coord_mapping['lat']
    lon_col = coord_mapping['lon']
    print(f"✅ Using coordinates: {lat_col}, {lon_col}")
    
    # Merge technology data
    df_merged = df_clustered.merge(match_df[[cluster_col, 'Best_Technology']], 
                                  on=cluster_col, how='left')

    try:
        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df_merged,
            geometry=gpd.points_from_xy(df_merged[lon_col], df_merged[lat_col]),
            crs="EPSG:4326"
        ).to_crs(epsg=3857)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        gdf.plot(
            ax=ax,
            column='Best_Technology',
            cmap='tab20',
            legend=True,
            markersize=30,
            edgecolor='k',
            alpha=0.8
        )

        # Add basemap if possible
        try:
            ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite)
        except Exception as e:
            print("⚠️ Basemap not available - continuing without")

        ax.set_title("Best Matched PV Technology by Location", fontsize=15)
        ax.set_axis_off()
        
        # Save plot
        os.makedirs("results/maps", exist_ok=True)
        plt.savefig("results/maps/technology_matches.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Technology matching map saved to results/maps/technology_matches.png")
        
    except Exception as e:
        print(f"⚠️ Could not create map: {e}")
        
def plot_clusters_map(df_clustered, lat_col='latitude', lon_col='longitude',
                      cluster_col='Cluster_ID', title='PV Performance Clusters',
                      borders_path='data/borders/ne_10m_admin_0_countries.shp',
                      eu_only=True):
    """
    Plot clusters over EU map with country borders, excluding sea areas.

    Parameters:
    - df_clustered: DataFrame with geographic and cluster info
    - lat_col, lon_col: Column names for lat/lon
    - cluster_col: Column name for cluster ID
    - title: Title for the plot
    - borders_path: Path to shapefile with country polygons
    - eu_only: If True, filter to EU countries only

    Returns:
    - None (shows map)
    """
    # 1. Load EU country borders
    borders = gpd.read_file(borders_path)
    if eu_only:
        eu_countries = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic',
                        'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary',
                        'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta',
                        'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia',
                        'Spain', 'Sweden']
        borders = borders[borders['ADMIN'].isin(eu_countries)]

    borders = borders.to_crs(epsg=3857)

    # 2. Convert data to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df_clustered,
        geometry=gpd.points_from_xy(df_clustered[lon_col], df_clustered[lat_col]),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    # 3. Clip points to land areas
    gdf_clipped = gpd.sjoin(gdf, borders, how="inner", predicate='intersects')

    # 4. Plot
    fig, ax = plt.subplots(figsize=(12, 9))
    gdf_clipped.plot(
        ax=ax,
        column=cluster_col,
        cmap='tab10',
        legend=True,
        markersize=30,
        alpha=0.8,
        edgecolor='black'
    )

    # 5. Add borders and basemap
    borders.boundary.plot(ax=ax, color='gray', linewidth=1)
    try:
        ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite)
    except Exception as e:
        print("Basemap not loaded — offline mode.")

    ax.set_title(title, fontsize=16)
    ax.set_axis_off()
    plt.show()

def plot_prediction_uncertainty(df, lat_col='latitude', lon_col='longitude', output_path='results/maps/prediction_uncertainty_map.png'):
    """
    Plot map of prediction uncertainty from Random Forest model.

    Parameters:
    - df: DataFrame with 'Prediction_Uncertainty' and location data
    - lat_col, lon_col: column names for coordinates
    - output_path: where to save the PNG file
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if 'Prediction_Uncertainty' not in df.columns:
        print("⚠️ No 'Prediction_Uncertainty' column found. Skipping plot.")
        return

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326").to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(14, 10))
    gdf.plot(ax=ax, column='Prediction_Uncertainty', cmap='coolwarm', legend=True, edgecolor='black', markersize=30)

    try:
        ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite)
    except Exception as e:
        print("Basemap not loaded — offline mode.")

    ax.set_title("Random Forest Prediction Uncertainty Map", fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"🖼️ Saved prediction uncertainty map to: {output_path}")


def add_koppen_geiger(df, kg_raster='DATASET/kg_classification.tif', lat_col='latitude', lon_col='longitude'):
    """Attach Köppen–Geiger climate classification to each row."""
    df_out = df.copy()
    if not os.path.exists(kg_raster):
        print(f"⚠️ KG raster not found at {kg_raster}")
        df_out['KG_Code'] = np.nan
        df_out['KG_Label'] = np.nan
        return df_out

    coords = list(zip(df_out[lon_col], df_out[lat_col]))
    with rasterio.open(kg_raster) as src:
        if src.crs and src.crs.to_string() != 'EPSG:4326':
            transformer = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
            coords = [transformer.transform(x, y) for x, y in coords]
        values = [val[0] if val.size > 0 else np.nan for val in src.sample(coords)]

    df_out['KG_Code'] = values

    kg_lookup = {
        1: 'Af', 2: 'Am', 3: 'Aw', 4: 'BWh', 5: 'BWk', 6: 'BSh', 7: 'BSk',
        8: 'Csa', 9: 'Csb', 10: 'Csc', 11: 'Cwa', 12: 'Cwb', 13: 'Cwc',
        14: 'Cfa', 15: 'Cfb', 16: 'Cfc', 17: 'Dsa', 18: 'Dsb', 19: 'Dsc',
        20: 'Dsd', 21: 'Dwa', 22: 'Dwb', 23: 'Dwc', 24: 'Dwd', 25: 'Dfa',
        26: 'Dfb', 27: 'Dfc', 28: 'Dfd', 29: 'ET', 30: 'EF'
    }
    df_out['KG_Label'] = df_out['KG_Code'].map(kg_lookup)
    return df_out


def plot_clusters_with_kg(df, kg_raster='DATASET/kg_classification.tif', lat_col='latitude', lon_col='longitude',
                          cluster_col='Cluster_ID', output_path='figures/cluster_map_with_kg.png'):
    """Overlay cluster IDs with Köppen–Geiger zones on a map."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not os.path.exists(kg_raster):
        print(f"⚠️ KG raster not found at {kg_raster}")
        return
    with rasterio.open(kg_raster) as src:
        fig, ax = plt.subplots(figsize=(12, 8))
        rio_show(src, ax=ax, cmap='terrain', alpha=0.4)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs='EPSG:4326')
        gdf = gdf.to_crs(src.crs)
        gdf.plot(ax=ax, column=cluster_col, cmap='tab10', legend=True, markersize=30, edgecolor='black')
        ax.set_title('Clusters with Köppen–Geiger Zones')
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"✅ Saved map with KG overlay to {output_path}")

def main_matching_pipeline(
    clustered_data_path='data/clustered_dataset_rh_albedo.csv',
    output_file='matched_dataset.csv',
    k_range=range(2, 10),
    db_url=None,
    db_table=None,
):
    """
    Full pipeline to assign best PV technology per location based on clustering
    + spectral analysis. The final mapping no longer requires a shapefile path.
    """
    print("\n=== PV Technology Matching Pipeline ===")
    
    # Create output directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/maps", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Step 1: Prepare dataset
    print("\n=== Step 1: Preparing Input Dataset ===")
    df = prepare_clustered_dataset(db_url=db_url, db_table=db_table)
    if df is None:
        print("❌ Failed to prepare dataset")
        return None

    # --- Load it ---
    print("\n=== Loading Input Dataset ===")
    if db_url:
        from database_utils import read_table
        df = read_table(db_table, db_url=db_url)
    else:
        df = pd.read_csv(clustered_data_path)

    print("\n=== Preparing Features for Clustering ===")
    X_scaled, features, scaler = prepare_features_for_clustering(df)

    print("\n=== Finding Optimal Number of Clusters ===")
    _, best_k = find_optimal_k(X_scaled, k_range=k_range)

    print(f"\n=== Running K-Medoids Clustering with k = {best_k} ===")
    kmedoids, labels, silhouette = run_kmedoids_clustering(X_scaled, n_clusters=best_k)

    print("\n=== Assigning Clusters to Data ===")
    df_clustered = assign_clusters_to_dataframe(df, labels)

    print("\n=== Loading PV Cell Profiles from CSV ===")
    pv_profiles = get_pv_cell_profiles()

    print("\n=== Computing Cluster-Average Spectral & Temperature ===")
    cluster_spectra = compute_cluster_spectra(df_clustered)

    print("\n=== Matching PV Technologies to Clusters ===")
    match_df = match_technology_to_clusters(cluster_spectra, pv_profiles)
    
    df_clustered = assign_clusters_to_dataframe(df, labels)
    df_clustered = label_clusters(df_clustered)

    print("\n=== Merging Technology Matches to Locations ===")
    df_final = df_clustered.merge(match_df[['Cluster_ID', 'Best_Technology']], on='Cluster_ID', how='left')
    

    df_final.to_csv(output_file, index=False)
    if db_url:
        from database_utils import write_dataframe
        write_dataframe(df_final, db_table, db_url=db_url, if_exists="replace")
    print(f"✅ Technology-matched dataset saved to {output_file}")
    # Compute adjusted yield for all PV techs at each location
    adjusted_yield_df = compute_adjusted_yield_by_technology(df_final, pv_profiles)
    plot_prediction_uncertainty_with_contours(df_final, use_hatching=False)
    #plot_prediction_uncertainty_with_contours(df_final, use_hatching=True)

    # Optional: Save to CSV
    adjusted_yield_df.to_csv("results/adjusted_yield_table.csv", index=False)
    print("✅ Adjusted PV yield table saved to results/adjusted_yield_table.csv")
    
    df_final = df_final.merge(df_clustered[['Cluster_ID', 'Cluster_Label']], on='Cluster_ID', how='left')

    print("\n=== Plotting Final Technology Recommendation Map ===")
    # plot_technology_matches no longer requires a shapefile path
    plot_technology_matches(df_final, match_df, cluster_col='Cluster_ID')
    plot_prediction_uncertainty(df_final, output_path='results/maps/prediction_uncertainty_map.png')
    
    compute_cluster_summary(df_final)
    compute_pv_potential_by_cluster_year(df_final)

    # --- Köppen–Geiger integration ---
    try:
        df_with_kg = add_koppen_geiger(df_final)
        df_with_kg.to_csv('clustered_dataset_with_kg.csv', index=False)
        plot_clusters_with_kg(df_with_kg, output_path='figures/cluster_map_with_kg.png')
        print('✅ Köppen–Geiger enrichment complete')
    except Exception as e:
        print(f'⚠️ KG enrichment failed: {e}')

    return df_final

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run PV technology matching pipeline")
    parser.add_argument("--input-file", default="data/clustered_dataset_rh_albedo.csv")
    parser.add_argument("--output-file", default="matched_dataset.csv")
    parser.add_argument("--db-url", default=os.getenv("PV_DB_URL"))
    parser.add_argument("--db-table", default=os.getenv("PV_DB_TABLE", "pv_data"))
    args = parser.parse_args()
    main_matching_pipeline(
        clustered_data_path=args.input_file,
        output_file=args.output_file,
        db_url=args.db_url,
        db_table=args.db_table,
    )
