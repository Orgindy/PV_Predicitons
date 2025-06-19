import os
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as ctx
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import get_path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, silhouette_score
from sklearn_extra.cluster import KMedoids

def add_land_mask(df, lat_col='latitude', lon_col='longitude', world_shapefile='naturalearth_lowres'):
    """
    Remove ocean/water points using Natural Earth country borders.

    Parameters:
    - df: DataFrame with coordinate columns.
    - lat_col, lon_col: column names for latitude and longitude.
    - world_shapefile: GeoPandas built-in dataset or path to a land polygon shapefile.

    Returns:
    - GeoDataFrame with only land points.
    """
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs='EPSG:4326')

    # Load land polygons (can be replaced with a high-res file)
    world = gpd.read_file(gpd.datasets.get_path(world_shapefile))

    # Spatial join: keep only points that intersect land
    gdf_land = gpd.sjoin(gdf, world, predicate='intersects', how='inner')
    gdf_land.drop(columns=['index_right'], inplace=True)

    print(f"🌍 Land mask applied: {len(gdf_land)} / {len(df)} points retained")

    return gdf_land

def overlay_technology_matches(geo_df, tech_col='Best_Technology', cluster_col='Cluster_ID',
                                title='Optimal PV Technologies by Location',
                                output_path='results/maps/pv_technology_map.png'):
    """
    Plot map of matched PV technologies by location.

    Parameters:
    - geo_df: GeoDataFrame with technology match and coordinates.
    - tech_col: column containing matched PV technologies.
    - cluster_col: optional cluster ID column for styling.
    - title: plot title.
    - output_path: file path to save PNG output.
    """
    if not isinstance(geo_df, gpd.GeoDataFrame):
        raise ValueError("Input must be a GeoDataFrame with geometry column.")
    if geo_df.crs is None or geo_df.crs.to_epsg() != 4326:
        raise ValueError("GeoDataFrame must use geographic CRS EPSG:4326")

    geo_df = geo_df.to_crs(epsg=3857)  # Web Mercator for plotting with basemaps

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.set(style="white")

    tech_palette = sns.color_palette("tab10", n_colors=geo_df[tech_col].nunique())

    geo_df.plot(column=tech_col, ax=ax, legend=True, markersize=35, edgecolor='black', cmap='tab10')
    try:
        ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite)
    except Exception as e:
        print("⚠️ Basemap could not be loaded — continuing without.")

    ax.set_title(title, fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"🖼️ Saved PV technology overlay map to: {output_path}")

def export_geojson(df, output_path='results/maps/final_clustered_map.geojson',
                   lat_col='latitude', lon_col='longitude', crs_epsg=4326):
    """
    Export a DataFrame with latitude and longitude to a GeoJSON file.

    Parameters:
    - df: DataFrame or GeoDataFrame containing clustered and matched tech data.
    - output_path: full path to save .geojson file.
    - lat_col, lon_col: name of columns with coordinates.
    - crs_epsg: EPSG code for geographic CRS (default is WGS84).
    """
    import geopandas as gpd
    from shapely.geometry import Point
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if 'geometry' not in df.columns:
        gdf = gpd.GeoDataFrame(df.copy(), 
                               geometry=[Point(xy) for xy in zip(df[lon_col], df[lat_col])],
                               crs=f"EPSG:{crs_epsg}")
    else:
        gdf = df if isinstance(df, gpd.GeoDataFrame) else gpd.GeoDataFrame(df)

    gdf.to_file(output_path, driver='GeoJSON')
    print(f"📤 Exported GeoJSON to: {output_path}")

from pv_potential import calculate_pv_potential

def prepare_features_for_ml(df):
    # Check if PV_Potential exists, if not, calculate it
    if 'PV_Potential' not in df.columns:
        print("📊 Calculating PV_Potential using improved physics model...")
        df['PV_Potential'] = calculate_pv_potential(
            df['GHI'].values,
            df['T_air'].values,
            df['RC_potential'].values,
            df['Red_band'].values,
            df['Total_band'].values
        )
    
    feature_names = [
        'GHI', 'T_air', 'RC_potential', 'Wind_Speed',
        'Dew_Point', 'Cloud_Cover', 'Red_band',
        'Blue_band', 'IR_band', 'Total_band'
    ]
    
    # Check which features are actually available
    available_features = [f for f in feature_names if f in df.columns]
    missing_features = [f for f in feature_names if f not in df.columns]
    
    if missing_features:
        print(f"⚠️ Missing features: {missing_features}")
        print(f"✅ Using available features: {available_features}")
        feature_names = available_features
    
    X = df[feature_names].copy()
    
    # Handle NaN values
    print(f"🔧 Handling {X.isnull().sum().sum()} NaN values...")
    X = X.fillna(X.median())
    
    y = df['PV_Potential']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, feature_names, scaler


def train_random_forest(X_scaled, y, feature_names, test_size=0.2, random_state=42, n_estimators=200, max_depth=12):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")

    # Feature importance plot
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(data=importances, x='importance', y='feature', palette='viridis')
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.show()

    return model, X_train, X_test, y_train, y_test, y_pred

def predict_pv_potential(model, X_scaled, df_original):
    """
    Makes predictions using the trained model and adds them to the original dataframe.
    
    Parameters:
    model - Trained ML model
    X_scaled - Scaled features for prediction (should be transformed with the same scaler used during training)
    df_original - Original dataframe to add predictions to
    
    Returns:
    DataFrame with added predictions
    """
    predictions = model.predict(X_scaled)
    df_result = df_original.copy()
    df_result['Predicted_PV_Potential'] = predictions
    
    # Optional: Add prediction confidence if model supports it (Random Forest does)
    try:
        pred_std = np.std([tree.predict(X_scaled) for tree in model.estimators_], axis=0)
        df_result['Prediction_Uncertainty'] = pred_std
    except Exception as e:
        pass
        
    return df_result

def prepare_features_for_clustering(df, use_predicted_pv=True):
    feature_names = [
        'GHI', 'T_air', 'RC_potential', 'Wind_Speed',
        'Dew_Point', 'Cloud_Cover', 'Red_band',
        'Blue_band', 'IR_band', 'Total_band', 'PV_Potential_physics'
    ]
    if use_predicted_pv and 'Predicted_PV_Potential' in df.columns:
        feature_names.append('Predicted_PV_Potential')
    X = df[feature_names].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, feature_names, scaler

def find_optimal_k(X_scaled, k_range=range(2, 11)):
    from sklearn_extra.cluster import KMedoids
    from sklearn.metrics import silhouette_score

    scores = {}
    for k in k_range:
        kmedoids = KMedoids(n_clusters=k, random_state=42).fit(X_scaled)
        score = silhouette_score(X_scaled, kmedoids.labels_)
        scores[k] = score
        print(f"k={k}, silhouette={score:.4f}")
    
    return scores


def run_kmedoids_clustering(X_scaled, n_clusters=4, metric='euclidean', random_state=42):
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

def plot_clusters_map(df, lat_col='latitude', lon_col='longitude', cluster_col='Cluster_ID', title='PV Performance Clusters'):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326").to_crs(epsg=3857)
    fig, ax = plt.subplots(figsize=(12, 8))
    gdf.plot(ax=ax, column=cluster_col, cmap='tab10', legend=True, markersize=35, edgecolor='k')
    try:
        ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite)
    except Exception as e:
        print("Basemap could not be loaded.")
    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

# -----------------------------
# PV Technology Mapping
# -----------------------------

def plot_technology_matches(df_clustered, match_df, lat_col='latitude', lon_col='longitude', cluster_col='Cluster_ID'):
    # Merge best tech into location DataFrame
    df_merged = df_clustered.merge(match_df[['Cluster_ID', 'Best_Technology']], on=cluster_col, how='left')
    gdf = gpd.GeoDataFrame(
        df_merged,
        geometry=gpd.points_from_xy(df_merged[lon_col], df_merged[lat_col]),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    # Plot
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

    try:
        ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite)
    except Exception as e:
        print("Basemap not loaded — offline mode.")

    ax.set_title("Best Matched PV Technology by Location", fontsize=15)
    ax.set_axis_off()
    plt.show()


def main_clustering_pipeline(input_file='merged_dataset.csv', output_file='clustered_dataset.csv', n_clusters=5):
    df = pd.read_csv(input_file)
    
    print("Calculating physics-based PV Potential...")
    df['PV_Potential_physics'] = calculate_pv_potential(
        df['GHI'].values,
        df['T_air'].values,
        df['RC_potential'].values,
        df['Red_band'].values,
        df['Total_band'].values
    )
    
    # Use physics-based calculation as target if PV_Potential doesn't exist
    if 'PV_Potential' not in df.columns:
        df['PV_Potential'] = df['PV_Potential_physics']
    
    print("Preparing features for Random Forest...")
    X_scaled, y, feature_names, scaler_rf = prepare_features_for_ml(df)
    
    print("Training Random Forest...")
    model, X_train, X_test, y_train, y_test, y_pred = train_random_forest(X_scaled, y, feature_names)
    
    print("Predicting PV Potential...")
    # Use the model to predict on the entire dataset - proper practice would be to only predict on unseen data
    # but for clustering purposes we want predictions for all locations
    df_with_pred = predict_pv_potential(model, X_scaled, df)
    
    # Optional: Add comparison between physics and ML models
    print("Comparing Physics-based and ML models...")
    correlation = np.corrcoef(df_with_pred['PV_Potential_physics'], df_with_pred['Predicted_PV_Potential'])[0, 1]
    print(f"Correlation between physics and ML predictions: {correlation:.4f}")
    

    print("Preparing features for clustering...")
    X_cluster_scaled, cluster_feats, scaler_cluster = prepare_features_for_clustering(df_with_pred)
    
    print("Running K-Medoids clustering...")
    kmedoids, labels, silhouette = run_kmedoids_clustering(X_cluster_scaled, n_clusters=n_clusters)
    
    print("Assigning cluster labels...")
    df_clustered = assign_clusters_to_dataframe(df_with_pred, labels)
    
    # Add cluster centers information
    cluster_centers = pd.DataFrame(
        scaler_cluster.inverse_transform(kmedoids.cluster_centers_),
        columns=cluster_feats
    )
    print("Cluster centers characteristics:")
    print(cluster_centers)
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df_clustered.to_csv(output_file, index=False)
    print(f"Saved clustered data to {output_file}")
    
    print("Plotting clusters on map...")
    plot_clusters_map(df_clustered)
    
    return df_clustered

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


def plot_overlay_rc_pv_zones(df, rc_col='RC_Cluster', tech_col='Best_Technology', output_path='results/maps/rc_pv_overlay.png'):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import contextily as ctx

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326").to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(14, 10))
    gdf.plot(ax=ax, column=rc_col, cmap='Pastel1', markersize=10, legend=True, alpha=0.6, edgecolor='none')

    gdf.plot(ax=ax, column=tech_col, cmap='tab10', markersize=6, legend=True, alpha=1, edgecolor='k', marker='x')

    ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite)
    ax.set_title("Overlay: RC Clusters and Best PV Technologies", fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description='PV Potential Analysis and Clustering')
    parser.add_argument('--input', default=get_path('merged_data_path'), help='Input CSV file path')
    parser.add_argument('--output', default=os.path.join(get_path('results_path'), 'clustered_dataset.csv'), help='Output CSV file path')
    parser.add_argument('--clusters', type=int, default=5, help='Number of clusters for K-Medoids')
    parser.add_argument('--landmask', action='store_true', help='Apply land mask and export GeoJSON')

    args = parser.parse_args()

    # Optional pre-processing
    if args.landmask:
        clustered_df = pd.read_csv(args.output)
        land_df = add_land_mask(clustered_df)
        results_dir = os.path.join(get_path('results_path'), 'clusters')
        os.makedirs(results_dir, exist_ok=True)
        land_df.to_file(os.path.join(results_dir, 'clustered_land_only.geojson'), driver='GeoJSON')

    # Main pipeline
    final_df = main_clustering_pipeline(
        input_file=args.input,
        output_file=args.output,
        n_clusters=args.clusters
    )
