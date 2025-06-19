import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
import geopandas as gpd
import matplotlib.pyplot as plt
from utils.plot_style import apply_plot_style, finalize_plot

apply_plot_style()
import seaborn as sns
import contextily as ctx
import glob
from pathlib import Path
import joblib
from datetime import datetime
import os
from xgboost import XGBRegressor
from sklearn.gaussian_process.kernels import RBF

# -----------------------------
# PV Cell Profile Management
# -----------------------------

def load_pv_profiles_from_csv(file_path):
    df = pd.read_csv(file_path)
    profiles = {}
    for _, row in df.iterrows():
        profiles[row['Technology']] = {
            "spectral_response": {
                "Blue": row['Blue'],
                "Green": row['Green'],
                "Red": row['Red'],
                "IR": row['IR']
            },
            "temperature_coefficient": row['TempCoeff']
        }
    return profiles

def get_pv_cell_profiles():
    pv_profiles = {
        "Silicon": {
            "spectral_response": {"Blue": 0.3, "Green": 0.7, "Red": 1.0, "IR": 1.0},
            "temperature_coefficient": -0.0045
        },
        "Perovskite": {
            "spectral_response": {"Blue": 1.0, "Green": 0.9, "Red": 0.4, "IR": 0.1},
            "temperature_coefficient": -0.0025
        },
        "Tandem": {
            "spectral_response": {"Blue": 1.0, "Green": 1.0, "Red": 1.0, "IR": 1.0},
            "temperature_coefficient": -0.0035
        },
        "CdTe": {
            "spectral_response": {"Blue": 0.8, "Green": 0.9, "Red": 0.7, "IR": 0.2},
            "temperature_coefficient": -0.0028
        }
    }
    return pv_profiles

# -----------------------------
# Cluster Spectra Calculation
# -----------------------------

def compute_cluster_spectra(df_clustered, cluster_col='Cluster_ID'):
    spectral_cols = ['Blue_band', 'Green_band', 'Red_band', 'IR_band']
    temp_col = 'T_air'
    grouped = df_clustered.groupby(cluster_col)
    cluster_spectra_df = grouped[spectral_cols + [temp_col]].mean().reset_index()
    spectrum_sum = cluster_spectra_df[spectral_cols].sum(axis=1)
    for col in spectral_cols:
        cluster_spectra_df[col] = cluster_spectra_df[col] / spectrum_sum
    return cluster_spectra_df

from pv_potential import calculate_pv_potential

def prepare_features_for_ml(df):
    feature_names = [
        'GHI', 'T_air', 'RC_potential', 'Wind_Speed',
        'Dew_Point', 'Cloud_Cover', 'Red_band',
        'Blue_band', 'IR_band', 'Total_band'
    ]
    X = df[feature_names].copy()
    y = df['PV_Potential']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, feature_names, scaler


def train_random_forest(X_scaled, y, feature_names, test_size=0.2, random_state=42,
                        n_estimators=200, max_depth=12, n_clusters=5,
                        output_plot=None, model_dir="results/models"):
    """
    Train a Random Forest model and save the model and feature importance plot.

    Returns:
        model, X_train, X_test, y_train, y_test, y_pred, model_path
    """
    # --- Split data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state)

    # --- Train model ---
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- Evaluation ---
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")

    # --- Save model ---
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = os.path.join(model_dir, f"rf_model_k{n_clusters}_{timestamp}.joblib")
    joblib.dump(model, model_path)
    print(f"💾 Random Forest model saved to: {model_path}")

    # --- Plot and save feature importance ---
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure()
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title("Feature Importance (Random Forest)")
    if output_plot:
        finalize_plot(os.path.basename(output_plot))
    else:
        finalize_plot("feature_importance.png")

    return model, X_train, X_test, y_train, y_test, y_pred, model_path


def train_ensemble_model(df, feature_cols, target_col='PV_Potential_physics', test_size=0.2, random_state=42):
    """
    Train ensemble of RF, XGBoost, and GPR, return ensemble predictions and uncertainty.
    """
    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train Models
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=random_state)
    rf.fit(X_train, y_train)

    xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=random_state)
    xgb.fit(X_train, y_train)

    gpr = GaussianProcessRegressor(kernel=RBF(), alpha=1e-2, normalize_y=True)
    gpr.fit(X_train, y_train)

    # Predict on Full Dataset
    rf_preds = rf.predict(X)
    xgb_preds = xgb.predict(X)
    gpr_preds, gpr_std = gpr.predict(X, return_std=True)

    # Ensemble prediction (average)
    ensemble_preds = (rf_preds + xgb_preds + gpr_preds) / 3

    # Add to dataframe
    df_out = df.copy()
    df_out['Predicted_PV_Potential'] = ensemble_preds
    df_out['Prediction_Uncertainty'] = gpr_std
    
    print(f"✅ Ensemble model trained with {len(feature_cols)} features")
    return df_out


def predict_pv_potential(model, X_scaled, df_original):
    predictions = model.predict(X_scaled)
    df_result = df_original.copy()
    df_result['Predicted_PV_Potential'] = predictions
    return df_result

def train_hybrid_ml_models(X, y):
    """
    Trains multiple ML models and compares their performance.
    
    Parameters:
    - X (pd.DataFrame): Input features.
    - y (pd.Series): Target variable (PV potential).
    
    Returns:
    - models (dict): Trained models with their scores.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    # Gaussian Process Regressor
    gpr = GaussianProcessRegressor()
    gpr.fit(X_train, y_train)
    gpr_pred = gpr.predict(X_test)
    
    # Model performance
    models = {
        "RandomForest": {
            "model": rf,
            "R2": r2_score(y_test, rf_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, rf_pred))
        },
        "GaussianProcess": {
            "model": gpr,
            "R2": r2_score(y_test, gpr_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, gpr_pred))
        }
    }
    
    print("\n=== Model Performance ===")
    for name, data in models.items():
        print(f"{name} - R²: {data['R2']:.4f}, RMSE: {data['RMSE']:.2f}")
    
    return models

def run_kmedoids_clustering(X_scaled, n_clusters=4, metric='euclidean', random_state=42):
    kmedoids = KMedoids(n_clusters=n_clusters, metric=metric, init='k-medoids++', random_state=random_state)
    kmedoids.fit(X_scaled)
    labels = kmedoids.labels_
    silhouette = silhouette_score(X_scaled, labels)
    print(f"K-Medoids Silhouette Score: {silhouette:.4f}")
    return kmedoids, labels, silhouette


def evaluate_cluster_quality(X_scaled, cluster_labels):
    """
    Evaluates clustering performance using multiple metrics.
    
    Parameters:
    - X_scaled (np.array): Standardized feature matrix used for clustering.
    - cluster_labels (np.array): Labels assigned to each sample by the clustering algorithm.
    
    Returns:
    - scores (dict): Dictionary of metric names and their values.
    """
    try:
        silhouette = silhouette_score(X_scaled, cluster_labels)
        calinski = calinski_harabasz_score(X_scaled, cluster_labels)
        davies = davies_bouldin_score(X_scaled, cluster_labels)

        scores = {
            "Silhouette Score": round(silhouette, 4),
            "Calinski-Harabasz Score": round(calinski, 2),
            "Davies-Bouldin Score": round(davies, 4)
        }

        print("\n=== Clustering Quality Metrics ===")
        for k, v in scores.items():
            print(f"{k}: {v}")

        return scores

    except Exception as e:
        print(f"❌ Cluster evaluation failed: {e}")
        return {}


def assign_clusters_to_dataframe(df, labels, column_name='Cluster_ID'):
    df_out = df.copy()
    df_out[column_name] = labels
    return df_out

# -----------------------------
# PV Technology Matching
# -----------------------------


def match_technology_to_clusters(cluster_spectra_df, pv_profiles, temp_col='T_air'):
    spectral_bands = ['Blue', 'Green', 'Red', 'IR']
    results = []
    for _, row in cluster_spectra_df.iterrows():
        cluster_id = row['Cluster_ID']
        cluster_temp = row[temp_col]
        tech_scores = {}
        for tech, profile in pv_profiles.items():
            score = sum(row[band + '_band'] * profile['spectral_response'][band] for band in spectral_bands)
            temp_coeff = profile['temperature_coefficient']
            temp_penalty = temp_coeff * (cluster_temp - 25)
            adjusted_score = score + temp_penalty
            tech_scores[tech] = adjusted_score
        best_match = max(tech_scores, key=tech_scores.get)
        results.append({
            'Cluster_ID': cluster_id,
            'Best_Technology': best_match,
            **tech_scores
        })
    return pd.DataFrame(results)


def plot_clusters_map(df, lat_col='latitude', lon_col='longitude', cluster_col='Cluster_ID', title='PV Performance Clusters'):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326").to_crs(epsg=3857)
    fig, ax = plt.subplots()
    gdf.plot(ax=ax, column=cluster_col, cmap='viridis', legend=True, markersize=35, edgecolor='k')
    try:
        ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite)
    except Exception as e:
        print("Basemap could not be loaded.")
    ax.set_title(title)
    ax.set_axis_off()
    finalize_plot("clusters_map.png")


def prepare_features_for_clustering(df, feature_cols):
    """
    Extract and standardize the feature space for clustering.

    Parameters:
    - df: DataFrame containing input features
    - feature_cols: list of column names to use for clustering

    Returns:
    - X_scaled: standardized numpy array of features
    - valid_idx: index of rows that passed filtering (for assigning cluster labels back)
    """
    df_features = df[feature_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)
    return X_scaled, df_features.index


def main_clustering_pipeline(input_file='merged_dataset.csv', output_dir='results', n_clusters=5):
    df = pd.read_csv(input_file)
    print("Calculating physics-based PV Potential...")
    df['PV_Potential_physics'] = calculate_pv_potential(
        df['GHI'].values,
        df['T_air'].values,
        df['RC_potential'].values,
        df['Red_band'].values,
        df['Total_band'].values
    )

    # --- Timestamp and folders ---
    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    cluster_dir = os.path.join(output_dir, "clusters")
    plot_dir = os.path.join(output_dir, "plots")
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(cluster_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # --- File paths ---
    output_clustered = os.path.join(cluster_dir, f"clustered_dataset_{run_id}.csv")
    output_matched = os.path.join(cluster_dir, f"matched_dataset_{run_id}.csv")
    output_plot = os.path.join(plot_dir, f"feature_importance_{run_id}.png")
    model_path = os.path.join(model_dir, f"rf_model_k{n_clusters}_{run_id}.joblib")

    print("Preparing features for Random Forest...")
    X_scaled, y, feature_names, scaler_rf = prepare_features_for_ml(df)

    print("Training Random Forest...")
    model, X_train, X_test, y_train, y_test, y_pred, model_path = train_random_forest(
        X_scaled,
        y.values,
        feature_names,
        n_clusters=n_clusters,
        output_plot=output_plot,
        model_dir=model_dir
    )
    
    metrics_path = os.path.join(cluster_dir, f"model_metrics_k{n_clusters}_{run_id}.txt")
    with open(metrics_path, 'w') as f:
        f.write("=== Random Forest Model Evaluation ===\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"n_clusters: {n_clusters}\n\n")
        f.write(f"R² Score: {r2_score(y_test, y_pred):.4f}\n")
        f.write(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}\n")
        f.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}\n")
    print(f"📄 Model evaluation metrics saved to: {metrics_path}")

    print("Predicting PV Potential...")
    df_with_pred = predict_pv_potential(model, X_scaled, df)
    # Update the variable name for consistency
    df = df_with_pred.copy()
    print("Preparing features for clustering...")
    clustering_features = ['GHI', 'T_air', 'RC_potential', 'Wind_Speed', 'Dew_Point', 
                          'Blue_band', 'Red_band', 'IR_band', 'Total_band', 'Predicted_PV_Potential']
    X_cluster_scaled, valid_idx = prepare_features_for_clustering(df_with_pred, clustering_features)    
    
    # Define features for ensemble model
    ensemble_features = ['GHI', 'T_air', 'RC_potential', 'Wind_Speed', 'Dew_Point', 
                        'Cloud_Cover', 'Blue_band', 'Red_band', 'IR_band', 'Total_band']
    
    print("Training ensemble model...")
    df_with_pred = train_ensemble_model(df_with_pred, ensemble_features)    
    
    print("Running K-Medoids clustering...")
    kmedoids, labels, silhouette = run_kmedoids_clustering(X_cluster_scaled, n_clusters=n_clusters)
    sil_path = os.path.join(cluster_dir, f"silhouette_score_{n_clusters}_{run_id}.txt")
    with open(sil_path, 'w') as f:
        f.write(f"Silhouette Score: {silhouette:.4f}\n")
    print(f"📄 Silhouette score saved to: {sil_path}")

    print("Assigning cluster labels...")
    df_clustered = assign_clusters_to_dataframe(df_with_pred, labels)

    # Cluster spectrum matching
    print("\n=== Computing Cluster-Averaged Spectra and Temperatures ===")
    cluster_spectra = compute_cluster_spectra(df_clustered, cluster_col='Cluster_ID')

    print("\n=== Matching PV Technologies to Clusters ===")
    pv_profiles = get_pv_cell_profiles()
    match_df = match_technology_to_clusters(cluster_spectra, pv_profiles)
    match_df.to_csv(output_matched, index=False)
    print(f"✅ Technology-matched dataset saved to: {output_matched}")

    df_clustered.to_csv(output_clustered, index=False)
    print(f"✅ Clustered dataset saved to: {output_clustered}")

    print("Plotting clusters on map...")
    plot_clusters_map(df_clustered)

    return df_clustered

def multi_year_clustering(input_dir='.', output_dir='clustered_outputs', n_clusters=5, file_pattern='merged_dataset_*.csv'):
    """
    Run main_clustering_pipeline across multiple years of merged datasets.

    Parameters:
    - input_dir: directory where merged yearly datasets are stored.
    - output_dir: directory to save clustered + matched results.
    - n_clusters: number of K-Medoids clusters.
    - file_pattern: glob pattern to match yearly files.

    Returns:
    - summary_df: DataFrame with all cluster centroids and matched technologies across years.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matched_tech_dfs = []

    input_files = sorted(input_dir.glob(file_pattern))
    if not input_files:
        print("⚠️ No matching input files found.")
        return

    for file in input_files:
        year = ''.join(filter(str.isdigit, file.stem))
        print(f"\n📅 Processing year: {year} — {file.name}")

        clustered_out = output_dir / f'clustered_dataset_{year}.csv'
        matched_out = output_dir / f'matched_dataset_{year}.csv'

        # Run core clustering logic
        df_clustered = main_clustering_pipeline(
        input_file=str(file),
        output_dir=str(output_dir),
        n_clusters=n_clusters
            )
    
        # Load and tag matched output
        match_df = pd.read_csv(matched_out)
        match_df['Year'] = year
        matched_tech_dfs.append(match_df)

    # Combine results for all years
    summary_df = pd.concat(matched_tech_dfs, ignore_index=True)
    summary_csv = output_dir / 'summary_technology_matching.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n✅ Multi-year clustering completed. Summary saved to: {summary_csv}")

    return summary_df

def generate_cluster_summaries(clustered_df, cluster_col='Cluster_ID', save_path=None):
    """
    Generate summaries per cluster: stats, spectral profiles, technology distributions.

    Parameters:
    - clustered_df: DataFrame with clustered and matched PV technology data.
    - cluster_col: name of column containing cluster labels.
    - save_path: if provided, saves summary CSV to this path.

    Returns:
    - summary_df: summary DataFrame.
    """
    summary_stats = clustered_df.groupby(cluster_col).agg({
        'GHI': 'mean',
        'T_air': 'mean',
        'RC_potential': 'mean',
        'Red_band': 'mean',
        'Blue_band': 'mean',
        'IR_band': 'mean',
        'Total_band': 'mean',
        'PV_Potential_physics': 'mean',
        'Predicted_PV_Potential': 'mean',
        'latitude': ['mean', 'std'],
        'longitude': ['mean', 'std']
    })

    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
    summary_stats.reset_index(inplace=True)

    # Count points per cluster
    cluster_sizes = clustered_df[cluster_col].value_counts().reset_index()
    cluster_sizes.columns = [cluster_col, 'n_points']
    summary_df = summary_stats.merge(cluster_sizes, on=cluster_col)

    # Most common technology per cluster
    if 'Best_Technology' in clustered_df.columns:
        top_tech = clustered_df.groupby(cluster_col)['Best_Technology'] \
                               .agg(lambda x: x.value_counts().idxmax()) \
                               .reset_index(name='Dominant_Technology')
        summary_df = summary_df.merge(top_tech, on=cluster_col)

    if save_path:
        summary_df.to_csv(save_path, index=False)
        print(f"✅ Cluster summary saved to: {save_path}")

    return summary_df

def generate_zone_descriptions(df, cluster_col='Cluster_ID'):
    grouped = df.groupby(cluster_col).agg({
        'T_air': 'mean',
        'RC_potential': 'mean',
        'Red_band': 'mean',
        'Best_Technology': lambda x: x.mode().iloc[0]
    })

    def describe_row(row):
        desc = []
        desc.append("Hot" if row['T_air'] > 18 else "Cool")
        desc.append("High RC" if row['RC_potential'] > 40 else "Low RC")
        desc.append("Red-rich" if row['Red_band'] > 0.35 else "Red-poor")
        desc.append(f"→ {row['Best_Technology']}")
        return " ".join(desc)

    grouped['Zone_Description'] = grouped.apply(describe_row, axis=1)
    return grouped[['Zone_Description']]


def summarize_and_plot_multi_year_clusters(summary_df, output_dir='results/clusters'):
    """
    Generate seasonal + yearly summaries and cluster maps.
    Saves summary CSVs and plots per year.

    Parameters:
    - summary_df: combined cluster-to-technology DataFrame with 'Year' column.
    - output_dir: location to save visualizations and summaries.
    """
    output_dir = Path(output_dir)

    # Plot best technology frequency per year
    tech_freq = summary_df.groupby(['Year', 'Best_Technology']).size().reset_index(name='Count')
    plt.figure()
    sns.barplot(data=tech_freq, x='Year', y='Count', hue='Best_Technology')
    plt.title("Best PV Technology Frequency per Year")
    finalize_plot('tech_frequency_per_year.png')

    # Compute basic yearly averages per tech
    avg_scores = summary_df.groupby(['Year', 'Best_Technology']).mean(numeric_only=True).reset_index()
    avg_scores.to_csv(output_dir / 'average_scores_per_tech_year.csv', index=False)

    print(f"📊 Saved summary plots and CSVs to {output_dir}")

def rc_only_clustering(df, n_clusters=5):
    from sklearn_extra.cluster import KMedoids
    from sklearn.preprocessing import StandardScaler

    features = ['P_rc_net', 'T_air', 'RH', 'Wind_Speed']
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmedoids = KMedoids(n_clusters=n_clusters, random_state=42)
    labels = kmedoids.fit_predict(X_scaled)
    df['RC_Cluster'] = -1
    df.loc[X.index, 'RC_Cluster'] = labels

    return df, kmedoids


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run PV prediction pipeline")
    parser.add_argument("--input-dir", default="data/merged_years")
    parser.add_argument("--output-dir", default="results/clusters")
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--file-pattern", default="merged_dataset_*.csv")
    parser.add_argument("--db-url", default=os.getenv("PV_DB_URL"))
    parser.add_argument("--db-table", default=os.getenv("PV_DB_TABLE", "pv_data"))
    args = parser.parse_args()

    if args.db_url:
        from database_utils import read_table, write_dataframe
        temp_dir = Path(args.output_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        df_db = read_table(args.db_table, db_url=args.db_url)
        temp_csv = temp_dir / "db_input.csv"
        df_db.to_csv(temp_csv, index=False)
        df_clustered = main_clustering_pipeline(
            input_file=str(temp_csv),
            output_dir=args.output_dir,
            n_clusters=args.n_clusters,
        )
        write_dataframe(df_clustered, args.db_table, db_url=args.db_url, if_exists="replace")
    else:
        summary = multi_year_clustering(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            n_clusters=args.n_clusters,
            file_pattern=args.file_pattern,
        )

        if summary is not None:
            summarize_and_plot_multi_year_clusters(summary, output_dir=args.output_dir)
