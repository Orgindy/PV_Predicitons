# compare_clustering_methods.py
# Purpose: Evaluate and compare multiple clustering methods on the dataset

import argparse
import pandas as pd
import numpy as np
from clustering_methods import run_kmeans, run_gmm, run_dbscan, run_agglomerative
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_and_scale_data(csv_path, feature_columns):
    """Load and scale data using the provided feature columns."""
    df = pd.read_csv(csv_path)
    X = df[feature_columns].copy()
    
    # Handle missing values
    missing_count = X.isnull().sum().sum()
    if missing_count > 0:
        print(f"🔧 Filling {missing_count} missing values with median")
        X = X.fillna(X.median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X_scaled

def apply_and_store(df, X, method_name, method_func, **kwargs):
    try:
        labels, score = method_func(X, **kwargs)
        col_name = f"Cluster_{method_name}"
        df[col_name] = labels
        print(f"[{method_name.upper()}] Silhouette Score: {score:.4f}")
        return (method_name, score)
    except Exception as e:
        print(f"Error with method {method_name}: {e}")
        return (method_name, None)

def main():
    parser = argparse.ArgumentParser(description="Compare clustering methods")
    parser.add_argument("--k", type=int, default=5, help="Number of clusters")
    args = parser.parse_args()

    n_clusters = args.k
    input_csv = "clustered_dataset.csv"
    output_csv = "clustered_dataset_with_methods.csv"
    score_log = "clustering_scores.csv"
    
    # Check if input file exists
    if not os.path.exists(input_csv):
        print(f"❌ Input file not found: {input_csv}")
        return
    
    # Check what columns are actually available
    df_check = pd.read_csv(input_csv)
    print(f"📊 Available columns: {list(df_check.columns)}")
    
    # Define flexible feature mapping
    potential_features = {
        'GHI': ['GHI', 'SSRD_power', 'Global_Horizontal_Irradiance'],
        'Temperature': ['T_air', 'T2M', 'Temperature'],
        'WindSpeed': ['Wind_Speed', 'WS', 'WindSpeed'],
        'Albedo': ['Albedo', 'fal', 'effective_albedo'],
        'RC_Potential': ['RC_potential', 'P_rc_net', 'QNET'],
        'RH': ['RH', 'Relative_Humidity'],
        'Cloud_Cover': ['Cloud_Cover', 'TCC', 'CloudCover'],
        'Dew_Point': ['Dew_Point', 'TD2M', 'Dewpoint', 'd2m'],
        'Blue_Band': ['Blue_band', 'Blue_Band', 'Blue'],
        'Green_Band': ['Green_band', 'Green_Band', 'Green'],
        'Red_Band': ['Red_band', 'Red_Band', 'Red'],
        'IR_Band': ['IR_band', 'NIR_band', 'IR_Band', 'IR']
    }
    
    # Find available features
    feature_columns = []
    feature_map = {}
    for feature_name, possible_cols in potential_features.items():
        for col in possible_cols:
            if col in df_check.columns:
                feature_columns.append(col)
                feature_map[feature_name] = col
                print(f"✅ Found {feature_name} as '{col}'")
                break
        else:
            print(f"⚠️ {feature_name} not found in dataset")
    
    if len(feature_columns) < 3:
        print(f"❌ Not enough features found ({len(feature_columns)}). Need at least 3 for clustering.")
        return
    
    print(f"🎯 Using {len(feature_columns)} features: {feature_columns}")
    
    # Load and scale the data
    df, X = load_and_scale_data(input_csv, feature_columns)

    # Store results
    scores = []

    # Run each method with error handling
    print("\n=== Running Clustering Methods ===")
    
    try:
        scores.append(apply_and_store(df, X, "kmeans", run_kmeans, n_clusters=n_clusters))
    except Exception as e:
        print(f"❌ KMeans failed: {e}")
        scores.append(("kmeans", None))
    
    try:
        scores.append(apply_and_store(df, X, "gmm", run_gmm, n_clusters=n_clusters))
    except Exception as e:
        print(f"❌ GMM failed: {e}")
        scores.append(("gmm", None))
    
    try:
        scores.append(apply_and_store(df, X, "dbscan", run_dbscan, eps=0.5, min_samples=5))
    except Exception as e:
        print(f"❌ DBSCAN failed: {e}")
        scores.append(("dbscan", None))
    
    try:
        scores.append(apply_and_store(df, X, "agglomerative", run_agglomerative, n_clusters=n_clusters))
    except Exception as e:
        print(f"❌ Agglomerative failed: {e}")
        scores.append(("agglomerative", None))

    # Save augmented dataset
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved clustered dataset to {output_csv}")
    
    # Save silhouette scores
    score_df = pd.DataFrame(scores, columns=["Method", "SilhouetteScore"])
    score_df.to_csv(score_log, index=False)
    print(f"✅ Saved clustering scores to {score_log}")
    
    # Create visualizations
    create_comparison_plots(df, X, scores)

def create_comparison_plots(df, X, scores):
    """Create PCA and silhouette score comparison plots."""
    try:
        # PCA Visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        plt.figure(figsize=(12, 5))
        
        # Plot 1: PCA comparison
        plt.subplot(1, 2, 1)
        colors = ['red', 'blue', 'green', 'orange']
        methods = ["kmeans", "gmm", "dbscan", "agglomerative"]
        
        for i, method in enumerate(methods):
            col = f"Cluster_{method}"
            if col in df.columns:
                plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df[col], 
                           label=method.upper(), alpha=0.6, s=20, 
                           cmap='tab10')
        
        plt.title("PCA of Climate Features by Clustering Method")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Silhouette scores
        plt.subplot(1, 2, 2)
        methods = [method for method, _ in scores]
        silhouettes = [score if score is not None else 0 for _, score in scores]
        
        bars = plt.bar(methods, silhouettes, color="skyblue", edgecolor="black")
        plt.title("Silhouette Scores by Clustering Method")
        plt.ylabel("Silhouette Score")
        plt.ylim(0, max(silhouettes) * 1.1 if max(silhouettes) > 0 else 1)
        plt.grid(axis="y", alpha=0.3)
        
        # Add score labels on bars
        for bar, score in zip(bars, silhouettes):
            if score > 0:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, 
                        f"{score:.3f}", ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig("clustering_comparison_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved comparison plots: clustering_comparison_plots.png")
        
    except Exception as e:
        print(f"⚠️ Could not create plots: {e}")


if __name__ == "__main__":
    main()
