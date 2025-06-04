import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def run_kmeans(X, n_clusters=5, random_state=42):
    """
    Run KMeans clustering and return labels and silhouette score.

    Parameters:
        X (np.ndarray): Feature matrix
        n_clusters (int): Number of clusters
        random_state (int): Seed for reproducibility

    Returns:
        labels (np.ndarray): Cluster labels
        score (float): Silhouette score
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    return labels, score

def run_gmm(X, n_clusters=5, random_state=42):
    """
    Run Gaussian Mixture Model clustering and return labels and silhouette score.

    Parameters:
        X (np.ndarray): Feature matrix
        n_clusters (int): Number of mixture components
        random_state (int): Seed for reproducibility

    Returns:
        labels (np.ndarray): Cluster labels
        score (float): Silhouette score
    """
    model = GaussianMixture(n_components=n_clusters, random_state=random_state)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    return labels, score

def run_dbscan(X, eps=0.5, min_samples=5):
    """
    Run DBSCAN clustering and return labels and silhouette score.

    Parameters:
        X (np.ndarray): Feature matrix
        eps (float): Maximum distance between samples to be considered neighbors
        min_samples (int): Minimum number of samples in a neighborhood to form a core point

    Returns:
        labels (np.ndarray): Cluster labels
        score (float): Silhouette score (or -1 if insufficient clusters)
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    # DBSCAN may assign -1 to noise points; remove them before scoring
    mask = labels != -1
    if len(set(labels[mask])) < 2:
        score = -1  # Not enough clusters to evaluate
    else:
        score = silhouette_score(X[mask], labels[mask])

    return labels, score

def run_agglomerative(X, n_clusters=5):
    """
    Run Agglomerative (hierarchical) clustering and return labels and silhouette score.

    Parameters:
        X (np.ndarray): Feature matrix
        n_clusters (int): Number of clusters

    Returns:
        labels (np.ndarray): Cluster labels
        score (float): Silhouette score
    """
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    return labels, score

def run_clustering(X, method="kmeans", **kwargs):
    """
    Unified interface to run the selected clustering method.

    Parameters:
        X (np.ndarray): Feature matrix
        method (str): Clustering method ('kmeans', 'gmm', 'dbscan', 'agglomerative')
        **kwargs: Parameters specific to the chosen clustering method

    Returns:
        labels (np.ndarray): Cluster labels
        score (float): Silhouette score
    """
    if method == "kmeans":
        return run_kmeans(X, **kwargs)
    elif method == "gmm":
        return run_gmm(X, **kwargs)
    elif method == "dbscan":
        return run_dbscan(X, **kwargs)
    elif method == "agglomerative":
        return run_agglomerative(X, **kwargs)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

