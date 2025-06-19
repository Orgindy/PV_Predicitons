import numpy as np
from clustering_methods import run_clustering


def test_run_kmeans():
    X = np.random.rand(10, 3)
    labels, score = run_clustering(X, method="kmeans", n_clusters=2, random_state=0)
    assert len(labels) == 10
    assert -1 <= score <= 1


def test_run_dbscan_edge_case():
    X = np.zeros((5, 2))
    labels, score = run_clustering(X, method="dbscan", eps=0.1, min_samples=2)
    assert len(labels) == 5
    assert score == -1 or -1 <= score <= 1
