import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

data, labels_true = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
data = StandardScaler().fit_transform(data)


kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(data)
hierarchical = AgglomerativeClustering(n_clusters=4)
hierarchical_labels = hierarchical.fit_predict(data)

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(data)

def evaluate_clustering(data, labels, method_name):
    silhouette = silhouette_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    print(f"{method_name} Clustering:")
    print(f"Silhouette Score: {silhouette:.2f}")
    print(f"Davies-Bouldin Index: {davies_bouldin:.2f}\n")

evaluate_clustering(data, kmeans_labels, "K-means")
evaluate_clustering(data, hierarchical_labels, "Hierarchical")
evaluate_clustering(data, dbscan_labels, "DBSCAN")


def plot_clusters(data, labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar()
    plt.show()

plot_clusters(data, kmeans_labels, "K-means Clustering")
plot_clusters(data, hierarchical_labels, "Hierarchical Clustering")
plot_clusters(data, dbscan_labels, "DBSCAN Clustering")


linkage_matrix = linkage(data, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()
