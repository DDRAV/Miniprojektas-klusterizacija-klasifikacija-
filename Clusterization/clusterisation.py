# Kode for dataset downloading
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors

# 1. Get dataset
path = "C:/Users/drawn/Mokymai/DDRAV Mokymai/Miniprojektas/Clusterization/Dry_Bean_Dataset.csv"
df = pd.read_csv(path)
df = df.drop(columns=['Class'])

# 2. Handling data
print(df.dtypes)
print(f"Number of duplicate rows before removal: {df.duplicated().sum()}")
data = df.drop_duplicates()
print(f"Number of duplicate rows after removal: {data.duplicated().sum()}")

print("Printing missing data:")
print(data.isnull().sum())
sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
plt.show()

print("Summary statistics of data:")
for column in data.columns:
    print(f"Column: {column}")
    print(data[column].describe())
    print("-" * 40)

data.hist(bins=20, figsize=(12, 10))
plt.show()

# Correlation matrix and removal of redundant features
corr = data.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()

corr_threshold = 0.9
high_corr_mask = (corr.abs() > corr_threshold) & (corr != 1.0)
to_drop = set()

for col in high_corr_mask.columns:
    if col not in to_drop:
        correlated_cols = high_corr_mask.index[high_corr_mask[col]].tolist()
        to_drop.update(correlated_cols)

print("The columns considered for dropping are: ", to_drop)
data_reduced = data.drop(columns=to_drop)
sns.heatmap(data_reduced.corr(), annot=True, cmap="coolwarm")
plt.show()


# Scale the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_reduced)

# Calculate the distances to the nearest neighbors
nbrs = NearestNeighbors(n_neighbors=10).fit(data_scaled)
distances, indices = nbrs.kneighbors(data_scaled)

# Sort the distances to look for a good eps value
sorted_distances = np.sort(distances[:, -1], axis=0)

# Plot the sorted distances (this is often used to determine a good eps value for DBSCAN)
plt.figure(figsize=(10, 6))
plt.plot(sorted_distances)
plt.title("Sorted Distances to Nearest Neighbors")
plt.xlabel("Points")
plt.ylabel("Distance")
plt.show()

# Define a Dunn index calculation function
def calculate_dunn_index(data, labels):
    unique_labels = np.unique(labels)
    inter_cluster_distances = []
    intra_cluster_distances = []

    for i, cluster_i in enumerate(unique_labels):
        points_i = data[labels == cluster_i]
        intra_cluster_distances.append(np.max(pairwise_distances(points_i)))

        for cluster_j in unique_labels[i + 1:]:
            points_j = data[labels == cluster_j]
            inter_cluster_distances.append(np.min(pairwise_distances(points_i, points_j)))

    return np.min(inter_cluster_distances) / np.max(intra_cluster_distances)

# Agglomerative Clustering dendrogram
linked = linkage(data_scaled, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, truncate_mode='level', p=10, leaf_rotation=45, leaf_font_size=10, show_contracted=True)
plt.title("Dendrogram for Agglomerative Clustering")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()

# K-means WCSS and elbow method
wcss = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(k_values, wcss, marker='o')
plt.xlabel("Cluster count (k)")
plt.ylabel("WCSS")
plt.title("WCSS depending on cluster count")
plt.show()

knee_locator = KneeLocator(k_values, wcss, curve="convex", direction="decreasing")
optimal_kmeans = knee_locator.knee
print(f"Optimal number of clusters detected: {optimal_kmeans}")

# Define parameter grids for each method
kmeans_params = {'n_clusters': range(3, 8)}
dbscan_params = {'eps': [0.5, 0.7, 0.9], 'min_samples': [5, 10, 20, 40]}
agg_params = {'n_clusters': range(3, 8), 'linkage': ['ward', 'complete', 'average']}

best_params = {}

# Evaluate KMeans
best_kmeans_score = -1
for params in ParameterGrid(kmeans_params):
    kmeans = KMeans(**params, random_state=42)
    labels = kmeans.fit_predict(data_scaled)
    score = silhouette_score(data_scaled, labels)
    if score > best_kmeans_score:
        best_kmeans_score = score
        best_params['KMeans'] = params

# Evaluate DBSCAN
best_dbscan_score = -1
for params in ParameterGrid(dbscan_params):
    dbscan = DBSCAN(**params)
    labels = dbscan.fit_predict(data_scaled)
    if len(np.unique(labels)) > 1:
        score = silhouette_score(data_scaled, labels)
        if score > best_dbscan_score:
            best_dbscan_score = score
            best_params['DBSCAN'] = params

# Evaluate Agglomerative Clustering
best_agg_score = -1
for params in ParameterGrid(agg_params):
    agg = AgglomerativeClustering(**params)
    labels = agg.fit_predict(data_scaled)
    score = silhouette_score(data_scaled, labels)
    if score > best_agg_score:
        best_agg_score = score
        best_params['Agglomerative'] = params

print("Best Parameters:")
print(best_params)

# Visualize clusters for each method
results = {}
methods = {
    "KMeans": KMeans(**best_params['KMeans'], random_state=42),
    "DBSCAN": DBSCAN(**best_params['DBSCAN']),
    "Agglomerative": AgglomerativeClustering(**best_params['Agglomerative']),
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for idx, (name, model) in enumerate(methods.items()):
    labels = model.fit_predict(data_scaled)

    # Visualize Clusters in PCA space
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)
    ax = axes[idx]
    if name == "DBSCAN" and -1 in labels:
        ax.scatter(data_pca[labels == -1, 0], data_pca[labels == -1, 1], c='red', s=30, marker='x', label='Noise')
    scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6, label="Clusters")
    ax.set_title(f"{name} Clusters")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    fig.colorbar(scatter, ax=ax, orientation='horizontal', fraction=0.02, pad=0.1)
    ax.legend(loc="upper right")

    # Store quality metrics
    if len(np.unique(labels)) > 1:
        silhouette_avg = silhouette_score(data_scaled, labels)
        davies_bouldin = davies_bouldin_score(data_scaled, labels)
        dunn_index = calculate_dunn_index(data_scaled, labels)
    else:
        silhouette_avg = davies_bouldin = dunn_index = -1

    results[name] = {
        "Silhouette Score": silhouette_avg,
        "Davies-Bouldin Index": davies_bouldin,
        "Dunn Index": dunn_index,
    }

plt.show()

# Convert results to DataFrame
results_df = pd.DataFrame.from_dict(results, orient="index")
print("Clustering Metrics Summary:")
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)     # Show all rows if needed (optional)
print(results_df)

# DBSCAN cluster analysis and noise removal
labels_dbscan = DBSCAN(**best_params['DBSCAN']).fit_predict(data_scaled)

# Exclude noise points (-1) and only keep valid clusters
valid_mask = labels_dbscan != -1
data_without_noise = data_scaled[valid_mask]
labels_without_noise = labels_dbscan[valid_mask]

# Analyze DBSCAN clusters
dbscan_cluster_counts = pd.Series(labels_without_noise).value_counts().sort_index()
print(f"DBSCAN Cluster Sizes (Excluding Noise):\n{dbscan_cluster_counts}")

# Data with clusters for visualization
data_with_clusters_dbscan = pd.DataFrame(data_without_noise, columns=data_reduced.columns)
data_with_clusters_dbscan['Cluster'] = labels_without_noise

# Plotting histograms for each feature in each DBSCAN cluster
num_features = len(data_with_clusters_dbscan.columns) - 1  # Exclude the 'Cluster' column
fig, axes = plt.subplots(num_features, 1, figsize=(8, 2 * num_features))  # Adjust figure height for clarity

for i, column in enumerate(data_with_clusters_dbscan.columns[:-1]):  # Exclude the 'Cluster' column
    ax = axes[i]
    for cluster in data_with_clusters_dbscan['Cluster'].unique():
        cluster_data = data_with_clusters_dbscan[data_with_clusters_dbscan['Cluster'] == cluster]
        ax.hist(cluster_data[column], bins=40, alpha=0.5, density=True, label=f"Cluster {cluster}")  # Use density=True
    ax.set_title(f"Distribution of {column} by DBSCAN Cluster")
    ax.set_xlabel(f"{column} Values")
    ax.set_ylabel("Density")  # Update to reflect density
    ax.legend(title="Clusters")
plt.tight_layout()  # Adjust layout
plt.show()

# KMeans Cluster Analysis
labels_kmeans = KMeans(**best_params['KMeans'], random_state=42).fit_predict(data_scaled)
# Analyze KMeans clusters
kmeans_cluster_counts = pd.Series(labels_kmeans).value_counts().sort_index()
print(f"KMeans Cluster Sizes:\n{kmeans_cluster_counts}")

# Data with clusters for analysis
data_with_clusters = pd.DataFrame(data_scaled, columns=data_reduced.columns)
data_with_clusters['Cluster'] = labels_kmeans

# Plotting histograms for each feature in each cluster
num_features = len(data_with_clusters.columns) - 1  # Exclude the 'Cluster' column
fig, axes = plt.subplots(num_features, 1, figsize=(12, 2 * num_features))  # Adjust figure height for clarity

for i, column in enumerate(data_with_clusters.columns[:-1]):  # Exclude the 'Cluster' column
    ax = axes[i]
    for cluster in data_with_clusters['Cluster'].unique():
        cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster]
        ax.hist(cluster_data[column], bins=40, alpha=0.5, density=True, label=f"Cluster {cluster}")  # Use density=True
    ax.set_title(f"Distribution of {column} by Cluster")
    ax.set_xlabel(f"{column} Values")
    ax.set_ylabel("Density")  # Update to reflect density
    ax.legend(title="Clusters")
plt.tight_layout()  # Adjust layout
plt.show()
