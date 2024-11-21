#Kode for dataset downloading
#import kagglehub
#https://www.kaggle.com/datasets/muratkokludataset/dry-bean-dataset/data
# Download latest version
#path = kagglehub.dataset_download("fatihb/coffee-quality-data-cqi")
#print("Path to dataset files:", path)


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

#path = "C:/Users/drawn/Mokymai/DDRAV Mokymai/Miniprojektas/Clusterization/Dry_Bean_Dataset.csv"
path = "C:/Users/ddrav/OneDrive - Everwest/Desktop/Projektas/mok/Miniprojektas/Clusterization/Dry_Bean_Dataset.csv"
#1. Get dataset

df = pd.read_csv(path)
df = df.drop(columns=['Class'])

#2. Handling data
#Printing data types:
print(df.dtypes)

# Get all rows that are duplicates, including original rows
all_duplicates = df[df.duplicated(keep=False)]
# Print the number of duplicates before removal
print(f"Number of duplicate rows before removal: {df.duplicated().sum()}")
# Create a function to find duplicated values in each row
def find_duplicated_values(row, df_columns):
    duplicates = {}
    for col in df_columns:
        if (df[df[col] == row[col]].shape[0] > 1):  # Check if the value appears more than once in the column
            duplicates[col] = row[col]
    return duplicates

# Apply the function to find specific duplicated values for each duplicate row
detailed_duplicates = all_duplicates.copy()
detailed_duplicates['Duplicated Values'] = all_duplicates.apply(
    lambda row: find_duplicated_values(row, df.columns), axis=1
)

# Print detailed duplicates with specific duplicated values
if not detailed_duplicates.empty:
    print(f"Number of duplicated rows (including original rows): {len(detailed_duplicates)}")
    print(detailed_duplicates[['Duplicated Values']])
else:
    print("No duplicated rows found.")


# Remove duplicates
data = df.drop_duplicates()

# Print the number of duplicates after removal
print(f"Number of duplicate rows after removal: {data.duplicated().sum()}")

# Check for missing values

print("Printing missing data:")
print(data.isnull().sum())

# Visualize missing data (optional, if you use seaborn)

sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
plt.show()

print("Summary statistics of data:")
for column in data.columns:
    print(f"Column: {column}")
    print(data[column].describe())
    print("-" * 40)

data.hist(bins=20, figsize=(12, 10))
plt.show()

# Checking feature correlation via correlation matrix
corr = data.corr()

# Heatmap visualization
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()

# Set a threshold for correlation
corr_threshold = 0.9

# Create a boolean mask to identify highly correlated features
high_corr_mask = (corr.abs() > corr_threshold) & (corr != 1.0)

# Track columns to drop
to_drop = set()

# Iterate through the correlation matrix to find features to drop
for col in high_corr_mask.columns:
    # If the column is not already in the list of features to drop
    if col not in to_drop:
        # Get correlated columns
        correlated_cols = high_corr_mask.index[high_corr_mask[col]].tolist()
        # Add correlated columns to drop (skip the current column itself)
        to_drop.update(correlated_cols)

# Remove the redundant features from the datframe

print("The columns considered for dropping are: ", to_drop)
data_reduced = data.drop(columns=to_drop)
corr_reduced = data_reduced.corr()

# Heatmap visualization
sns.heatmap(corr_reduced, annot=True, cmap="coolwarm")
plt.show()

#Scaling data
#
# Jei turiu duomenu kurie ne virsyja 1 ar reikia skalizuot?
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_reduced.select_dtypes(include=["float64", "int64"]))


linked = linkage(data_scaled, method='ward')  # 'ward' for minimizing variance
# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked, truncate_mode='level', p=10, leaf_rotation=45, leaf_font_size=10, show_contracted=True)
plt.title("Dendrogram for Agglomerative Clustering")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()

#Kmeans Clustering:
def calculate_dunn_index(data, labels):
    # Pairwise distances
    distances = pairwise_distances(data)

    # Find unique clusters
    unique_labels = np.unique(labels)

    # Inter-cluster distance (minimum distance between clusters)
    inter_cluster_distances = []
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            cluster_i = data[labels == unique_labels[i]]
            cluster_j = data[labels == unique_labels[j]]
            inter_cluster_distances.append(
                np.min(pairwise_distances(cluster_i, cluster_j))
            )
    min_inter_cluster_distance = np.min(inter_cluster_distances)

    # Intra-cluster distance (maximum within each cluster)
    intra_cluster_distances = []
    for label in unique_labels:
        cluster_points = data[labels == label]
        intra_cluster_distances.append(np.max(pairwise_distances(cluster_points)))
    max_intra_cluster_distance = np.max(intra_cluster_distances)

    # Compute Dunn Index
    dunn_index = min_inter_cluster_distance / max_intra_cluster_distance
    return dunn_index

# Counting WCSS depending on cluster count
#wcss = []
#for k in range(1, 11):
#    kmeans = KMeans(n_clusters=k, random_state=42)
#    kmeans.fit(data_scaled)
#    wcss.append(kmeans.inertia_)

#plt.plot(range(1, 11), wcss, marker='o')
#plt.xlabel("Cluster count (k)")
#plt.ylabel("WCSS")
#plt.title("WCSS depending on cluster count")
#plt.show()
#optimal_kmeans = 7  # Choose based on graph
#2 metodas
# Counting WCSS depending on cluster count
wcss = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

# Plot WCSS graph
plt.plot(k_values, wcss, marker='o')
plt.xlabel("Cluster count (k)")
plt.ylabel("WCSS")
plt.title("WCSS depending on cluster count")
plt.show()

# Automatically detect the "elbow" point
knee_locator = KneeLocator(k_values, wcss, curve="convex", direction="decreasing")
optimal_kmeans = knee_locator.knee  # The optimal number of clusters

print(f"Optimal number of clusters detected: {optimal_kmeans}")
# Choosing best cluster count based on wcss
# Define narrower parameter ranges for eps and min_samples
eps_values = np.linspace(0.1, 2.0, 20)  # Reduced eps range
min_samples_values = range(5, 50, 10)  # Adjusted min_samples range

# Initialize lists to store the results
results_eps_min_samples = []

# Perform grid search over eps and min_samples
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data_scaled)

        # Exclude noise points (-1) for valid clusters
        core_samples_mask = labels != -1
        num_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)  # Excluding noise (-1)
        num_noise_points = np.sum(labels == -1)

        # Calculate clustering metrics if there are valid clusters
        if num_clusters > 1:
            silhouette_avg = silhouette_score(data_scaled[core_samples_mask], labels[core_samples_mask])
            davies_bouldin = davies_bouldin_score(data_scaled[core_samples_mask], labels[core_samples_mask])
            dunn_index = calculate_dunn_index(data_scaled[core_samples_mask], labels[core_samples_mask])
        else:
            silhouette_avg = -1
            davies_bouldin = -1
            dunn_index = -1

        # Store the results for each combination
        results_eps_min_samples.append({
            "eps": eps,
            "min_samples": min_samples,
            "num_clusters": num_clusters,
            "num_noise_points": num_noise_points,
            "Silhouette Score": silhouette_avg,
            "Davies-Bouldin Index": davies_bouldin,
            "Dunn Index": dunn_index
        })

# Convert results to a DataFrame for easy analysis
results_df = pd.DataFrame(results_eps_min_samples)

# Create heatmaps to visualize the impact of eps and min_samples on clustering metrics
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Heatmap for Number of Clusters
pivot_clusters = results_df.pivot("min_samples", "eps", "num_clusters")
sns.heatmap(pivot_clusters, cmap="viridis", annot=True, fmt="d", ax=axes[0, 0])
axes[0, 0].set_title("Number of Clusters (Excluding Noise)")
axes[0, 0].set_xlabel('eps')
axes[0, 0].set_ylabel('min_samples')

# Heatmap for Number of Noise Points
pivot_noise = results_df.pivot("min_samples", "eps", "num_noise_points")
sns.heatmap(pivot_noise, cmap="viridis", annot=True, fmt="d", ax=axes[0, 1])
axes[0, 1].set_title("Number of Noise Points")
axes[0, 1].set_xlabel('eps')
axes[0, 1].set_ylabel('min_samples')

# Heatmap for Silhouette Score
pivot_silhouette = results_df.pivot("min_samples", "eps", "Silhouette Score")
sns.heatmap(pivot_silhouette, cmap="coolwarm", annot=True, fmt=".2f", ax=axes[1, 0])
axes[1, 0].set_title("Silhouette Score")
axes[1, 0].set_xlabel('eps')
axes[1, 0].set_ylabel('min_samples')

# Heatmap for Davies-Bouldin Index
pivot_db = results_df.pivot("min_samples", "eps", "Davies-Bouldin Index")
sns.heatmap(pivot_db, cmap="coolwarm", annot=True, fmt=".2f", ax=axes[1, 1])
axes[1, 1].set_title("Davies-Bouldin Index")
axes[1, 1].set_xlabel('eps')
axes[1, 1].set_ylabel('min_samples')

plt.tight_layout()
plt.show()

# Display results for best performing parameters based on metrics
best_silhouette_score = results_df.loc[results_df['Silhouette Score'].idxmax()]
best_db_index = results_df.loc[results_df['Davies-Bouldin Index'].idxmin()]
best_dunn_index = results_df.loc[results_df['Dunn Index'].idxmax()]

print("Best Parameters based on Silhouette Score:", best_silhouette_score)
print("Best Parameters based on Davies-Bouldin Index:", best_db_index)
print("Best Parameters based on Dunn Index:", best_dunn_index)


# Define parameter grids for each method
kmeans_params = {'n_clusters': range(3 , 8)}
dbscan_params = {'eps': [0.1, 0.2, 0.5, 1.0, 1.3], 'min_samples': [5, 10]}
agg_params = {'n_clusters': range(3 , 8), 'linkage': ['ward', 'complete', 'average'], }

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

    core_samples_mask = labels != -1
    if np.sum(core_samples_mask) > 1:
        score = silhouette_score(data_scaled[core_samples_mask], labels[core_samples_mask])
        if score > best_dbscan_score:
            best_dbscan_score = score
            best_params['DBSCAN'] = params

dbscan = DBSCAN(**best_params['DBSCAN'])
labels_dbscan = dbscan.fit_predict(data_scaled)
core_samples_mask = labels_dbscan != -1
if np.sum(core_samples_mask) > 1:
    silhouette_avg_dbscan = silhouette_score(data_scaled[core_samples_mask], labels_dbscan[core_samples_mask])
    davies_bouldin_dbscan = davies_bouldin_score(data_scaled[core_samples_mask], labels_dbscan[core_samples_mask])
    dunn_index_dbscan = calculate_dunn_index(data_scaled[core_samples_mask], labels_dbscan[core_samples_mask])
else:
    silhouette_avg_dbscan = -1
    davies_bouldin_dbscan = -1
    dunn_index_dbscan = -1

print("DBSCAN Metrics (excluding noise):")
print(f"Silhouette Score: {silhouette_avg_dbscan}")
print(f"Davies-Bouldin Index: {davies_bouldin_dbscan}")
print(f"Dunn Index: {dunn_index_dbscan}")

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

# Visualize Clusters for Each Method (with 3 subplots)
# Store results for comparison
results = {}

# Define the clustering methods
methods = {
    "KMeans": KMeans(**best_params['KMeans'], random_state=42),
    "DBSCAN": DBSCAN(**best_params['DBSCAN']),
    "Agglomerative": AgglomerativeClustering(**best_params['Agglomerative']),
}

# Create a figure for the subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. Apply each method and evaluate quality metrics
for idx, (name, model) in enumerate(methods.items()):
    # Apply clustering model
    labels = model.fit_predict(data_scaled)

    # Visualize Clusters
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)

    # Scatter plot of clusters
    ax = axes[idx]  # Choose the correct subplot axis

    # Check if DBSCAN has noise points and label them as 'Noise'
    if -1 in labels:  # DBSCAN noise points
        ax.scatter(data_pca[labels == -1, 0], data_pca[labels == -1, 1], c='red', s=30, marker='x', label='Noise')

    # For regular points, add each cluster and label them
    scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6, label="Clusters")

    ax.set_title(f"{name} Clusters")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    # Add colorbar for cluster labels
    fig.colorbar(scatter, ax=ax, orientation='horizontal', fraction=0.02, pad=0.1)

    # Add legend
    ax.legend(loc="upper right")

    # Store quality metrics for comparison
    silhouette_avg = silhouette_score(data_scaled, labels) if len(np.unique(labels)) > 1 else -1
    davies_bouldin = davies_bouldin_score(data_scaled, labels) if len(np.unique(labels)) > 1 else -1
    dunn_index = calculate_dunn_index(data_scaled, labels) if len(np.unique(labels)) > 1 else -1

    results[name] = {
        "Silhouette Score": silhouette_avg,
        "Davies-Bouldin Index": davies_bouldin,
        "Dunn Index": dunn_index,
    }

# Show the plots
plt.show()


# 3. Display the results
print("Clustering Quality Metrics Comparison:")
for method, metrics in results.items():
    print(f"\n{method} Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

#Method with Best Scores: Based on the metrics you compute, select the clustering
# method with the highest Silhouette Score, the lowest Davies-Bouldin Index, and the highest Dunn Index.

# Convert results dictionary to DataFrame
results_df = pd.DataFrame.from_dict(results, orient="index")

# Add a column for the best metric per row
results_df["Best Metric"] = results_df.idxmax(axis=1)

# Display the summary table
print("Clustering Metrics Summary:")
print(results_df)

# Analyze DBSCAN cluster distribution
dbscan_cluster_counts = pd.Series(labels_dbscan).value_counts().sort_index()
noise_points = dbscan_cluster_counts[-1] if -1 in dbscan_cluster_counts else 0

print("DBSCAN Cluster Analysis:")
print(f"Total clusters (excluding noise): {len(dbscan_cluster_counts) - 1}")
print(f"Noise points: {noise_points}")
print("Cluster sizes:")
print(dbscan_cluster_counts)

# Compare to other methods
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# DBSCAN Clusters
axes[0].scatter(data_pca[:, 0], data_pca[:, 1], c=labels_dbscan, cmap='viridis', s=50, alpha=0.6)
axes[0].set_title("DBSCAN Clusters")
axes[0].set_xlabel("PCA Component 1")
axes[0].set_ylabel("PCA Component 2")

# Overlay comparison with KMeans
labels_kmeans = KMeans(**best_params['KMeans'], random_state=42).fit_predict(data_scaled)
axes[1].scatter(data_pca[:, 0], data_pca[:, 1], c=labels_kmeans, cmap='viridis', s=50, alpha=0.6)
axes[1].set_title("KMeans Clusters")
axes[1].set_xlabel("PCA Component 1")
axes[1].set_ylabel("PCA Component 2")

plt.tight_layout()
plt.show()

