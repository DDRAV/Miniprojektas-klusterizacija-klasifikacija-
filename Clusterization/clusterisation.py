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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances
from sklearn.decomposition import PCA
from sklearn_extra.cluster import dunn_score


path = "C:/Users/drawn/Mokymai/DDRAV Mokymai/Miniprojektas/Clusterization/Dry_Bean_Dataset.csv"

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


#Kmeans Clustering:

# Counting WCSS depending on cluster count
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Cluster count (k)")
plt.ylabel("WCSS")
plt.title("WCSS depending on cluster count")
plt.show()

# Choosing best cluster count based on wcss
optimal_kmeans = 6  # Choose based on graph
#4 clusters:
#Silhouette Score: 0.23571996908972387
#Davies-Bouldin Index: 1.4448610625416118
kmeans = KMeans(n_clusters=optimal_kmeans, random_state=42)
res_kmeans = kmeans.fit_predict(data_scaled)

# Evaluate using metrics
silhouette_avg_kmeans = silhouette_score(data_scaled, res_kmeans)
print(f"Silhouette Score: {silhouette_avg_kmeans}")

davies_bouldin_kmeans = davies_bouldin_score(data_scaled, res_kmeans)
print(f"Davies-Bouldin Index: {davies_bouldin_kmeans}")

distance_matrix = pairwise_distances(data_scaled)
dunn_kmeans = dunn_score(distance_matrix, res_kmeans)
print(f"Dunn Index: {dunn_kmeans}")


pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=res_kmeans, cmap='viridis', s=50, alpha=0.6)
plt.title("KMeans Clusters after PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster Labels")
plt.show()





