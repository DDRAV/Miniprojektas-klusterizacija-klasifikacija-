#Kode for dataset downloading
#import kagglehub
# Download latest version
#path = kagglehub.dataset_download("fatihb/coffee-quality-data-cqi")
#print("Path to dataset files:", path)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = "C:/Users/ddrav/OneDrive - Everwest/Desktop/Projektas/mok/Miniprojektas/Clusterization/df_arabica_clean.csv"

#1. Get dataset

df = pd.read_csv(path)
df.head()
print(df.head())

#2. Handling data
# Check for duplicate rows
duplicates = df[df.duplicated()]

# Print duplicates if any
if not duplicates.empty:
    print("Duplicate rows found:")
    print(duplicates)
else:
    print("No duplicate rows found.")

# Check for missing values
print(df.isnull().sum())

# Visualize missing data (optional, if you use seaborn)
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.show()

