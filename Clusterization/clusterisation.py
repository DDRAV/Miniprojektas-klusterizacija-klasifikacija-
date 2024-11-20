#kodas duomenu rinikio atisuntimui
#import kagglehub
# Download latest version
#path = kagglehub.dataset_download("fatihb/coffee-quality-data-cqi")
#print("Path to dataset files:", path)

import pandas as pd

df = pd.read_csv("C:/Users/ddrav/OneDrive - Everwest/Desktop/Projektas/mok/Miniprojektas/Clusterization/df_arabica_clean.csv")
df.head()
