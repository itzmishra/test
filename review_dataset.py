# quick_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load your master dataset
df = pd.read_csv('new_healthy_engine_features.csv')

print("=== DATASET OVERVIEW ===")
print(f"Total samples: {len(df)}")
print(f"Healthy (0): {len(df[df['label']==0])}")
print(f"Unhealthy (1): {len(df[df['label']==1])}")
print(f"Feature count: {len(df.columns) - 2}")  # minus label and filename

print("\n=== FEATURE STATISTICS ===")
print(df.describe())

print("\n=== CHECK FOR ISSUES ===")
print("Missing values:", df.isnull().sum().sum())
print("Infinite values:", np.isinf(df.select_dtypes(include=[np.number])).sum().sum())

# Plot class distribution
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
df['label'].value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Label (0=Healthy, 1=Unhealthy)')
plt.ylabel('Count')

# Plot feature correlations with label
plt.subplot(1, 2, 2)
correlations = df.corr()['label'].drop('label').sort_values()
correlations.head(10).plot(kind='barh')
plt.title('Top 10 Features Correlated with Label')
plt.tight_layout()
plt.show()