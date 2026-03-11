import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Load clustered data
# -------------------------------
data_path = '/home/samruddhi/Project/data/phase4_clustered_intents_small.csv'
df = pd.read_csv(data_path)

# Filter valid clusters
df = df[df['cluster'] != -1]  # remove noise points (HDBSCAN assigns -1 to noise)
feature_cols = [c for c in df.columns if c.startswith('emb_')]

print(f"Loaded {len(df)} samples across {df['cluster'].nunique()} clusters.")

X = df[feature_cols].to_numpy(dtype=np.float32)
labels = df['cluster'].to_numpy()

# -------------------------------
# 1 Evaluate with Silhouette Score
# -------------------------------
if len(np.unique(labels)) > 1:
    sil_score = silhouette_score(X, labels)
    print(f" Silhouette Score: {sil_score:.3f} (higher = better clustering)")
else:
    print(" Only one cluster found — cannot compute silhouette score.")

# -------------------------------
# 2 Compute average cluster sizes
# -------------------------------
cluster_sizes = df['cluster'].value_counts()
print("\n Cluster Size Summary:")
print(cluster_sizes.describe())

# -------------------------------
# 3 Visualize clusters (2D PCA)
# -------------------------------
print("\n Creating 2D PCA Visualization...")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df, x='pca1', y='pca2', hue='cluster',
    palette='tab10', s=25, alpha=0.8, linewidth=0
)
plt.title("Intent Clusters — PCA Projection")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# -------------------------------
# 4 Optional: Cluster Centroid Heatmap
# -------------------------------
print("\n Generating cluster centroid visualization...")

centroids = df.groupby('cluster')[feature_cols].mean()
centroid_df = pd.DataFrame(centroids)

plt.figure(figsize=(10, 5))
sns.heatmap(centroid_df.iloc[:, :15], cmap='viridis', cbar=True)
plt.title("Cluster Centroid Heatmap (first 15 embedding dims)")
plt.show()
