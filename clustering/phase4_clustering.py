# ================================================
# Phase 4: Intent Clustering (Low-Memory Edition)
# ================================================
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load only necessary columns
# -------------------------------
features_path = '/home/samruddhi/Project/data/phase3_features.csv'
print(" Loading data efficiently...")

# Read a small sample to detect column names
sample = pd.read_csv(features_path, nrows=5)
available_cols = list(sample.columns)
feature_cols = [c for c in available_cols if c.startswith('emb_')]

# Optional columns (load only if they exist)
optional_cols = [c for c in ['is_question', 'is_user', 'utterance'] if c in available_cols]

use_cols = feature_cols + optional_cols
print(f" Using columns: {use_cols[:10]}{'...' if len(use_cols) > 10 else ''}")

# Load only selected columns
df = pd.read_csv(features_path, usecols=use_cols)
print(f" Loaded data with shape: {df.shape}")

# -------------------------------
# 2. Downsample for safety
# -------------------------------
MAX_SAMPLES = 5000  # reduce to prevent memory overflow
if len(df) > MAX_SAMPLES:
    df = df.sample(MAX_SAMPLES, random_state=42).reset_index(drop=True)
    print(f" Using subset of {len(df)} samples for clustering.")

# -------------------------------
# 3. Convert to NumPy array (float32)
# -------------------------------
feature_cols = [c for c in df.columns if c.startswith('emb_')]
X = df[feature_cols].to_numpy(dtype=np.float32)
print(f"Matrix shape: {X.shape}, dtype: {X.dtype}")

# -------------------------------
# 4. Dimensionality reduction (PCA)
# -------------------------------
print("Running PCA (20 components)...")
pca = PCA(n_components=20, random_state=42)
X_reduced = pca.fit_transform(X)
print(f"PCA reduced shape: {X_reduced.shape}")

# -------------------------------
# 5. Intent clustering (HDBSCAN)
# -------------------------------
print(" Running HDBSCAN clustering...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=25, metric='euclidean', cluster_selection_method='eom')
df['cluster'] = clusterer.fit_predict(X_reduced)

print("Clustering complete!")
print(df['cluster'].value_counts())

# -------------------------------
# 6. Save clustered results
# -------------------------------
output_csv = '/home/samruddhi/Project/data/phase4_clustered_intents_small.csv'
df.to_csv(output_csv, index=False)
print(f" Saved results to {output_csv}")

# -------------------------------
# 7. Optional: Visualize clusters
# -------------------------------
try:
    print("Plotting 2D visualization using PCA components...")
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=X_reduced[:, 0], y=X_reduced[:, 1],
        hue=df['cluster'], palette='tab10', s=20, alpha=0.7
    )
    plt.title("Intent Clusters (PCA + HDBSCAN)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"(Skipped plot due to: {e})")

# -------------------------------
# 8. Print sample utterances per cluster
# -------------------------------
if 'utterance' in df.columns:
    print("\nSample utterances per cluster:")
    for c in df['cluster'].unique()[:5]:  # show first 5 clusters
        samples = df[df['cluster'] == c]['utterance'].head(3).tolist()
        print(f"\nCluster {c}:")
        for s in samples:
            print("   •", s[:120])
else:
    print("\n No utterance column found — skipping example printing.")
