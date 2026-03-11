import pandas as pd
import numpy as np

df = pd.read_csv("/home/samruddhi/Project/data/kmeans_intent_clusters.csv")

centroids = {}

for cluster_id in df["cluster"].unique():
    cluster_data = df[df["cluster"] == cluster_id]
    embeddings = cluster_data.drop(columns=["utterance", "cluster"]).values
    centroids[cluster_id] = embeddings.mean(axis=0)

np.save("/home/samruddhi/Project/data/cluster_centroids.npy", centroids)
print("Cluster centroids regenerated")
