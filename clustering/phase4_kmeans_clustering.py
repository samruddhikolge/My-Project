import pandas as pd
from sklearn.cluster import KMeans

# Load embeddings
df = pd.read_csv("/home/samruddhi/Project/data/intent_embeddings.csv")
X = df.drop(columns=["utterance"]).values

# Apply K-Means
k = 6
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X)

# Save result
df.to_csv("/home/samruddhi/Project/data/kmeans_intent_clusters.csv", index=False)
print("K-Means clustering done with k =", k)
