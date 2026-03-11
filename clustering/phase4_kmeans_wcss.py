import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load embeddings
df = pd.read_csv("/home/samruddhi/Project/data/intent_embeddings.csv")
X = df.drop(columns=["utterance"]).values

# WCSS (Elbow Method)
wcss = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(K, wcss, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method for Intent Clustering")
plt.show()
