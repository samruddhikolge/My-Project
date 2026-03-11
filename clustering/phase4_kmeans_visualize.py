import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv("/home/samruddhi/Project/data/kmeans_intent_clusters.csv")
X = df.drop(columns=["utterance", "cluster"]).values

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:,0], X_pca[:,1], c=df["cluster"], cmap="tab10")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("K-Means Intent Clusters (PCA)")
plt.show()
