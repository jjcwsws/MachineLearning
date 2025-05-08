# import matplotlib.pyplot as plt
# from sklearn.datasets import load_wine
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# wine = load_wine()
# X = wine.data
# scaler= StandardScaler()
# X_scaled = scaler.fit_transform(X)
# kmeans= KMeans(n_clusters=3, random_state=42)
# kmeans.fit(X_scaled)
# labels = kmeans.labels_
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)
# centroids = kmeans.cluster_centers_
# centroids_pca =pca.transform(centroids)
# plt.figure(figsize=(10,6))
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', edgecolor='k')
# plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='*', s=200, label='Centroids')
# plt.title('K-means Clustering of Wine Dataset')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.show()


from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
wine = load_wine ()
X =wine.data
scaler = StandardScaler()
X =scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=20,n_init=10)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, color='red')
plt.scatter(X[:,0],X[:,1],c=labels, s=10)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Kmeans(Sklearn)')
plt.show()
