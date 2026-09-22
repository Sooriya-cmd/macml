import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(r"C:\Users\Admin\Documents\sem5\ml\data7.csv")

num_cols = data.select_dtypes(include=['number']).columns
data[num_cols] = data[num_cols].fillna(data[num_cols].median())

data_encoded = pd.get_dummies(data, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_encoded)

k = 3

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
data['cluster'] = kmeans.fit_predict(X_scaled)

for cluster_id in range(k):
    print(f"\nCluster {cluster_id}:")
    print(data[data['cluster'] == cluster_id])

plt.figure(figsize=(8, 6))
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=data['cluster'],
    cmap='viridis',
    alpha=0.7
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    c='red',
    marker='X',
    label='Centroids'
)

plt.title("K-Means Clustering")
plt.xlabel("Feature 1 (Scaled)")
plt.ylabel("Feature 2 (Scaled)")
plt.legend()
plt.show()