import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from scipy.cluster.hierarchy import linkage, dendrogram

df = pd.read_csv("datasets/USArrests.csv", index_col=0)
df.head()
df.isnull().sum()
df.info()
df.describe().T

sc = MinMaxScaler((0,1))
df = sc.fit_transform(df)
df[0:5]

# K-Means
kmeans = KMeans(n_clusters=4, random_state=17).fit(df)
kmeans.get_params()

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
kmeans.inertia_

#Determining the Optimum Number of Clusters
#1
kmeans = KMeans()
ssd = []
K = range(1, 30)
for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)
plt.plot(K, ssd, "bx-")
plt.xlabel("SSE/SSR/SSD vs. Different K Values")
plt.title("Elbow Method for Optimum Number of Clusters")
plt.show()

#2 KElbowVisualizer()
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()
elbow.elbow_value_ #5


#Creating Final Clusters
kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)
df[0:5]

clusters_kmeans = kmeans.labels_
df = pd.read_csv("datasets/USArrests.csv", index_col=0)

df["cluster"] = clusters_kmeans
df["cluster"] = df["cluster"] + 1
df.head()

df[df["cluster"] == 5]

df.groupby("cluster").agg(["count","mean","median"])
df.to_csv("clusters.csv")

# Hierarchical Clustering
df = pd.read_csv("datasets/USArrests.csv", index_col=0)
sc = MinMaxScaler((0,1))
df = sc.fit_transform(df)

hc_average = linkage(df, "average")
plt.figure(figsize=(10, 5))
plt.title("Dendogram")
plt.xlabel("Observation Units")
plt.ylabel("Distances")
dendrogram(hc_average, leaf_font_size=10)
plt.show()

plt.figure(figsize=(7, 5))
plt.title("Dendogram")
plt.xlabel("Observation Units")
plt.ylabel("Distances")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.show()

#determine the number of clusters
plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.axhline(y=0.6, color='b', linestyle='--')
plt.show()

#Creating the Final Model
cluster = AgglomerativeClustering(n_clusters=5, linkage="average")
clusters = cluster.fit_predict(df)
df = pd.read_csv("datasets/USArrests.csv", index_col=0)

df["hi_cluster_no"] = clusters
df["hi_cluster_no"] = df["hi_cluster_no"] + 1

df["kmeans_cluster_no"] = clusters_kmeans
df["kmeans_cluster_no"] = df["kmeans_cluster_no"] + 1









