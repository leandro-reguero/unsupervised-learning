# Importing libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# clustering
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

# metricas
from sklearn.metrics import silhouette_score

# reduccion de dimensionalidad
from sklearn.decomposition import PCA
from pickle import dump

import warnings
warnings.filterwarnings('ignore')

# reading the data
rawdata = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv')
rawdata.head()
rawdata.shape
# saving the data
rawdata.to_csv('../data/raw/rawdata.csv')
df = rawdata[['MedInc', 'Latitude', 'Longitude']]
df.head()

# splitting dataset for train and test
X = df
X_train, X_test = train_test_split(X, test_size=0.2, random_state=21)
X_train

# training the kmeans model
model = KMeans(n_clusters=6, random_state=21)
model.fit(X_train)

# creating centroids and cluster lables
centroids = model.cluster_centers_
lables = model.labels_

# assigining clusters column
X_train['cluster'] = lables

predisione = model.predict(X_test) 
predisione
X_test['cluster'] = predisione

# now we plot the results
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
# Scatter plot para el conjunto completo (X_train)
scatter = ax.scatter(X_train['Longitude'], X_train['Latitude'], X_train['MedInc'], 
                     c=X_train['cluster'], cmap='viridis', s=10, label='Conjunto de Entrenamiento', alpha=0.1)
# Scatter plot para el conjunto de prueba (X_test)
# Asumiendo que X_test tiene las mismas columnas y una columna 'cluster' para el color
scatter_test = ax.scatter(X_test['Longitude'], X_test['Latitude'], X_test['MedInc'], 
                          c=X_test['cluster'], cmap='viridis', marker='x', s=30, label='Conjunto de Prueba', alpha=0.5)
ax.set_title('K-Means Clustering (3D)')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('MedInc')
# Ángulo de vista
ax.view_init(elev=20, azim=30)
# Leyenda
legend = ax.legend(*scatter.legend_elements(), title='Clusters')
ax.add_artist(legend)
# Agregar leyenda para el conjunto de prueba
plt.legend(loc='upper right')
plt.show()


# we unite the parts
X = pd.concat((X_train, X_test), axis=0)
X = X.sort_index(axis=0)

# ! Finding the optimal amount of clusters with inertia
# definir rango de valores para k
k_values = range(2, 21)
inertias = []

# calcular inercia para cada valor de k
for k in k_values:
    modelito = KMeans(n_clusters=k, random_state=21)
    modelito.fit(X)
    inertias.append(modelito.inertia_)

# Imprimir las inercias para cada k
for k, inertia in zip(k_values, inertias):
    print(f'k = {k}, Inercia = {inertia}')


# Graficar la inercia en función de k
plt.plot(k_values, inertias, marker='o')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.title('Inercia vs. Número de Clusters')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Calcular la inercia y el índice de silueta para cada valor de k
inertias2 = []
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=21)
    kmeans.fit(X)
    inertias2.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Imprimir las inercias y los índices de silueta para cada k
for k, inertia, silhouette in zip(k_values, inertias2, silhouette_scores):
    print(f'k = {k}, Inercia = {inertia}, Índice de Silueta = {silhouette}')


# ! plotting all together
# Graficar la inercia en función de k
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_values, inertias2, marker='o')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.title('Inercia vs. Número de Clusters')
plt.xticks(k_values)
plt.grid(True)

# Graficar el índice de silueta en función de k
plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Índice de Silueta')
plt.title('Índice de Silueta vs. Número de Clusters')
plt.xticks(k_values)
plt.grid(True)

plt.tight_layout()
plt.show()

# ! saving the model
dump(kmeans, open('../models/kmeans_clustering.pkl', 'wb'))
