import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn



data = pd.read_csv('customize_offerings.csv')
data.info()

x=data.values

#scale the data
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)

#Figure out optimal number of clusters required.
from sklearn.cluster import KMeans
cluster_range=range(1,10)
cluster_errors=[]

for num_clusters in cluster_range:
    clusters=KMeans(num_clusters)
    clusters.fit(x_scaled)
    cluster_errors.append(clusters.inertia_)
    
clusters_df=pd.DataFrame({"num_clusters":cluster_range,"cluster_errors":cluster_errors})
clusters_df[0:10]

plt.figure(figsize=(10,7))
plt.plot(clusters_df.num_clusters,clusters_df.cluster_errors,marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("wss value")
#elbo turning point is at clustere 4 

kmeans=KMeans(n_clusters=4)
kmeans.fit(x_scaled)
y_kmeans=kmeans.predict(x_scaled)    #predicts the cluster group for each row

centers = kmeans.cluster_centers_
centers

plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=y_kmeans)  #c means color
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.xlabel('Premium savings are important')
plt.ylabel('Agent not important') 

