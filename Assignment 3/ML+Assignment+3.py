
# coding: utf-8

# # K-MEANS

# Clustering is a technique for finding similarity groups in a data, called clusters. It attempts to group individuals in a population together by similarity, but not driven by a specific purpose. 
# K-means is a famous centroid based clustering algorithm where each cluster is assigned with a centre called a centroid and each point is assigned to a cluster with the closest centroid.

# In[ ]:


# IMPORTING LIBRARIES
from sklearn.cluster import KMeans
# from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


# READING THE DATASET
data = pd.read_csv("xclara.csv") 
data.head()


# In[ ]:


# Storing Instance values into variables
X1 = data['V1'].values
X2 = data['V2'].values

# Using an Iterator to create a list with 
# equal no elements of both the iteables.
X = np.array(list(zip(X1, X2)))
plt.figure(figsize=(7, 7), dpi=80)
plt.scatter(X1, X2, c='blue', s=7,alpha=0.4)


# In[10]:


K=[2,3,4,5,6,7]
for i in K: 
    kmeans = KMeans(n_clusters=i)
# Fitting the input data
    kmeans = kmeans.fit(X)
# Getting the cluster labels
    labels = kmeans.predict(X)
# Centroid values
    centroids = kmeans.cluster_centers_
# Comparing with scikit-learn centroids
    
    print(centroids) # From sci-kit learn
    plt.figure(figsize=(9, 9), dpi=80)
    plt.scatter(X[:, 0], X[:, 1], c=labels,alpha=0.3,s=50)
    centers = kmeans.cluster_centers_ 
    plt.scatter(centers[:, 0], centers[:, 1], c='red',marker='*', s=200, alpha=0.9)
    plt.show()


# # K-MEANS++

# It is somewhat similar to the Simple K-Means Clustering but instead of selecting all the required centroids randomly, we select the first one randomly, then find the points that are farthest to the first center{These points most probably do not belong to the first cluster center as they are far from it} and assign the second cluster center nearby those far points, and so on.
# This introduces an overhead in the initialization of the algorithm, but it reduces the probability of a bad initialization leading to a bad clustering result.
