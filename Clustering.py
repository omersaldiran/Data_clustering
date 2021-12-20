#!/usr/bin/env python
# coding: utf-8

# # Applying Kmeans Cluster to the IRIS dataset

# First import the necessary libraries

# In[34]:


from sklearn.datasets import make_blobs
import numpy as np
import time
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pylab as pl
from IPython import display
from sklearn.cluster import KMeans
import pandas as pd


# Import the dataset

# In[35]:


dataset = pd.read_csv('iris.data')


# In[36]:


x = dataset.iloc[:, [1, 2, 3]].values


# Applying the Kmeans clustering with random_state is 1.
# SSE is 47.

# In[37]:


converged = False
n_clusters = 3
random_state = 1
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=1, max_iter=1, init="random").fit(x)
prev_inertia = kmeans.inertia_
i = 2
while not converged:
    plt.clf()
    y_pred = kmeans.predict(x)
    plt.scatter(x[:, 0], x[:, 1], c=y_pred);
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker="x", c="r", s=100)
    plt.axis("equal")
    plt.text(10,0,"iter: {} \nSSE: {}".format(i-1, int(prev_inertia)))
    display.display(pl.gcf())
    display.clear_output(wait=True)
    time.sleep(2)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=1, max_iter=i, init="random").fit(x)
    converged = prev_inertia == kmeans.inertia_ 
    prev_inertia = kmeans.inertia_
    i=i+1


# Applying the Kmeans clustering with random_state is 2.
# SSE is 100.

# In[38]:


converged = False
n_clusters = 3
random_state = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=1, max_iter=1, init="random").fit(x)
prev_inertia = kmeans.inertia_
i = 2
while not converged:
    plt.clf()
    y_pred = kmeans.predict(x)
    plt.scatter(x[:, 0], x[:, 1], c=y_pred);
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker="x", c="r", s=100)
    plt.axis("equal")
    plt.text(10,0,"iter: {} \nSSE: {}".format(i-1, int(prev_inertia)))
    display.display(pl.gcf())
    display.clear_output(wait=True)
    time.sleep(2)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=1, max_iter=i, init="random").fit(x)
    converged = prev_inertia == kmeans.inertia_ 
    prev_inertia = kmeans.inertia_
    i=i+1


# Applying the Kmeans clustering with random_state is 3.
# SSE is 48.

# In[39]:


converged = False
n_clusters = 3
random_state = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=1, max_iter=1, init="random").fit(x)
prev_inertia = kmeans.inertia_
i = 2
while not converged:
    plt.clf()
    y_pred = kmeans.predict(x)
    plt.scatter(x[:, 0], x[:, 1], c=y_pred);
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker="x", c="r", s=100)
    plt.axis("equal")
    plt.text(10,0,"iter: {} \nSSE: {}".format(i-1, int(prev_inertia)))
    display.display(pl.gcf())
    display.clear_output(wait=True)
    time.sleep(2)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=1, max_iter=i, init="random").fit(x)
    converged = prev_inertia == kmeans.inertia_ 
    prev_inertia = kmeans.inertia_
    i=i+1


# Applying the Kmeans clustering with random_state is 4.
# SSE is 100.

# In[40]:


converged = False
n_clusters = 3
random_state = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=1, max_iter=1, init="random").fit(x)
prev_inertia = kmeans.inertia_
i = 2
while not converged:
    plt.clf()
    y_pred = kmeans.predict(x)
    plt.scatter(x[:, 0], x[:, 1], c=y_pred);
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker="x", c="r", s=100)
    plt.axis("equal")
    plt.text(10,0,"iter: {} \nSSE: {}".format(i-1, int(prev_inertia)))
    display.display(pl.gcf())
    display.clear_output(wait=True)
    time.sleep(2)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=1, max_iter=i, init="random").fit(x)
    converged = prev_inertia == kmeans.inertia_ 
    prev_inertia = kmeans.inertia_
    i=i+1


# Applying the Kmeans clustering with random_state is 5.
# SSE is 48.

# In[41]:


converged = False
n_clusters = 3
random_state = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=1, max_iter=1, init="random").fit(x)
prev_inertia = kmeans.inertia_
i = 2
while not converged:
    plt.clf()
    y_pred = kmeans.predict(x)
    plt.scatter(x[:, 0], x[:, 1], c=y_pred);
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker="x", c="r", s=100)
    plt.axis("equal")
    plt.text(10,0,"iter: {} \nSSE: {}".format(i-1, int(prev_inertia)))
    display.display(pl.gcf())
    display.clear_output(wait=True)
    time.sleep(2)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=1, max_iter=i, init="random").fit(x)
    converged = prev_inertia == kmeans.inertia_ 
    prev_inertia = kmeans.inertia_
    i=i+1


# If we compare the random_state number and SSE, random_state = 1 is the best choose.
# And the comparing the original dataset, we have 49,50,50 member of each cluster. But if we check the Kmeans prediction, we have 49,54,46 member of each cluster.
# They are looking natural clusters.
# 

# 3D plotting can be seen in here.

# In[42]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(4,4))

ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:, 0], x[:, 1], c=y_pred);
ax.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker="x", c="r", s=100)
plt.show()

