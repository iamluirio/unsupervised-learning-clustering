# Unsupervised Learning Data Analysis: Clustering
**Clustering** is a type of unsupervised learning in machine learning and data analysis that involves grouping a set of objects in such a way that objects in the same group (called a **cluster**) are more similar to each other than to those in other groups (clusters). It is a common technique for statistical data analysis used in many fields, including machine learning, pattern recognition, image analysis, information retrieval, and bioinformatics.

The project contains two types of clustering approaches: **K-means** and **DBSCAN**, respectively.

## K-means Clustering
K-means is a popular clustering algorithm that **partitions the data into k clusters**, where k is a user-defined parameter. The initialisation starts with choosing **k initial centroids** randomly from the data points. It assigns each data point to the nearest centroid, forming k clusters, and recalculates the centroids of the k clusters by taking the mean of all data points in each cluster. It repeats the assignment and update steps until the centroids no longer change or change very little (i.e., the algorithm converges).

The objective of K-means is to minimize the **within-cluster sum of squares (WCSS)**, which measures the variance within each cluster.


