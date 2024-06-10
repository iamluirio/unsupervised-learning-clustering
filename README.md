# Unsupervised Learning Data Analysis: DBSCAN Clustering Algorithm
<div align="left">
<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" />
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white" />
<div/>

**Clustering** is a type of unsupervised learning in machine learning and data analysis that involves grouping a set of objects in such a way that objects in the same group (called a **cluster**) are more similar to each other than to those in other groups (clusters). It is a common technique for statistical data analysis used in many fields, including machine learning, pattern recognition, image analysis, information retrieval, and bioinformatics.

The project contains the [**DBSCAN Clustering algorithm**](https://it.wikipedia.org/wiki/Dbscan): it exploits the [**Euclidean distance**](https://en.wikipedia.org/wiki/Euclidean_distance) as a default metric to group connected points for density. Euclidean distance is a common measure used to calculate the distance between two points in the Euclidean space.

In addition to the traditional approach, I used a different metric for the grouping of the elements in the formats trained: we want to change the algorithm to obtain a clustering based not on the similarity among the features of feature, but **on the similarity of each sample with the other samples**. To obtain this goal, I implemented the DBSCAN algorithm using the [**Cosine Similarity**](https://en.wikipedia.org/wiki/Cosine_similarity) metric: it's a measure that evaluates the corner between two vectors, indicating how similar they are in the direction. In the context of clustering or data analysis, the similarity of the cosine can be used to calculate how similar two samples are based on their characteristics.

We then appear the different models, to observe the results.


## Dataset Import
I based my analysis on [**_Mall_Customers.csv_**](https://www.kaggle.com/datasets/shwetabh123/mall-customers) downloadable from [**Kaggle**](https://www.kaggle.com/).

The dataset contains the data of **200 customers of a generic mall**, with 5 columns: **the customer's ID, Gender, Age, Annual income (in k$) and Spending Score (1-100)**.

| Customer's ID | Gender | Age | Annual Income (in k$) | Spending Score (1-100) |
|---------------|--------|-----|-----------------------|------------------------|
| 1             | Male   | 19  | 15                    | 39                     |
| 2             | Male   | 21  | 15                    | 81                     |
| 3             | Female | 20  | 16                    | 6                      |
| 4             | Female | 23  | 16                    | 77                     |
| 5             | Female | 31  | 17                    | 40                     |

## Traditional Approach
```python 
standard_matrix = df.iloc[:,[3,4]].values
```

We extract the _Annual Income_ and _Spending Score_ columns from the dataframe to apply the algorithm, using these two features to compare the different samples, and to be able to group clusters of samples.

This is the classic approach based on **based on the similarities between the feature vectors**. 
If you are interested in the traditional Euclidean Distance based DBSCAN approach,, please refer to **metti file tradizionale**.

## Innovative Approach
We want to modify the algorithm to obtain a clustering based not on the similarities between the feature vectors, but **on the similarities of each sample with the other samples**.

We then compare the two different models, to observe the results.

### Cosine Similarity Metric
```cosine_similarity``` is a function that calculates the **cosine similarity between vectors**. It is often used to calculate the similarity between samples or objects in a multidimensional space. Cosine similarity is a measure that evaluates the angle between two vectors, indicating how similar they are in direction.

In the context of clustering or data analysis, cosine similarity can be used to calculate how similar two samples are based on their characteristics. Values ​​closer to 1 indicate that the samples are similar, while values ​​closer to 0 indicate that the samples are different.
![Screenshot from 2024-06-10 17-12-22](https://github.com/iamluirio/unsupervised-learning-clustering/assets/118205581/7a16c134-f92b-414c-82ae-c7e7fc766361)


<div align="center">
  <img src="https://github.com/iamluirio/unsupervised-learning-clustering/assets/118205581/e2a59896-cb5c-4515-a3d5-0d7181fe5801" />
<div/>

### Ideal Fundamental Parameters
These parameters are essential for the operation of DBSCAN and will influence the shape and size of the clusters identified.

- **eps**: This is the **maximum radius** around each point that will be considered during the clustering process. It is one of the key parameters of DBSCAN and **determines the "maximum distance" between points** to define if they belong to the same cluster.

- **min_samples**: This parameter represents **the minimum number of points that must be present within the eps radius** for a point to be considered as a **"core point"**. Core points are those that are central in a cluster.

**Exhaustive research** to find the best values ​​of _eps_ and _min_samples_ parameters for the DBSCAN algorithm. The metric used to evaluate the different parameter combinations is [**the Silhouette score**](https://en.wikipedia.org/wiki/Silhouette_(clustering)).

```python 
best_score = -1
best_eps = None
best_min_samples = None

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(standard_matrix)
        labels = dbscan.labels_
        if len(set(labels)) > 1:
            score = silhouette_score(standard_matrix, labels)
            
            if score > best_score:
                best_score = score
                best_eps = eps
                best_min_samples = min_samples

print(f"Best score: {best_score}")
print(f"Best eps: {best_eps}")
print(f"Best min_samples: {best_min_samples}")
```

For example, with the following output: 

```
Best score: 0.4148124800517594
Best eps: 0.4
Best min_samples: 8
```

- A silhouette score of 0.4148 indicates a moderate level of clustering quality, where clusters are reasonably well-separated and cohesive.
- The 0.4 eps (epsilon) parameter defines the maximum distance between two samples for them to be considered as in the same neighborhood.
- The 8 value parameter defines the minimum number of samples in a neighborhood for a point to be considered a core point.

### DBSCAN Model
Assigning the previous parameters, we can create a **DBSCAN istance**:

```python
model1 = DBSCAN(eps=0.4,min_samples=8)
clusters1 = model1.fit_predict(standard_matrix)
```

We also calculate the **Homogeneity and Heterogeneity** values:
- **Homogeneity** measures how similar the points within a cluster are to each other in terms of their characteristics. A high value of homogeneity indicates that the points within a cluster are very similar to each other. If a cluster contains only samples from a single class, then the homogeneity is maximum (1.0).

- **Heterogeneity** measures how distinct and separated the points of different clusters are. A low value of heterogeneity indicates that the clusters are well separated and distinct from each other. A high value of heterogeneity may indicate overlap between clusters or the presence of very similar clusters.

```python
# Display cluster properties
n_clusters1 = len(set(clusters1)) - (1 if -1 in clusters1 else 0)
n_noise1 = list(clusters1).count(-1)
print('Number of clusters:', n_clusters1)
print('Number of noise points:', n_noise1)

for i in range(n_clusters1):
    print('Cluster', i+1, ':')
    cluster_size = len(standard_matrix[clusters1 == i])
    print('Number of observations:', cluster_size)
    cluster_homogeneity = np.sum(clusters1 == i) / cluster_size
    print('Homogeneity:', cluster_homogeneity)
    cluster_heterogeneity = np.sum(clusters1 != i) / (len(standard_matrix) - cluster_size)
    print('Heterogeneity:', cluster_heterogeneity)
    print('------------------------')
```

# Cluster 
