# Unsupervised Learning Data Analysis: DBSCAN Clustering Algorithm
<div align="left">
<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" />
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white" />
<div/>

**Clustering** is a type of unsupervised learning in machine learning and data analysis that involves grouping a set of objects in such a way that objects in the same group (called a **cluster**) are more similar to each other than to those in other groups (clusters). It is a common technique for statistical data analysis used in many fields, including machine learning, pattern recognition, image analysis, information retrieval, and bioinformatics.

The project contains the [**DBSCAN Clustering algorithm**](https://it.wikipedia.org/wiki/Dbscan): it exploits the [**Euclidean Distance**](https://en.wikipedia.org/wiki/Euclidean_distance) as a default metric to group connected points for density. Euclidean distance is a common measure used to calculate the distance between two points in the Euclidean space.

In addition to the traditional approach, I used a different metric for the grouping of the elements in the formats trained: we want to change the algorithm to obtain a clustering **based not on the similarity among the features of feature**, but **on the similarity of each sample with the other samples**. To obtain this goal, I implemented the DBSCAN algorithm using the [**Cosine Similarity**](https://en.wikipedia.org/wiki/Cosine_similarity) metric: it's a measure that evaluates the corner between two vectors, indicating how similar they are in the direction. In the context of clustering or data analysis, the similarity of the cosine can be used to calculate how similar two samples are based on their characteristics.

## :ledger: Index
- [Dataset Import](#dataset-import)
- [Project Structure](https://github.com/iamluirio/face-morphing-java-android?tab=readme-ov-file#project-structure)
- [Morphing Process](https://github.com/iamluirio/face-morphing-java-android?tab=readme-ov-file#morphing-process)
  - [Identifying the Facial Landmarks](https://github.com/iamluirio/face-morphing-java-android?tab=readme-ov-file#identifying-the-facial-landmarks)
  - [Delaunay Triangulation](https://github.com/iamluirio/face-morphing-java-android?tab=readme-ov-file#delaunay-triangulation)
  - [Calculating Linear Interpolation for Morphed Image](https://github.com/iamluirio/face-morphing-java-android?tab=readme-ov-file#calculating-linear-interpolation-for-morphed-image)
  - [Getting Delaunay Indexes and Reshaping](https://github.com/iamluirio/face-morphing-java-android?tab=readme-ov-file#getting-delaunay-indexes-and-reshaping)
  - [Morphing Triangles and Images](https://github.com/iamluirio/face-morphing-java-android?tab=readme-ov-file#morphing-triangles-and-images)
  - [Results](https://github.com/iamluirio/face-morphing-java-android?tab=readme-ov-file#results)

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
If you are interested in the traditional Euclidean Distance based DBSCAN approach, please refer to **metti file tradizionale**.

## Innovative Approach
We want to modify the algorithm to obtain a clustering based not on the similarities between the feature vectors, but **on the similarities of each sample with the other samples**.

### Cosine Similarity Metric
```cosine_similarity``` is a function that calculates the [**cosine similarity between vectors**](https://en.wikipedia.org/wiki/Cosine_similarity). It is often used to calculate the similarity between samples or objects in a multidimensional space. Cosine similarity is a measure that evaluates the angle between two vectors, indicating how similar they are in direction.

In the context of clustering or data analysis, cosine similarity can be used to calculate how similar two samples are based on their characteristics. Values ​​closer to 1 indicate that the samples are similar, while values ​​closer to 0 indicate that the samples are different.

<div align="center">
<img src="https://github.com/iamluirio/unsupervised-learning-clustering/assets/118205581/7a16c134-f92b-414c-82ae-c7e7fc766361" alt="Screenshot" width="300"/>
</div>

```python 
# We extract numeric features excluding 'CustomerID'
numeric_features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# We extract 'Gender' column and we convert it in a dummy variable
gender_dummies = pd.get_dummies(df['Gender'], drop_first=True)

features = pd.concat([numeric_features, gender_dummies], axis=1)

# Similarity matrix calculation
cosine_matrix = cosine_similarity(features)
```

```
Similarity Matrix:
[[1.         0.96940834 0.70708157 ... 0.78406032 0.51687605 0.78357168]
 [0.96940834 1.         0.51241578 ... 0.67283958 0.34713448 0.67909188]
 [0.70708157 0.51241578 1.         ... 0.78778502 0.78795584 0.76742685]
 ...
 [0.78406032 0.67283958 0.78778502 ... 1.         0.92466312 0.99944667]
 [0.51687605 0.34713448 0.78795584 ... 0.92466312 1.         0.91818242]
 [0.78357168 0.67909188 0.76742685 ... 0.99944667 0.91818242 1.        ]]
```

In the **similarity matrix**, each value represents **how similar two samples are to each other based on the specified conditions and relationships**. In our case, **we consider the gender of the samples along with the differences in the other properties**, and calculate a similarity measure based on these factors.

The values ​​of the similarity matrix can range from 0 to 1, where:
- **0** represents **no similarity**: Two samples are not similar based on the conditions and relationships considered.
- **1** represents **maximum similarity**: Two samples are very similar based on the conditions and relationships considered.

Since we have combined the similarity of the gender with the similarity of the differences in the other properties, the values ​​in the matrix will reflect how similar the samples are based on this combination of factors.

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
        dbscan.fit(cosine_matrix)
        labels = dbscan.labels_
        if len(set(labels)) > 1:
            score = silhouette_score(cosine_matrix, labels)
            
            if score > best_score:
                best_score = score
                best_eps = eps
                best_min_samples = min_samples
```

For example, with the following output: 

```
Best score: 0.5107709022017272
Best eps: 0.4
Best min_samples: 4
```

- A silhouette score of 0.5107709022017272 indicates a moderate level of clustering quality, where clusters are reasonably well-separated and cohesive.
- The 0.4 eps (epsilon) parameter defines the maximum distance between two samples for them to be considered as in the same neighborhood.
- The 4 value parameter defines the minimum number of samples in a neighborhood for a point to be considered a core point.

### DBSCAN Model
Assigning the previous parameters, we can create a **DBSCAN istance**:

```python
model2 = DBSCAN(eps=0.4,min_samples=4)
clusters2 = model1.fit_predict(cosine_matrix)
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

For each cluster, we display its caratheristics:

```
Number of clusters: 3
Number of noise points: 20
Cluster 1 :
Number of observations: 20
Homogeneity: 1.0
Heterogeneity: 1.0
```

Label values ​​can be integers, where points with the same label belong to the same cluster. Outliers, which do not belong to any cluster, will be labeled -1.

### Visualizing and Comparing Clusters
Considering we are working with high-dimensional data, we consider **dimension reduction using techniques such as [PCA](https://it.wikipedia.org/wiki/Analisi_delle_componenti_principali) and [t-SNE](https://it.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) before plotting the data**. This helps **visualize the clusters in a two-dimensional plane**.

Main goal of **PCA** is to **transform a set of correlated variables into a new set of uncorrelated variables**, called **"principal components"**. This process allows to reduce the dimensionality of the data, that is, to go from a large number of original variables to a smaller number of principal components, while preserving most of the relevant information.

```python
# PCA 2-dimensional reduce
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(cosine_matrix)

plt.figure(figsize=(8, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=clusters, cmap='rainbow', marker='o', alpha=0.7)
plt.scatter(reduced_features[clusters == 0, 0], reduced_features[clusters == 0, 1], color='green', marker='o', label='Cluster 1')
plt.scatter(reduced_features[clusters == 1, 0], reduced_features[clusters == 1, 1], color='blue', marker='o', label='Cluster 2')
plt.scatter(reduced_features[clusters == 2, 0], reduced_features[clusters == 2, 1], color='orange', marker='o', label='Cluster 3')
plt.xlabel('Componente Principale 1')
plt.ylabel('Componente Principale 2')
plt.title('Visualizzazione dei Cluster')
plt.legend()
plt.show()
```

<div align="center">
<img src="https://github.com/iamluirio/unsupervised-learning-clustering/assets/118205581/497765b0-c62a-4259-b25d-192f92a84de1" />
</div>

- In the context of dimension reduction, the X-axis represents the first principal component (PC1) obtained from PCA. This component represents the direction along which the data varies the most.

- The Y-axis represents the second principal component (PC2) obtained from PCA. This component is orthogonal to the first and represents the second direction of maximum variation.

Let's look at the cluster labels assigned by DBSCAN and compare them with the colors in the graph to understand the situation of the red dots.

```python
# Confronto tra etichette dei cluster e colori nel grafico
plt.figure(figsize=(10, 8))

for i in range(n_clusters):
    cluster_points = reduced_features[clusters == i]
    
    if i == 0:
        cluster_color = 'green'
    elif i == 1:
        cluster_color = 'blue'
    elif i == 2:
        cluster_color = 'orange'
    else:
        cluster_color = colormaps.get_cmap('rainbow')(i / n_clusters)
    
    print('Cluster', i+1, ':')
    print('Number of observations in cluster:', len(cluster_points))
    print('Cluster color:', cluster_color)
    print('------------------------')
    
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=cluster_color, label=f'Cluster {i+1}')

# Punti noise (outliers)
noise_points = reduced_features[clusters == -1]
plt.scatter(noise_points[:, 0], noise_points[:, 1], color='red', marker='x', label='Noise (Outliers)')

plt.xlabel('Componente Principale 1')
plt.ylabel('Componente Principale 2')
plt.title('Visualizzazione dei Cluster e Noise')
plt.legend()
plt.colorbar()
plt.show()
```

```
Cluster 1 :
Number of observations in cluster: 20
Cluster color: green
------------------------
Cluster 2 :
Number of observations in cluster: 127
Cluster color: blue
------------------------
Cluster 3 :
Number of observations in cluster: 33
Cluster color: orange
------------------------
Cluster 4 :
Number of observations in cluster: 0
Cluster color: (0.503921568627451, 0.9999810273487268, 0.7049255469061472, 1.0)
------------------------
Cluster 5 :
Number of observations in cluster: 0
Cluster color: (0.8333333333333333, 0.8660254037844387, 0.5000000000000001, 1.0)
------------------------
Cluster 6 :
Number of observations in cluster: 0
Cluster color: (1.0, 0.4946558433997788, 0.2558427775944356, 1.0)
------------------------
```

<div align="center">
<img src="https://github.com/iamluirio/unsupervised-learning-clustering/assets/118205581/67752cb7-8a52-4de6-86e9-677af2e40183" />
</div>

The red dots that do not form a connected region are actually outliers.

### Explained Variance
Let's look at the variance explained by the first two principal components. This will give us an idea of ​​how well the two components capture the variation in the original data. These values ​​are important because they give you an idea of ​​how well the first two principal components summarize the information in the original data.

```python
explained_variance = pca.explained_variance_ratio_
print(f"Variance explained by principal component 1: {explained_variance[0]*100:.2f}%")
print(f"Variance explained by principal component 2: {explained_variance[1]*100:.2f}%")
```

```
Varianza spiegata dalla componente principale 1: 53.68%
Varianza spiegata dalla componente principale 2: 31.95%
```

### Davies-Bouldin Index
It measures the "feasibility" of clusters, considering both the average distance between cluster points and the distance between cluster centroids. The lower the value, the better the clusters.

```python
# Calcola la matrice delle distanze
distance_matrix = pairwise_distances(cosine_matrix, metric='euclidean')

# Calcola l'indice di Davies-Bouldin
davies_bouldin_index2 = davies_bouldin_score(cosine_matrix, clusters)
```
```
Davies-Bouldin Index: 1.651289505943405
```

A DBI value of 1.6 is generally considered a good value. DBI ranges from 0 to a theoretically infinite value. Lower values ​​indicate more compact and well-separated clusters, which is a desirable goal for a clustering algorithm.

### Scatter Plot with t-SNE
The data is projected into a two-dimensional space using [**t-SNE (t-distributed Stochastic Neighbor Embedding)**](https://it.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding). This technique is particularly useful for visualizing high-dimensional data so that the relationships between observations are preserved. Again, the points are colored according to the cluster label.

```python
# Riduzione delle dimensioni con t-SNE a 2 componenti principali
tsne = TSNE(n_components=2)
reduced_features_tsne = tsne.fit_transform(cosine_matrix)
```
```python
plt.figure(figsize=(8, 6))
for i in range(n_clusters2):
    cluster_points = reduced_features_tsne[clusters == i]
    
    if i == 0:
        cluster_color = 'green'
    elif i == 1:
        cluster_color = 'blue'
    elif i == 2:
        cluster_color = 'orange'
    else:
        cluster_color = 'red'
    
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=cluster_color, label=f'Cluster {i}')
    
noise_points_tsne = reduced_features_tsne[clusters == -1]
plt.scatter(noise_points_tsne[:, 0], noise_points_tsne[:, 1], color='red', marker='x', label='Noise (Outliers)')

plt.xlabel('Componente T-SNE 1')
plt.ylabel('Componente T-SNE 2')
plt.title('Scatter Plot con t-SNE')
plt.legend()
plt.show()
```

<div align="center">
  <img src="https://github.com/iamluirio/unsupervised-learning-clustering/assets/118205581/8ef384ea-01dc-4cef-87ad-9a1f8a722a35" />
</div>

```python
### Distance Heatmap
The distance matrix calculated using ```pairwise_distances``` is displayed as a heatmap. A heatmap is a graphical representation where different intensities of an attribute are represented by colors. In this case, the colors represent the distance between observations. This can help identify groups of observations that are closest to each other.

# Calcola la matrice delle distanze
distance_matrix = pairwise_distances(cosine_matrix, metric='euclidean')

# Crea una heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(distance_matrix, cmap='YlGnBu', xticklabels=False, yticklabels=False)
plt.title('Heatmap delle Distanze')
plt.xlabel('Osservazioni')
plt.ylabel('Osservazioni')
plt.show()
```

<div align="center">
  <img src="https://github.com/iamluirio/unsupervised-learning-clustering/assets/118205581/6ca76557-ad37-4e61-be9d-732762ff4fcd" />
</div>

