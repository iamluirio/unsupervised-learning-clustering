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

