# Unsupervised Learning Data Analysis: Clustering


**Clustering** is a type of unsupervised learning in machine learning and data analysis that involves grouping a set of objects in such a way that objects in the same group (called a **cluster**) are more similar to each other than to those in other groups (clusters). It is a common technique for statistical data analysis used in many fields, including machine learning, pattern recognition, image analysis, information retrieval, and bioinformatics.

The project contains the [**DBSCAN Clustering algorithm**](https://it.wikipedia.org/wiki/Dbscan): it exploits the [**Euclidean distance**](https://en.wikipedia.org/wiki/Euclidean_distance) as a default metric to group connected points for density. Euclidean distance is a common measure used to calculate the distance between two points in the Euclidean space.

In addition to the traditional approach, I used a different metric for the grouping of the elements in the formats trained: we want to change the algorithm to obtain a clustering based not on the similarity among the features of feature, but **on the similarity of each sample with the other samples**. To obtain this goal, I implemented the DBSCAN algorithm using the [**cosine similarity**](https://en.wikipedia.org/wiki/Cosine_similarity): it's a measure that evaluates the corner between two vectors, indicating how similar they are in the direction. In the context of clustering or data analysis, the similarity of the cosine can be used to calculate how similar two samples are based on their characteristics.

We then appear the different models, to observe the results.


## Datasets
| CUST_ID | BALANCE     | BALANCE_FREQUENCY | PURCHASES | ONEOFF_PURCHASES | INSTALLMENTS_PURCHASES | CASH_ADVANCE  | PURCHASES_FREQUENCY | ONEOFF_PURCHASES_FREQUENCY | PURCHASES_INSTALLMENTS_FREQUENCY | CASH_ADVANCE_FREQUENCY | CASH_ADVANCE_TRX | PURCHASES_TRX | CREDIT_LIMIT | PAYMENTS     | MINIMUM_PAYMENTS | PRC_FULL_PAYMENT | TENURE |
|---------|-------------|-------------------|-----------|------------------|------------------------|---------------|----------------------|----------------------------|-------------------------------|------------------------|-----------------|----------------|---------------|-------------|-----------------|------------------|--------|
| C10001  | 40.90       | 0.818             | 95.40     | 0.00             | 95.40                  | 0.00          | 0.167                | 0.00                       | 0.083                        | 0.00                   | 0               | 2             | 1000          | 201.80       | 139.51           | 0.00             | 12     |
| C10002  | 3202.47     | 0.909             | 0.00      | 0.00             | 0.00                   | 6442.95       | 0.00                 | 0.00                       | 0.00                         | 0.25                   | 4               | 0             | 7000          | 4103.03      | 1072.34          | 0.222            | 12     |
| C10003  | 2495.15     | 1.000             | 773.17    | 773.17           | 0.00                   | 0.00          | 1.00                 | 1.00                       | 0.00                         | 0.00                   | 0               | 12            | 7500          | 622.07       | 627.28           | 0.00             | 12     |
| C10004  | 1666.67     | 0.636             | 1499.00   | 1499.00          | 0.00                   | 205.79        | 0.083                | 0.083                      | 0.00                         | 0.083                  | 1               | 1             | 7500          | 0.00         |                 | 0.00             | 12     |
| C10005  | 817.71      | 1.000             | 16.00     | 16.00            | 0.00                   | 0.00          | 0.083                | 0.083                      | 0.00                         | 0.00                   | 0               | 1             | 1200          | 678.33       | 244.79           | 0.00             | 12     |
| C10006  | 1809.83     | 1.000             | 1333.28   | 0.00             | 1333.28                | 0.00          | 0.667                | 0.00                       | 0.583                        | 0.00                   | 0               | 8             | 1800          | 1400.06      | 2407.25          | 0.00             | 12     |


