# Unsupervised Learning Data Analysis: Clustering
**Clustering** is a type of unsupervised learning in machine learning and data analysis that involves grouping a set of objects in such a way that objects in the same group (called a **cluster**) are more similar to each other than to those in other groups (clusters). It is a common technique for statistical data analysis used in many fields, including machine learning, pattern recognition, image analysis, information retrieval, and bioinformatics.

The project contains two types of clustering approaches: **K-means** and **DBSCAN**, respectively.

## Datasets
| CUST_ID | BALANCE     | BALANCE_FREQUENCY | PURCHASES | ONEOFF_PURCHASES | INSTALLMENTS_PURCHASES | CASH_ADVANCE  | PURCHASES_FREQUENCY | ONEOFF_PURCHASES_FREQUENCY | PURCHASES_INSTALLMENTS_FREQUENCY | CASH_ADVANCE_FREQUENCY | CASH_ADVANCE_TRX | PURCHASES_TRX | CREDIT_LIMIT | PAYMENTS     | MINIMUM_PAYMENTS | PRC_FULL_PAYMENT | TENURE |
|---------|-------------|-------------------|-----------|------------------|------------------------|---------------|----------------------|----------------------------|-------------------------------|------------------------|-----------------|----------------|---------------|-------------|-----------------|------------------|--------|
| C10001  | 40.90       | 0.818             | 95.40     | 0.00             | 95.40                  | 0.00          | 0.167                | 0.00                       | 0.083                        | 0.00                   | 0               | 2             | 1000          | 201.80       | 139.51           | 0.00             | 12     |
| C10002  | 3202.47     | 0.909             | 0.00      | 0.00             | 0.00                   | 6442.95       | 0.00                 | 0.00                       | 0.00                         | 0.25                   | 4               | 0             | 7000          | 4103.03      | 1072.34          | 0.222            | 12     |
| C10003  | 2495.15     | 1.000             | 773.17    | 773.17           | 0.00                   | 0.00          | 1.00                 | 1.00                       | 0.00                         | 0.00                   | 0               | 12            | 7500          | 622.07       | 627.28           | 0.00             | 12     |
| C10004  | 1666.67     | 0.636             | 1499.00   | 1499.00          | 0.00                   | 205.79        | 0.083                | 0.083                      | 0.00                         | 0.083                  | 1               | 1             | 7500          | 0.00         |                 | 0.00             | 12     |
| C10005  | 817.71      | 1.000             | 16.00     | 16.00            | 0.00                   | 0.00          | 0.083                | 0.083                      | 0.00                         | 0.00                   | 0               | 1             | 1200          | 678.33       | 244.79           | 0.00             | 12     |
| C10006  | 1809.83     | 1.000             | 1333.28   | 0.00             | 1333.28                | 0.00          | 0.667                | 0.00                       | 0.583                        | 0.00                   | 0               | 8             | 1800          | 1400.06      | 2407.25          | 0.00             | 12     |



## K-means Clustering
K-means is a popular clustering algorithm that **partitions the data into k clusters**, where k is a user-defined parameter. The initialisation starts with choosing **k initial centroids** randomly from the data points. It assigns each data point to the nearest centroid, forming k clusters, and recalculates the centroids of the k clusters by taking the mean of all data points in each cluster. It repeats the assignment and update steps until the centroids no longer change or change very little (i.e., the algorithm converges).

The objective of K-means is to minimize the **within-cluster sum of squares (WCSS)**, which measures the variance within each cluster.


