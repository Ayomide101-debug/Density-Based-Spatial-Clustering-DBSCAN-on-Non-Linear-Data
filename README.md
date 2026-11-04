# Density-Based Spatial Clustering (DBSCAN) on Non-Linear Data

A practical machine learning experiment demonstrating the effectiveness of **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** on complex, non-linearly separable datasets (concentric circles) using Python and Scikit-Learn.

---

## The Problem & Objective
Traditional clustering algorithms like **K-Means** rely on centroids and assume spherical, evenly-sized clusters. When faced with complex geometries—such as one cluster wrapped entirely inside another (concentric circles)—K-Means fails because it partitions space based on linear boundaries. 

The objective of this project is to implement **DBSCAN** to correctly identify and separate these nested geometric structures based on local density rather than distance to a center point[cite: 1].

---

## Tech Stack & Libraries
* **Python**[cite: 1]
* **Scikit-Learn** (for dataset generation via `make_circles` and the `DBSCAN` implementation)[cite: 1]
* **NumPy** (for array operations)[cite: 1]
* **Matplotlib** (for data visualization)[cite: 1]

---

## Code Implementation

import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN
import numpy as np

# Create a concentric circle dataset
X, _ = make_circles(n_samples=500, factor=.5, noise=.03, random_state=4)

# Apply DBSCAN to the dataset
dbscan = DBSCAN(eps=0.1, min_samples=5)
clusters = dbscan.fit_predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', marker='o')
plt.title("DBSCAN Clustering of Concentric Circles")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
