# 🌀 Density-Based Spatial Clustering (DBSCAN) on Non-Linear Geometries

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A clean, hands-on implementation demonstrating the power of **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** on complex, non-linearly separable datasets using Python and Scikit-Learn[cite: 1].

---

## 📌 Executive Summary
Traditional distance-based clustering algorithms (such as **K-Means**) partition feature space using linear boundaries and assume spherical, evenly-sized clusters. When applied to nested structures—like concentric circles—K-Means fundamentally fails. 

This project explores how **DBSCAN** bypasses centroid constraints by grouping points based on **local spatial density**, successfully isolating complex geometric boundaries and filtering out noise.

---

## 🔬 Algorithmic Deep Dive: Why DBSCAN?

Unlike partition-based methods, DBSCAN relies on two core hyperparameters:
1. **$\epsilon$ (epsilon):** The maximum distance between two samples for one to be considered as in the neighborhood of the other.
2. **`min_samples`:** The number of samples in a neighborhood for a point to be considered as a core point (density threshold).

### The Geometric Challenge & Solution
* **The Failure Mode of K-Means:** Tries to draw straight-line boundaries through space, splitting the inner and outer circles incorrectly.
* **The DBSCAN Advantage:** Evaluates the density connectivity of points. Because the points within each circle are densely packed relative to the space between them, DBSCAN traces the continuous curve of each ring independently regardless of its shape.

---

## 🛠️ Tech Stack
* **Language:** Python[cite: 1]
* **Core Libraries:** 
  * `scikit-learn` (Dataset generation via `make_circles`, clustering via `DBSCAN`)[cite: 1]
  * `numpy` (Numerical vector operations)[cite: 1]
  * `matplotlib` (Data visualization and plotting)[cite: 1]

---

## 💻 Implementation Code

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN
import numpy as np

# Reproducibility seed & dataset generation (concentric circles)
X, _ = make_circles(n_samples=500, factor=0.5, noise=0.03, random_state=4)

# Initialize and fit DBSCAN with optimized density parameters
dbscan = DBSCAN(eps=0.1, min_samples=5)
clusters = dbscan.fit_predict(X)

# High-fidelity plotting setup
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', edgecolor='k', s=40, marker='o')
plt.title("DBSCAN Clustering of Concentric Circles (eps=0.1, min_samples=5)", fontsize=12, fontweight='bold')
plt.xlabel("Feature 0", fontsize=10)
plt.ylabel("Feature 1", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
