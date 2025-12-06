# unsupervised
**Module:** `Python/Machine_Learning/unsupervised.py`
## Overview
Unsupervised Learning Algorithms for Scientific Computing
This module implements unsupervised learning algorithms specifically designed for
scientific computing applications, including clustering, dimensionality reduction,
and density estimation methods.
Classes:
KMeans: K-means clustering with scientific extensions
HierarchicalClustering: Agglomerative and divisive clustering
DBSCAN: Density-based clustering for arbitrary shaped clusters
PCA: Principal Component Analysis with uncertainty quantification
ICA: Independent Component Analysis for signal separation
tSNE: t-distributed Stochastic Neighbor Embedding
UMAP: Uniform Manifold Approximation and Projection
GaussianMixture: Gaussian Mixture Models for density estimation
## Functions
### `create_test_datasets()`
Create test datasets for unsupervised learning validation.
**Source:** [Line 614](Python/Machine_Learning/unsupervised.py#L614)
### `plot_clustering_results(model, X, title)`
Plot clustering results with Berkeley styling.
**Source:** [Line 650](Python/Machine_Learning/unsupervised.py#L650)
### `plot_pca_results(pca_model, X, title)`
Plot PCA results with Berkeley styling.
**Source:** [Line 690](Python/Machine_Learning/unsupervised.py#L690)
## Classes
### `ClusteringResults`
Container for clustering results and diagnostics.
**Class Source:** [Line 34](Python/Machine_Learning/unsupervised.py#L34)
### `DimensionalityReductionResults`
Container for dimensionality reduction results.
**Class Source:** [Line 46](Python/Machine_Learning/unsupervised.py#L46)
### `UnsupervisedModel`
Abstract base class for unsupervised learning models.
#### Methods
##### `__init__(self)`
*No documentation available.*
**Source:** [Line 59](Python/Machine_Learning/unsupervised.py#L59)
##### `fit(self, X)`
Fit the model to data.
**Source:** [Line 64](Python/Machine_Learning/unsupervised.py#L64)
##### `transform(self, X)`
Transform data using fitted model.
**Source:** [Line 69](Python/Machine_Learning/unsupervised.py#L69)
##### `fit_transform(self, X)`
Fit model and transform data in one step.
**Source:** [Line 73](Python/Machine_Learning/unsupervised.py#L73)
##### `_validate_input(self, X)`
Validate input data.
**Source:** [Line 77](Python/Machine_Learning/unsupervised.py#L77)
**Class Source:** [Line 56](Python/Machine_Learning/unsupervised.py#L56)
### `KMeans`
K-means clustering with scientific computing enhancements.
Features:
- Multiple initialization methods
- Convergence diagnostics
- Cluster validation metrics
- Robust distance metrics
#### Methods
##### `__init__(self, n_clusters, init, n_init, max_iter, tol, random_state)`
*No documentation available.*
**Source:** [Line 96](Python/Machine_Learning/unsupervised.py#L96)
##### `_init_centroids(self, X)`
Initialize cluster centroids.
**Source:** [Line 115](Python/Machine_Learning/unsupervised.py#L115)
##### `_assign_clusters(self, X, centroids)`
Assign points to nearest cluster centroid.
**Source:** [Line 155](Python/Machine_Learning/unsupervised.py#L155)
##### `_update_centroids(self, X, labels)`
Update cluster centroids as mean of assigned points.
**Source:** [Line 160](Python/Machine_Learning/unsupervised.py#L160)
##### `_compute_inertia(self, X, labels, centroids)`
Compute within-cluster sum of squares (inertia).
**Source:** [Line 171](Python/Machine_Learning/unsupervised.py#L171)
##### `fit(self, X)`
Fit K-means clustering to data.
**Source:** [Line 180](Python/Machine_Learning/unsupervised.py#L180)
##### `transform(self, X)`
Transform data to cluster-distance space.
**Source:** [Line 227](Python/Machine_Learning/unsupervised.py#L227)
##### `predict(self, X)`
Predict cluster labels for new data.
**Source:** [Line 235](Python/Machine_Learning/unsupervised.py#L235)
##### `silhouette_score(self, X)`
Compute silhouette score for clustering quality.
**Source:** [Line 243](Python/Machine_Learning/unsupervised.py#L243)
##### `_compute_silhouette_score(self, X, labels)`
Compute silhouette score.
**Source:** [Line 250](Python/Machine_Learning/unsupervised.py#L250)
**Class Source:** [Line 85](Python/Machine_Learning/unsupervised.py#L85)
### `PCA`
Principal Component Analysis with scientific computing enhancements.
Features:
- Multiple algorithms (SVD, eigendecomposition)
- Uncertainty quantification
- Explained variance analysis
- Reconstruction capabilities
#### Methods
##### `__init__(self, n_components, algorithm, whiten, random_state)`
*No documentation available.*
**Source:** [Line 291](Python/Machine_Learning/unsupervised.py#L291)
##### `fit(self, X)`
Fit PCA to data.
**Source:** [Line 308](Python/Machine_Learning/unsupervised.py#L308)
##### `_fit_svd(self, X_centered)`
Fit PCA using SVD.
**Source:** [Line 338](Python/Machine_Learning/unsupervised.py#L338)
##### `_fit_eigen(self, X_centered)`
Fit PCA using eigendecomposition of covariance matrix.
**Source:** [Line 354](Python/Machine_Learning/unsupervised.py#L354)
##### `transform(self, X)`
Transform data to principal component space.
**Source:** [Line 375](Python/Machine_Learning/unsupervised.py#L375)
##### `inverse_transform(self, X_transformed)`
Transform data back to original space.
**Source:** [Line 391](Python/Machine_Learning/unsupervised.py#L391)
##### `reconstruction_error(self, X)`
Compute reconstruction error.
**Source:** [Line 403](Python/Machine_Learning/unsupervised.py#L403)
##### `explained_variance_score(self, threshold)`
Find number of components needed to explain threshold variance.
**Source:** [Line 408](Python/Machine_Learning/unsupervised.py#L408)
**Class Source:** [Line 280](Python/Machine_Learning/unsupervised.py#L280)
### `ICA`
Independent Component Analysis for blind source separation.
Features:
- FastICA algorithm
- Multiple contrast functions
- Prewhitening
- Source separation quality metrics
#### Methods
##### `__init__(self, n_components, algorithm, fun, max_iter, tol, random_state)`
*No documentation available.*
**Source:** [Line 428](Python/Machine_Learning/unsupervised.py#L428)
##### `_contrast_functions(self, x)`
Contrast functions for ICA.
**Source:** [Line 447](Python/Machine_Learning/unsupervised.py#L447)
##### `_whiten(self, X)`
Whiten the data using PCA.
**Source:** [Line 464](Python/Machine_Learning/unsupervised.py#L464)
##### `fit(self, X)`
Fit ICA to data.
**Source:** [Line 471](Python/Machine_Learning/unsupervised.py#L471)
##### `_symmetric_decorrelation(self, W)`
Symmetric decorrelation of matrix W.
**Source:** [Line 516](Python/Machine_Learning/unsupervised.py#L516)
##### `transform(self, X)`
Transform data to independent component space.
**Source:** [Line 521](Python/Machine_Learning/unsupervised.py#L521)
##### `inverse_transform(self, X_transformed)`
Transform data back to original space.
**Source:** [Line 530](Python/Machine_Learning/unsupervised.py#L530)
**Class Source:** [Line 417](Python/Machine_Learning/unsupervised.py#L417)
### `DBSCAN`
Density-Based Spatial Clustering of Applications with Noise.
Features:
- Automatic cluster number detection
- Noise point identification
- Arbitrary cluster shapes
- Distance metric options
#### Methods
##### `__init__(self, eps, min_samples, metric)`
*No documentation available.*
**Source:** [Line 549](Python/Machine_Learning/unsupervised.py#L549)
##### `fit(self, X)`
Fit DBSCAN clustering to data.
**Source:** [Line 560](Python/Machine_Learning/unsupervised.py#L560)
##### `fit_predict(self, X)`
Fit model and return cluster labels.
**Source:** [Line 609](Python/Machine_Learning/unsupervised.py#L609)
**Class Source:** [Line 538](Python/Machine_Learning/unsupervised.py#L538)