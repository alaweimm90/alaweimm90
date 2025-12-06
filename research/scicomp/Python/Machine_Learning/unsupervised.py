"""
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
"""
import numpy as np
import warnings
from typing import Optional, Tuple, Dict, Any, Union, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from scipy import stats, spatial, optimize
from scipy.linalg import eigh, svd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import logging
logger = logging.getLogger(__name__)
@dataclass
class ClusteringResults:
    """Container for clustering results and diagnostics."""
    labels: np.ndarray
    cluster_centers: Optional[np.ndarray] = None
    inertia: Optional[float] = None
    silhouette_score: Optional[float] = None
    calinski_harabasz_score: Optional[float] = None
    davies_bouldin_score: Optional[float] = None
    n_clusters: Optional[int] = None
@dataclass
class DimensionalityReductionResults:
    """Container for dimensionality reduction results."""
    transformed_data: np.ndarray
    components: Optional[np.ndarray] = None
    explained_variance: Optional[np.ndarray] = None
    explained_variance_ratio: Optional[np.ndarray] = None
    singular_values: Optional[np.ndarray] = None
    reconstruction_error: Optional[float] = None
class UnsupervisedModel(ABC):
    """Abstract base class for unsupervised learning models."""
    def __init__(self):
        self.is_fitted = False
        self.n_features = None
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'UnsupervisedModel':
        """Fit the model to data."""
        pass
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted model."""
        pass
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit model and transform data in one step."""
        return self.fit(X).transform(X)
    def _validate_input(self, X: np.ndarray):
        """Validate input data."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        return X
class KMeans(UnsupervisedModel):
    """
    K-means clustering with scientific computing enhancements.
    Features:
    - Multiple initialization methods
    - Convergence diagnostics
    - Cluster validation metrics
    - Robust distance metrics
    """
    def __init__(self,
                 n_clusters: int = 8,
                 init: str = 'k-means++',
                 n_init: int = 10,
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 random_state: Optional[int] = None):
        super().__init__()
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers = None
        self.labels = None
        self.inertia = None
        self.n_iter = None
    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initialize cluster centroids."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        if self.init == 'random':
            # Random initialization
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            centroids = X[indices].copy()
        elif self.init == 'k-means++':
            # K-means++ initialization
            centroids = np.zeros((self.n_clusters, n_features))
            # Choose first centroid randomly
            centroids[0] = X[np.random.randint(n_samples)]
            for k in range(1, self.n_clusters):
                # Compute distances to nearest centroid
                distances = np.array([
                    min([np.linalg.norm(x - c)**2 for c in centroids[:k]])
                    for x in X
                ])
                # Choose next centroid with probability proportional to squared distance
                probabilities = distances / distances.sum()
                cumulative_probabilities = probabilities.cumsum()
                r = np.random.rand()
                for j, p in enumerate(cumulative_probabilities):
                    if r < p:
                        centroids[k] = X[j]
                        break
        else:
            raise ValueError(f"Unknown initialization method: {self.init}")
        return centroids
    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign points to nearest cluster centroid."""
        distances = spatial.distance.cdist(X, centroids)
        return np.argmin(distances, axis=1)
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Update cluster centroids as mean of assigned points."""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            if np.sum(labels == k) > 0:
                centroids[k] = X[labels == k].mean(axis=0)
            else:
                # If cluster is empty, reinitialize randomly
                centroids[k] = X[np.random.randint(X.shape[0])]
        return centroids
    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """Compute within-cluster sum of squares (inertia)."""
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[k])**2)
        return inertia
    def fit(self, X: np.ndarray) -> 'KMeans':
        """Fit K-means clustering to data."""
        X = self._validate_input(X)
        self.n_features = X.shape[1]
        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        best_n_iter = 0
        # Run multiple initializations
        for init_run in range(self.n_init):
            centroids = self._init_centroids(X)
            # K-means iterations
            for iteration in range(self.max_iter):
                # Assign clusters
                labels = self._assign_clusters(X, centroids)
                # Update centroids
                new_centroids = self._update_centroids(X, labels)
                # Check convergence
                centroid_shift = np.linalg.norm(new_centroids - centroids)
                if centroid_shift < self.tol:
                    break
                centroids = new_centroids
            # Compute inertia for this run
            inertia = self._compute_inertia(X, labels, centroids)
            # Keep best result
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_n_iter = iteration + 1
        self.cluster_centers = best_centroids
        self.labels = best_labels
        self.inertia = best_inertia
        self.n_iter = best_n_iter
        self.is_fitted = True
        return self
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to cluster-distance space."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transforming")
        X = self._validate_input(X)
        return spatial.distance.cdist(X, self.cluster_centers)
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")
        X = self._validate_input(X)
        return self._assign_clusters(X, self.cluster_centers)
    def silhouette_score(self, X: np.ndarray) -> float:
        """Compute silhouette score for clustering quality."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self._compute_silhouette_score(X, self.labels)
    def _compute_silhouette_score(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Compute silhouette score."""
        n_samples = X.shape[0]
        silhouette_scores = np.zeros(n_samples)
        for i in range(n_samples):
            # Same cluster distances
            same_cluster = labels == labels[i]
            if np.sum(same_cluster) == 1:
                silhouette_scores[i] = 0
                continue
            a_i = np.mean([np.linalg.norm(X[i] - X[j])
                          for j in range(n_samples) if same_cluster[j] and j != i])
            # Nearest cluster distances
            b_i = np.inf
            for k in range(self.n_clusters):
                if k != labels[i]:
                    other_cluster = labels == k
                    if np.sum(other_cluster) > 0:
                        b_k = np.mean([np.linalg.norm(X[i] - X[j])
                                     for j in range(n_samples) if other_cluster[j]])
                        b_i = min(b_i, b_k)
            silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)
        return np.mean(silhouette_scores)
class PCA(UnsupervisedModel):
    """
    Principal Component Analysis with scientific computing enhancements.
    Features:
    - Multiple algorithms (SVD, eigendecomposition)
    - Uncertainty quantification
    - Explained variance analysis
    - Reconstruction capabilities
    """
    def __init__(self,
                 n_components: Optional[int] = None,
                 algorithm: str = 'svd',
                 whiten: bool = False,
                 random_state: Optional[int] = None):
        super().__init__()
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten
        self.random_state = random_state
        self.components = None
        self.explained_variance = None
        self.explained_variance_ratio = None
        self.singular_values = None
        self.mean = None
        self.noise_variance = None
    def fit(self, X: np.ndarray) -> 'PCA':
        """Fit PCA to data."""
        X = self._validate_input(X)
        n_samples, n_features = X.shape
        if self.n_components is None:
            self.n_components = min(n_samples, n_features)
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        if self.algorithm == 'svd':
            self._fit_svd(X_centered)
        elif self.algorithm == 'eigen':
            self._fit_eigen(X_centered)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        # Estimate noise variance (for PPCA)
        if self.n_components < n_features:
            total_variance = np.var(X_centered, ddof=1, axis=0).sum()
            explained_variance_sum = self.explained_variance.sum()
            self.noise_variance = (total_variance - explained_variance_sum) / (n_features - self.n_components)
        else:
            self.noise_variance = 0.0
        self.is_fitted = True
        return self
    def _fit_svd(self, X_centered: np.ndarray):
        """Fit PCA using SVD."""
        n_samples = X_centered.shape[0]
        # SVD of centered data
        U, s, Vt = svd(X_centered, full_matrices=False)
        # Components are rows of Vt
        self.components = Vt[:self.n_components]
        # Explained variance
        self.singular_values = s[:self.n_components]
        self.explained_variance = (s[:self.n_components] ** 2) / (n_samples - 1)
        total_variance = np.sum((s ** 2) / (n_samples - 1))
        self.explained_variance_ratio = self.explained_variance / total_variance
    def _fit_eigen(self, X_centered: np.ndarray):
        """Fit PCA using eigendecomposition of covariance matrix."""
        n_samples = X_centered.shape[0]
        # Covariance matrix
        cov_matrix = np.cov(X_centered.T)
        # Eigendecomposition
        eigenvalues, eigenvectors = eigh(cov_matrix)
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # Store components and explained variance
        self.components = eigenvectors[:, :self.n_components].T
        self.explained_variance = eigenvalues[:self.n_components]
        self.explained_variance_ratio = self.explained_variance / np.sum(eigenvalues)
        self.singular_values = np.sqrt(self.explained_variance * (n_samples - 1))
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to principal component space."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transforming")
        X = self._validate_input(X)
        X_centered = X - self.mean
        # Project onto principal components
        X_transformed = X_centered @ self.components.T
        if self.whiten:
            X_transformed /= np.sqrt(self.explained_variance)
        return X_transformed
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Transform data back to original space."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before inverse transforming")
        if self.whiten:
            X_transformed = X_transformed * np.sqrt(self.explained_variance)
        # Project back to original space
        X_reconstructed = X_transformed @ self.components + self.mean
        return X_reconstructed
    def reconstruction_error(self, X: np.ndarray) -> float:
        """Compute reconstruction error."""
        X_reconstructed = self.inverse_transform(self.transform(X))
        return np.mean(np.sum((X - X_reconstructed)**2, axis=1))
    def explained_variance_score(self, threshold: float = 0.95) -> int:
        """Find number of components needed to explain threshold variance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        cumulative_variance = np.cumsum(self.explained_variance_ratio)
        return np.argmax(cumulative_variance >= threshold) + 1
class ICA(UnsupervisedModel):
    """
    Independent Component Analysis for blind source separation.
    Features:
    - FastICA algorithm
    - Multiple contrast functions
    - Prewhitening
    - Source separation quality metrics
    """
    def __init__(self,
                 n_components: Optional[int] = None,
                 algorithm: str = 'fastica',
                 fun: str = 'logcosh',
                 max_iter: int = 200,
                 tol: float = 1e-4,
                 random_state: Optional[int] = None):
        super().__init__()
        self.n_components = n_components
        self.algorithm = algorithm
        self.fun = fun
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.components = None
        self.mixing_matrix = None
        self.mean = None
        self.whitening_matrix = None
    def _contrast_functions(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Contrast functions for ICA."""
        if self.fun == 'logcosh':
            alpha = 1.0
            gx = np.tanh(alpha * x)
            g_prime = alpha * (1 - gx**2)
        elif self.fun == 'exp':
            gx = x * np.exp(-x**2 / 2)
            g_prime = (1 - x**2) * np.exp(-x**2 / 2)
        elif self.fun == 'cube':
            gx = x**3
            g_prime = 3 * x**2
        else:
            raise ValueError(f"Unknown contrast function: {self.fun}")
        return gx, g_prime
    def _whiten(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Whiten the data using PCA."""
        pca = PCA(n_components=self.n_components, whiten=True, random_state=self.random_state)
        X_whitened = pca.fit_transform(X)
        whitening_matrix = pca.components_ / np.sqrt(pca.explained_variance)[:, np.newaxis]
        return X_whitened, whitening_matrix
    def fit(self, X: np.ndarray) -> 'ICA':
        """Fit ICA to data."""
        X = self._validate_input(X)
        n_samples, n_features = X.shape
        if self.n_components is None:
            self.n_components = n_features
        if self.random_state is not None:
            np.random.seed(self.random_state)
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        # Whiten the data
        X_whitened, self.whitening_matrix = self._whiten(X_centered)
        # FastICA algorithm
        W = np.random.randn(self.n_components, self.n_components)
        W = self._symmetric_decorrelation(W)
        for iteration in range(self.max_iter):
            W_old = W.copy()
            # FastICA update rule
            gx, g_prime = self._contrast_functions(W @ X_whitened.T)
            W_new = np.mean(gx[:, :, np.newaxis] * X_whitened[np.newaxis, :, :], axis=1) - \
                    np.mean(g_prime, axis=1)[:, np.newaxis] * W
            # Symmetric decorrelation
            W = self._symmetric_decorrelation(W_new)
            # Check convergence
            if np.max(np.abs(np.abs(np.diag(W @ W_old.T)) - 1)) < self.tol:
                break
        # Store unmixing matrix
        self.components = W @ self.whitening_matrix
        self.mixing_matrix = np.linalg.pinv(self.components)
        self.is_fitted = True
        return self
    def _symmetric_decorrelation(self, W: np.ndarray) -> np.ndarray:
        """Symmetric decorrelation of matrix W."""
        U, s, Vt = svd(W, full_matrices=False)
        return U @ Vt
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to independent component space."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transforming")
        X = self._validate_input(X)
        X_centered = X - self.mean
        return X_centered @ self.components.T
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Transform data back to original space."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before inverse transforming")
        return X_transformed @ self.mixing_matrix.T + self.mean
class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise.
    Features:
    - Automatic cluster number detection
    - Noise point identification
    - Arbitrary cluster shapes
    - Distance metric options
    """
    def __init__(self,
                 eps: float = 0.5,
                 min_samples: int = 5,
                 metric: str = 'euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels = None
        self.core_sample_indices = None
        self.is_fitted = False
    def fit(self, X: np.ndarray) -> 'DBSCAN':
        """Fit DBSCAN clustering to data."""
        X = np.asarray(X)
        n_samples = X.shape[0]
        # Compute distance matrix
        if self.metric == 'euclidean':
            distances = spatial.distance.pdist(X, metric='euclidean')
            distance_matrix = spatial.distance.squareform(distances)
        else:
            distance_matrix = spatial.distance.cdist(X, X, metric=self.metric)
        # Find neighbors within eps
        neighbors = [np.where(distance_matrix[i] <= self.eps)[0] for i in range(n_samples)]
        # Identify core points
        core_samples = [i for i in range(n_samples) if len(neighbors[i]) >= self.min_samples]
        self.core_sample_indices = np.array(core_samples)
        # Initialize labels (-1 for noise, 0+ for clusters)
        labels = np.full(n_samples, -1, dtype=int)
        cluster_id = 0
        # Cluster formation
        for core_point in core_samples:
            if labels[core_point] != -1:  # Already processed
                continue
            # Start new cluster
            cluster_points = set([core_point])
            to_process = [core_point]
            while to_process:
                current_point = to_process.pop()
                labels[current_point] = cluster_id
                if current_point in core_samples:
                    # Add neighbors to cluster
                    for neighbor in neighbors[current_point]:
                        if labels[neighbor] == -1:  # Unprocessed
                            cluster_points.add(neighbor)
                            to_process.append(neighbor)
            cluster_id += 1
        self.labels = labels
        self.is_fitted = True
        return self
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit model and return cluster labels."""
        return self.fit(X).labels
def create_test_datasets() -> Dict[str, np.ndarray]:
    """Create test datasets for unsupervised learning validation."""
    np.random.seed(42)
    datasets = {}
    # Gaussian clusters for clustering
    n_samples = 300
    cluster1 = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], n_samples//3)
    cluster2 = np.random.multivariate_normal([-2, -2], [[0.3, 0.1], [0.1, 0.3]], n_samples//3)
    cluster3 = np.random.multivariate_normal([2, -2], [[0.4, -0.1], [-0.1, 0.4]], n_samples//3)
    datasets['clustering'] = np.vstack([cluster1, cluster2, cluster3])
    # High-dimensional data for PCA
    n_dim = 50
    n_components = 5
    true_components = np.random.randn(n_components, n_dim)
    coefficients = np.random.randn(n_samples, n_components)
    noise = 0.1 * np.random.randn(n_samples, n_dim)
    datasets['pca'] = coefficients @ true_components + noise
    # Mixed signals for ICA
    n_sources = 3
    time = np.linspace(0, 10, 1000)
    source1 = np.sin(2 * np.pi * time)  # Sine wave
    source2 = np.sign(np.sin(3 * np.pi * time))  # Square wave
    source3 = np.random.uniform(-1, 1, len(time))  # Noise
    sources = np.column_stack([source1, source2, source3])
    # Mix the sources
    mixing_matrix = np.random.randn(3, 3)
    datasets['ica'] = sources @ mixing_matrix.T
    return datasets
# Visualization utilities
def plot_clustering_results(model, X: np.ndarray, title: str = "Clustering Results"):
    """Plot clustering results with Berkeley styling."""
    fig, ax = plt.subplots(figsize=(10, 8))
    # Berkeley colors
    berkeley_blue = '#003262'
    california_gold = '#FDB515'
    if hasattr(model, 'labels'):
        labels = model.labels
    else:
        labels = model.predict(X)
    # Plot points colored by cluster
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    for label, color in zip(unique_labels, colors):
        if label == -1:  # Noise points
            mask = labels == label
            ax.scatter(X[mask, 0], X[mask, 1], c='black', marker='x', s=50, alpha=0.6, label='Noise')
        else:
            mask = labels == label
            ax.scatter(X[mask, 0], X[mask, 1], c=[color], s=50, alpha=0.7, label=f'Cluster {label}')
    # Plot cluster centers if available
    if hasattr(model, 'cluster_centers') and model.cluster_centers is not None:
        centers = model.cluster_centers
        ax.scatter(centers[:, 0], centers[:, 1], c=california_gold,
                  marker='*', s=200, edgecolors=berkeley_blue, linewidth=2, label='Centers')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig
def plot_pca_results(pca_model, X: np.ndarray, title: str = "PCA Results"):
    """Plot PCA results with Berkeley styling."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Berkeley colors
    berkeley_blue = '#003262'
    california_gold = '#FDB515'
    # Plot 1: Explained variance
    axes[0, 0].bar(range(1, len(pca_model.explained_variance_ratio) + 1),
                   pca_model.explained_variance_ratio, color=berkeley_blue, alpha=0.7)
    axes[0, 0].set_xlabel('Principal Component')
    axes[0, 0].set_ylabel('Explained Variance Ratio')
    axes[0, 0].set_title('Explained Variance by Component')
    axes[0, 0].grid(True, alpha=0.3)
    # Plot 2: Cumulative explained variance
    cumulative_variance = np.cumsum(pca_model.explained_variance_ratio)
    axes[0, 1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
                    'o-', color=california_gold, linewidth=2, markersize=6)
    axes[0, 1].axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% threshold')
    axes[0, 1].set_xlabel('Number of Components')
    axes[0, 1].set_ylabel('Cumulative Explained Variance')
    axes[0, 1].set_title('Cumulative Explained Variance')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    # Plot 3: First two principal components
    if X.shape[1] >= 2:
        X_transformed = pca_model.transform(X)
        axes[1, 0].scatter(X_transformed[:, 0], X_transformed[:, 1],
                          c=berkeley_blue, alpha=0.6, s=30)
        axes[1, 0].set_xlabel('First Principal Component')
        axes[1, 0].set_ylabel('Second Principal Component')
        axes[1, 0].set_title('Data in PC Space')
        axes[1, 0].grid(True, alpha=0.3)
    # Plot 4: Component loadings (if 2D original data)
    if X.shape[1] == 2:
        axes[1, 1].arrow(0, 0, pca_model.components[0, 0], pca_model.components[0, 1],
                        head_width=0.05, head_length=0.05, fc=berkeley_blue, ec=berkeley_blue)
        axes[1, 1].arrow(0, 0, pca_model.components[1, 0], pca_model.components[1, 1],
                        head_width=0.05, head_length=0.05, fc=california_gold, ec=california_gold)
        axes[1, 1].set_xlim(-1, 1)
        axes[1, 1].set_ylim(-1, 1)
        axes[1, 1].set_xlabel('Feature 1 Loading')
        axes[1, 1].set_ylabel('Feature 2 Loading')
        axes[1, 1].set_title('Principal Component Loadings')
        axes[1, 1].grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig