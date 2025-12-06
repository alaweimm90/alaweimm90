"""
Advanced aggregation strategies for federated learning.

Implements FedAvg, Byzantine-robust aggregation, personalized FL,
and adaptive aggregation strategies.
"""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    """Result of aggregation operation."""
    global_weights: Dict[str, np.ndarray]
    client_weights: Optional[Dict[str, Dict[str, np.ndarray]]] = None
    metadata: Dict[str, Any] = None
    excluded_clients: List[str] = None


class BaseAggregator(ABC):
    """Base class for aggregation strategies."""

    @abstractmethod
    def aggregate(self, updates: List[Dict[str, Any]]) -> AggregationResult:
        """Aggregate client updates."""
        pass

    @abstractmethod
    def validate_updates(self, updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and filter updates."""
        pass


class FedAvgAggregator(BaseAggregator):
    """Federated Averaging aggregator."""

    def __init__(self, weighted: bool = True):
        """Initialize FedAvg aggregator.

        Args:
            weighted: Whether to weight by number of samples
        """
        self.weighted = weighted

    def aggregate(self, updates: List[Dict[str, Any]]) -> AggregationResult:
        """Perform FedAvg aggregation.

        Args:
            updates: List of client updates

        Returns:
            Aggregation result
        """
        valid_updates = self.validate_updates(updates)

        if not valid_updates:
            raise ValueError("No valid updates to aggregate")

        if self.weighted:
            return self._weighted_average(valid_updates)
        else:
            return self._simple_average(valid_updates)

    def validate_updates(self, updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate updates have required fields."""
        valid = []
        for update in updates:
            if 'weights' in update and 'num_samples' in update:
                valid.append(update)
            else:
                logger.warning(f"Invalid update from client {update.get('client_id')}")
        return valid

    def _weighted_average(self, updates: List[Dict[str, Any]]) -> AggregationResult:
        """Compute weighted average of updates."""
        total_samples = sum(u['num_samples'] for u in updates)
        aggregated = {}

        # Get all layer names
        layer_names = set()
        for update in updates:
            layer_names.update(update['weights'].keys())

        for layer in layer_names:
            weighted_sum = None
            for update in updates:
                if layer in update['weights']:
                    weight = update['num_samples'] / total_samples
                    if weighted_sum is None:
                        weighted_sum = update['weights'][layer] * weight
                    else:
                        weighted_sum += update['weights'][layer] * weight

            if weighted_sum is not None:
                aggregated[layer] = weighted_sum

        return AggregationResult(
            global_weights=aggregated,
            metadata={'total_samples': total_samples, 'num_clients': len(updates)}
        )

    def _simple_average(self, updates: List[Dict[str, Any]]) -> AggregationResult:
        """Compute simple average of updates."""
        aggregated = {}

        # Get all layer names
        layer_names = set()
        for update in updates:
            layer_names.update(update['weights'].keys())

        for layer in layer_names:
            layer_updates = []
            for update in updates:
                if layer in update['weights']:
                    layer_updates.append(update['weights'][layer])

            if layer_updates:
                aggregated[layer] = np.mean(layer_updates, axis=0)

        return AggregationResult(
            global_weights=aggregated,
            metadata={'num_clients': len(updates)}
        )


class ByzantineRobustAggregator(BaseAggregator):
    """Byzantine-robust aggregation strategies."""

    def __init__(self, method: str = 'krum', byzantine_fraction: float = 0.2):
        """Initialize Byzantine-robust aggregator.

        Args:
            method: Aggregation method ('krum', 'median', 'trimmed_mean', 'bulyan')
            byzantine_fraction: Expected fraction of Byzantine clients
        """
        self.method = method
        self.byzantine_fraction = byzantine_fraction
        self.aggregation_methods = {
            'krum': self._krum_aggregation,
            'median': self._median_aggregation,
            'trimmed_mean': self._trimmed_mean_aggregation,
            'bulyan': self._bulyan_aggregation
        }

    def aggregate(self, updates: List[Dict[str, Any]]) -> AggregationResult:
        """Perform Byzantine-robust aggregation.

        Args:
            updates: List of client updates

        Returns:
            Aggregation result
        """
        valid_updates = self.validate_updates(updates)

        if self.method not in self.aggregation_methods:
            raise ValueError(f"Unknown aggregation method: {self.method}")

        return self.aggregation_methods[self.method](valid_updates)

    def validate_updates(self, updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and detect Byzantine updates."""
        # Basic validation
        valid = []
        for update in updates:
            if self._is_valid_update(update):
                valid.append(update)

        # Statistical outlier detection
        if len(valid) > 3:
            valid = self._filter_statistical_outliers(valid)

        return valid

    def _is_valid_update(self, update: Dict[str, Any]) -> bool:
        """Check if update is valid."""
        if 'weights' not in update:
            return False

        # Check for NaN or Inf values
        for layer, weights in update['weights'].items():
            if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                return False

        return True

    def _filter_statistical_outliers(self, updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter statistical outliers using MAD."""
        # Calculate distances between updates
        distances = self._calculate_pairwise_distances(updates)

        # Use Median Absolute Deviation
        median_dist = np.median(distances)
        mad = np.median(np.abs(distances - median_dist))

        # Filter outliers
        threshold = median_dist + 3 * mad
        filtered = []
        for i, update in enumerate(updates):
            if distances[i] < threshold:
                filtered.append(update)

        return filtered

    def _calculate_pairwise_distances(self, updates: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate pairwise distances between updates."""
        n = len(updates)
        distances = np.zeros(n)

        for i in range(n):
            dist_sum = 0
            count = 0
            for j in range(n):
                if i != j:
                    dist = self._calculate_distance(
                        updates[i]['weights'],
                        updates[j]['weights']
                    )
                    dist_sum += dist
                    count += 1

            distances[i] = dist_sum / count if count > 0 else 0

        return distances

    def _calculate_distance(self, weights1: Dict[str, np.ndarray],
                          weights2: Dict[str, np.ndarray]) -> float:
        """Calculate distance between two weight updates."""
        distance = 0
        count = 0

        for layer in weights1.keys():
            if layer in weights2:
                dist = np.linalg.norm(weights1[layer] - weights2[layer])
                distance += dist
                count += 1

        return distance / count if count > 0 else float('inf')

    def _krum_aggregation(self, updates: List[Dict[str, Any]]) -> AggregationResult:
        """Multi-Krum aggregation."""
        n = len(updates)
        f = int(self.byzantine_fraction * n)  # Number of Byzantine clients
        m = n - f - 2  # Number of clients to select

        if m <= 0:
            # Fall back to median if not enough clients
            return self._median_aggregation(updates)

        # Calculate scores for each client
        scores = []
        for i in range(n):
            distances = []
            for j in range(n):
                if i != j:
                    dist = self._calculate_distance(
                        updates[i]['weights'],
                        updates[j]['weights']
                    )
                    distances.append(dist)

            # Sort distances and take sum of m smallest
            distances.sort()
            score = sum(distances[:m])
            scores.append(score)

        # Select client with minimum score
        best_idx = np.argmin(scores)
        selected_update = updates[best_idx]

        return AggregationResult(
            global_weights=selected_update['weights'],
            metadata={'method': 'krum', 'selected_client': selected_update.get('client_id')}
        )

    def _median_aggregation(self, updates: List[Dict[str, Any]]) -> AggregationResult:
        """Coordinate-wise median aggregation."""
        aggregated = {}

        # Get all layer names
        layer_names = set()
        for update in updates:
            layer_names.update(update['weights'].keys())

        for layer in layer_names:
            layer_updates = []
            for update in updates:
                if layer in update['weights']:
                    layer_updates.append(update['weights'][layer])

            if layer_updates:
                # Coordinate-wise median
                aggregated[layer] = np.median(layer_updates, axis=0)

        return AggregationResult(
            global_weights=aggregated,
            metadata={'method': 'median', 'num_clients': len(updates)}
        )

    def _trimmed_mean_aggregation(self, updates: List[Dict[str, Any]]) -> AggregationResult:
        """Trimmed mean aggregation."""
        trim_fraction = self.byzantine_fraction
        aggregated = {}

        # Get all layer names
        layer_names = set()
        for update in updates:
            layer_names.update(update['weights'].keys())

        for layer in layer_names:
            layer_updates = []
            for update in updates:
                if layer in update['weights']:
                    layer_updates.append(update['weights'][layer])

            if layer_updates:
                # Trimmed mean
                aggregated[layer] = stats.trim_mean(
                    layer_updates,
                    trim_fraction,
                    axis=0
                )

        return AggregationResult(
            global_weights=aggregated,
            metadata={'method': 'trimmed_mean', 'trim_fraction': trim_fraction}
        )

    def _bulyan_aggregation(self, updates: List[Dict[str, Any]]) -> AggregationResult:
        """Bulyan aggregation (Krum + Trimmed mean)."""
        n = len(updates)
        f = int(self.byzantine_fraction * n)

        # First step: Select subset using Krum
        theta = n - 2 * f
        if theta <= 0:
            return self._median_aggregation(updates)

        selected_indices = []
        remaining_indices = list(range(n))

        for _ in range(theta):
            # Find best remaining client using Krum criterion
            best_score = float('inf')
            best_idx = -1

            for i in remaining_indices:
                distances = []
                for j in remaining_indices:
                    if i != j:
                        dist = self._calculate_distance(
                            updates[i]['weights'],
                            updates[j]['weights']
                        )
                        distances.append(dist)

                distances.sort()
                score = sum(distances[:theta-f-1])

                if score < best_score:
                    best_score = score
                    best_idx = i

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        # Second step: Trimmed mean on selected subset
        selected_updates = [updates[i] for i in selected_indices]
        return self._trimmed_mean_aggregation(selected_updates)


class PersonalizedAggregator(BaseAggregator):
    """Personalized federated learning aggregator."""

    def __init__(self, personalization_method: str = 'local_fine_tuning',
                 alpha: float = 0.5):
        """Initialize personalized aggregator.

        Args:
            personalization_method: Method for personalization
            alpha: Interpolation parameter between local and global
        """
        self.personalization_method = personalization_method
        self.alpha = alpha
        self.client_clusters = {}
        self.cluster_models = {}

    def aggregate(self, updates: List[Dict[str, Any]]) -> AggregationResult:
        """Perform personalized aggregation.

        Args:
            updates: List of client updates

        Returns:
            Aggregation result with personalized models
        """
        valid_updates = self.validate_updates(updates)

        if self.personalization_method == 'local_fine_tuning':
            return self._local_fine_tuning_aggregation(valid_updates)
        elif self.personalization_method == 'clustering':
            return self._clustering_aggregation(valid_updates)
        elif self.personalization_method == 'meta_learning':
            return self._meta_learning_aggregation(valid_updates)
        else:
            return self._interpolation_aggregation(valid_updates)

    def validate_updates(self, updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate updates."""
        valid = []
        for update in updates:
            if 'weights' in update and 'client_id' in update:
                valid.append(update)
        return valid

    def _local_fine_tuning_aggregation(self, updates: List[Dict[str, Any]]) -> AggregationResult:
        """Aggregate with local fine-tuning support."""
        # Compute global model
        fedavg = FedAvgAggregator(weighted=True)
        global_result = fedavg.aggregate(updates)

        # Create personalized models
        client_models = {}
        for update in updates:
            client_id = update['client_id']
            personalized = {}

            for layer, global_weights in global_result.global_weights.items():
                if layer in update['weights']:
                    # Interpolate between local and global
                    personalized[layer] = (
                        self.alpha * update['weights'][layer] +
                        (1 - self.alpha) * global_weights
                    )
                else:
                    personalized[layer] = global_weights

            client_models[client_id] = personalized

        return AggregationResult(
            global_weights=global_result.global_weights,
            client_weights=client_models,
            metadata={'personalization': 'local_fine_tuning', 'alpha': self.alpha}
        )

    def _clustering_aggregation(self, updates: List[Dict[str, Any]]) -> AggregationResult:
        """Cluster clients and create cluster-specific models."""
        # Extract features for clustering
        features = []
        client_ids = []

        for update in updates:
            # Use weight statistics as features
            feature_vec = []
            for layer, weights in update['weights'].items():
                feature_vec.extend([
                    np.mean(weights),
                    np.std(weights),
                    np.min(weights),
                    np.max(weights)
                ])
            features.append(feature_vec)
            client_ids.append(update['client_id'])

        # Pad features to same length
        max_len = max(len(f) for f in features)
        features = [f + [0] * (max_len - len(f)) for f in features]

        # Cluster clients
        n_clusters = min(3, len(updates) // 2)
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(features)
        else:
            clusters = [0] * len(updates)

        # Aggregate per cluster
        cluster_updates = defaultdict(list)
        for i, update in enumerate(updates):
            cluster_updates[clusters[i]].append(update)

        # Create cluster models
        fedavg = FedAvgAggregator(weighted=True)
        cluster_models = {}
        for cluster_id, cluster_members in cluster_updates.items():
            cluster_result = fedavg.aggregate(cluster_members)
            cluster_models[f"cluster_{cluster_id}"] = cluster_result.global_weights

        # Assign personalized models
        client_models = {}
        for i, client_id in enumerate(client_ids):
            client_models[client_id] = cluster_models[f"cluster_{clusters[i]}"]

        # Global model as average of cluster models
        global_weights = {}
        for layer in cluster_models[f"cluster_0"].keys():
            layer_weights = [m[layer] for m in cluster_models.values() if layer in m]
            global_weights[layer] = np.mean(layer_weights, axis=0)

        return AggregationResult(
            global_weights=global_weights,
            client_weights=client_models,
            metadata={'personalization': 'clustering', 'n_clusters': n_clusters}
        )

    def _meta_learning_aggregation(self, updates: List[Dict[str, Any]]) -> AggregationResult:
        """Meta-learning based personalization (MAML-style)."""
        # Compute meta-model optimized for fast adaptation
        fedavg = FedAvgAggregator(weighted=True)
        global_result = fedavg.aggregate(updates)

        # Simulate one-step gradient update for each client
        client_models = {}
        learning_rate = 0.01

        for update in updates:
            client_id = update['client_id']
            personalized = {}

            for layer, global_weights in global_result.global_weights.items():
                if layer in update['weights']:
                    # Compute pseudo-gradient
                    gradient = update['weights'][layer] - global_weights
                    # One-step update
                    personalized[layer] = global_weights + learning_rate * gradient
                else:
                    personalized[layer] = global_weights

            client_models[client_id] = personalized

        return AggregationResult(
            global_weights=global_result.global_weights,
            client_weights=client_models,
            metadata={'personalization': 'meta_learning', 'lr': learning_rate}
        )

    def _interpolation_aggregation(self, updates: List[Dict[str, Any]]) -> AggregationResult:
        """Simple interpolation between local and global."""
        return self._local_fine_tuning_aggregation(updates)


class AdaptiveAggregator(BaseAggregator):
    """Adaptive aggregator that selects strategy based on conditions."""

    def __init__(self):
        """Initialize adaptive aggregator."""
        self.aggregators = {
            'fedavg': FedAvgAggregator(),
            'byzantine': ByzantineRobustAggregator(),
            'personalized': PersonalizedAggregator()
        }
        self.history = []

    def aggregate(self, updates: List[Dict[str, Any]]) -> AggregationResult:
        """Adaptively select and apply aggregation strategy.

        Args:
            updates: List of client updates

        Returns:
            Aggregation result
        """
        strategy = self._select_strategy(updates)
        result = self.aggregators[strategy].aggregate(updates)

        self.history.append({
            'strategy': strategy,
            'num_clients': len(updates),
            'result': result
        })

        return result

    def validate_updates(self, updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate updates."""
        return updates

    def _select_strategy(self, updates: List[Dict[str, Any]]) -> str:
        """Select aggregation strategy based on current conditions."""
        n = len(updates)

        # Check for Byzantine behavior
        if self._detect_byzantine_behavior(updates):
            return 'byzantine'

        # Check for heterogeneity
        if self._detect_heterogeneity(updates):
            return 'personalized'

        # Default to FedAvg
        return 'fedavg'

    def _detect_byzantine_behavior(self, updates: List[Dict[str, Any]]) -> bool:
        """Detect potential Byzantine behavior."""
        if len(updates) < 3:
            return False

        # Calculate variance in updates
        variances = []
        for layer in updates[0]['weights'].keys():
            layer_updates = [u['weights'][layer] for u in updates if layer in u['weights']]
            if layer_updates:
                variance = np.var([np.linalg.norm(u) for u in layer_updates])
                variances.append(variance)

        # High variance indicates potential Byzantine behavior
        avg_variance = np.mean(variances) if variances else 0
        return avg_variance > 10.0  # Threshold can be tuned

    def _detect_heterogeneity(self, updates: List[Dict[str, Any]]) -> bool:
        """Detect data heterogeneity among clients."""
        if len(updates) < 2:
            return False

        # Compare weight distributions
        similarities = []
        for i in range(len(updates)):
            for j in range(i + 1, len(updates)):
                sim = self._calculate_similarity(
                    updates[i]['weights'],
                    updates[j]['weights']
                )
                similarities.append(sim)

        # Low similarity indicates heterogeneity
        avg_similarity = np.mean(similarities) if similarities else 1.0
        return avg_similarity < 0.7  # Threshold can be tuned

    def _calculate_similarity(self, weights1: Dict[str, np.ndarray],
                            weights2: Dict[str, np.ndarray]) -> float:
        """Calculate cosine similarity between weight updates."""
        vec1 = []
        vec2 = []

        for layer in weights1.keys():
            if layer in weights2:
                vec1.extend(weights1[layer].flatten())
                vec2.extend(weights2[layer].flatten())

        if not vec1 or not vec2:
            return 0.0

        return float(cosine_similarity([vec1], [vec2])[0, 0])