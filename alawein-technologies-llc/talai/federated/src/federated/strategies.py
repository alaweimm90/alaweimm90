"""
Strategies for efficient federated learning.

Implements client selection, model compression, communication optimization,
and contribution tracking using Shapley values.
"""

import hashlib
import logging
import zlib
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import sparse
from scipy.stats import entropy

logger = logging.getLogger(__name__)


@dataclass
class ClientProfile:
    """Profile of a federated client."""
    client_id: str
    compute_capacity: float  # FLOPS
    bandwidth: float  # Mbps
    data_size: int  # Number of samples
    data_quality: float  # 0-1 score
    reliability: float  # Historical success rate
    latency: float  # Network latency in ms
    battery_level: Optional[float] = None  # For mobile clients
    data_distribution: Optional[Dict[str, float]] = None


class ClientSelector:
    """Advanced client selection strategies for federated learning."""

    def __init__(self, strategy: str = 'random'):
        """Initialize client selector.

        Args:
            strategy: Selection strategy ('random', 'oort', 'adaptive', 'fairness')
        """
        self.strategy = strategy
        self.client_history = {}
        self.round_count = 0

    def select_clients(self, client_profiles: List[ClientProfile],
                      num_clients: int,
                      constraints: Optional[Dict[str, Any]] = None) -> List[str]:
        """Select clients for training round.

        Args:
            client_profiles: List of client profiles
            num_clients: Number of clients to select
            constraints: Optional constraints (min_data_size, max_latency, etc.)

        Returns:
            List of selected client IDs
        """
        if self.strategy == 'random':
            return self._random_selection(client_profiles, num_clients, constraints)
        elif self.strategy == 'oort':
            return self._oort_selection(client_profiles, num_clients, constraints)
        elif self.strategy == 'adaptive':
            return self._adaptive_selection(client_profiles, num_clients, constraints)
        elif self.strategy == 'fairness':
            return self._fairness_selection(client_profiles, num_clients, constraints)
        else:
            return self._random_selection(client_profiles, num_clients, constraints)

    def _random_selection(self, profiles: List[ClientProfile],
                         num_clients: int,
                         constraints: Optional[Dict[str, Any]]) -> List[str]:
        """Random client selection."""
        eligible = self._filter_by_constraints(profiles, constraints)

        if len(eligible) <= num_clients:
            return [p.client_id for p in eligible]

        selected_indices = np.random.choice(
            len(eligible),
            size=num_clients,
            replace=False
        )
        return [eligible[i].client_id for i in selected_indices]

    def _oort_selection(self, profiles: List[ClientProfile],
                       num_clients: int,
                       constraints: Optional[Dict[str, Any]]) -> List[str]:
        """Oort: Efficient and balanced client selection."""
        eligible = self._filter_by_constraints(profiles, constraints)

        if len(eligible) <= num_clients:
            return [p.client_id for p in eligible]

        # Calculate utility score for each client
        utilities = []
        for profile in eligible:
            # Combine statistical utility and system utility
            stat_utility = self._calculate_statistical_utility(profile)
            sys_utility = self._calculate_system_utility(profile)

            # Balance exploration and exploitation
            exploration_bonus = self._get_exploration_bonus(profile.client_id)

            utility = 0.5 * stat_utility + 0.3 * sys_utility + 0.2 * exploration_bonus
            utilities.append(utility)

        # Select top clients by utility
        utilities = np.array(utilities)
        selected_indices = np.argsort(utilities)[-num_clients:]

        selected = [eligible[i].client_id for i in selected_indices]
        self._update_selection_history(selected)

        return selected

    def _adaptive_selection(self, profiles: List[ClientProfile],
                          num_clients: int,
                          constraints: Optional[Dict[str, Any]]) -> List[str]:
        """Adaptive selection based on round progress."""
        eligible = self._filter_by_constraints(profiles, constraints)

        # Early rounds: explore diverse clients
        # Later rounds: exploit best performers
        exploration_rate = max(0.1, 1.0 - self.round_count / 100)

        num_explore = int(num_clients * exploration_rate)
        num_exploit = num_clients - num_explore

        # Exploration: random selection
        explore_pool = [p for p in eligible if self._is_underexplored(p.client_id)]
        if explore_pool:
            explore_indices = np.random.choice(
                len(explore_pool),
                size=min(num_explore, len(explore_pool)),
                replace=False
            )
            explored = [explore_pool[i].client_id for i in explore_indices]
        else:
            explored = []

        # Exploitation: select best performers
        exploit_pool = [p for p in eligible if p.client_id not in explored]
        scores = [self._get_performance_score(p) for p in exploit_pool]

        if scores:
            best_indices = np.argsort(scores)[-num_exploit:]
            exploited = [exploit_pool[i].client_id for i in best_indices]
        else:
            exploited = []

        selected = explored + exploited
        self._update_selection_history(selected)
        self.round_count += 1

        return selected

    def _fairness_selection(self, profiles: List[ClientProfile],
                          num_clients: int,
                          constraints: Optional[Dict[str, Any]]) -> List[str]:
        """Fair selection ensuring all clients participate."""
        eligible = self._filter_by_constraints(profiles, constraints)

        # Calculate fairness score (inverse of participation count)
        fairness_scores = []
        for profile in eligible:
            participation_count = self.client_history.get(profile.client_id, {}).get('count', 0)
            fairness_score = 1.0 / (1 + participation_count)
            fairness_scores.append(fairness_score)

        # Probabilistic selection based on fairness
        fairness_scores = np.array(fairness_scores)
        probabilities = fairness_scores / fairness_scores.sum()

        selected_indices = np.random.choice(
            len(eligible),
            size=min(num_clients, len(eligible)),
            replace=False,
            p=probabilities
        )

        selected = [eligible[i].client_id for i in selected_indices]
        self._update_selection_history(selected)

        return selected

    def _filter_by_constraints(self, profiles: List[ClientProfile],
                              constraints: Optional[Dict[str, Any]]) -> List[ClientProfile]:
        """Filter clients by constraints."""
        if not constraints:
            return profiles

        filtered = []
        for profile in profiles:
            if constraints.get('min_data_size') and profile.data_size < constraints['min_data_size']:
                continue
            if constraints.get('max_latency') and profile.latency > constraints['max_latency']:
                continue
            if constraints.get('min_bandwidth') and profile.bandwidth < constraints['min_bandwidth']:
                continue
            if constraints.get('min_battery') and profile.battery_level and profile.battery_level < constraints['min_battery']:
                continue

            filtered.append(profile)

        return filtered

    def _calculate_statistical_utility(self, profile: ClientProfile) -> float:
        """Calculate statistical utility of client's data."""
        # Data quantity utility
        quantity_util = np.log(1 + profile.data_size)

        # Data quality utility
        quality_util = profile.data_quality

        # Data diversity utility (if distribution available)
        if profile.data_distribution:
            # Use entropy as diversity measure
            distribution = list(profile.data_distribution.values())
            diversity_util = entropy(distribution) / np.log(len(distribution))
        else:
            diversity_util = 0.5

        return 0.4 * quantity_util + 0.4 * quality_util + 0.2 * diversity_util

    def _calculate_system_utility(self, profile: ClientProfile) -> float:
        """Calculate system utility of client."""
        # Normalize metrics
        compute_util = np.log(1 + profile.compute_capacity) / 20  # Normalized
        bandwidth_util = np.log(1 + profile.bandwidth) / 10
        latency_util = 1.0 / (1 + profile.latency / 100)
        reliability_util = profile.reliability

        return 0.3 * compute_util + 0.3 * bandwidth_util + 0.2 * latency_util + 0.2 * reliability_util

    def _get_exploration_bonus(self, client_id: str) -> float:
        """Get exploration bonus for underexplored clients."""
        if client_id not in self.client_history:
            return 1.0

        rounds_since_selection = self.round_count - self.client_history[client_id].get('last_round', 0)
        return min(1.0, rounds_since_selection / 10)

    def _is_underexplored(self, client_id: str) -> bool:
        """Check if client is underexplored."""
        if client_id not in self.client_history:
            return True

        participation_rate = self.client_history[client_id]['count'] / max(1, self.round_count)
        return participation_rate < 0.1

    def _get_performance_score(self, profile: ClientProfile) -> float:
        """Get historical performance score."""
        if profile.client_id not in self.client_history:
            return 0.5

        history = self.client_history[profile.client_id]
        # Combine various performance metrics
        return history.get('avg_accuracy', 0.5) * profile.reliability

    def _update_selection_history(self, selected_clients: List[str]):
        """Update client selection history."""
        for client_id in selected_clients:
            if client_id not in self.client_history:
                self.client_history[client_id] = {'count': 0, 'last_round': 0}

            self.client_history[client_id]['count'] += 1
            self.client_history[client_id]['last_round'] = self.round_count


class ModelCompressor:
    """Model compression techniques for efficient communication."""

    def __init__(self, method: str = 'quantization', compression_ratio: float = 0.1):
        """Initialize model compressor.

        Args:
            method: Compression method ('quantization', 'sparsification', 'low_rank', 'hybrid')
            compression_ratio: Target compression ratio
        """
        self.method = method
        self.compression_ratio = compression_ratio
        self.decompression_info = {}

    def compress(self, weights: Dict[str, np.ndarray]) -> Tuple[bytes, Dict[str, Any]]:
        """Compress model weights.

        Args:
            weights: Model weights to compress

        Returns:
            Compressed data and decompression info
        """
        if self.method == 'quantization':
            return self._quantization_compress(weights)
        elif self.method == 'sparsification':
            return self._sparsification_compress(weights)
        elif self.method == 'low_rank':
            return self._low_rank_compress(weights)
        elif self.method == 'hybrid':
            return self._hybrid_compress(weights)
        else:
            return self._quantization_compress(weights)

    def decompress(self, compressed_data: bytes,
                  decompression_info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Decompress model weights.

        Args:
            compressed_data: Compressed data
            decompression_info: Information for decompression

        Returns:
            Decompressed weights
        """
        method = decompression_info.get('method', 'quantization')

        if method == 'quantization':
            return self._quantization_decompress(compressed_data, decompression_info)
        elif method == 'sparsification':
            return self._sparsification_decompress(compressed_data, decompression_info)
        elif method == 'low_rank':
            return self._low_rank_decompress(compressed_data, decompression_info)
        elif method == 'hybrid':
            return self._hybrid_decompress(compressed_data, decompression_info)
        else:
            return {}

    def _quantization_compress(self, weights: Dict[str, np.ndarray]) -> Tuple[bytes, Dict[str, Any]]:
        """Compress using quantization."""
        compressed = {}
        info = {'method': 'quantization', 'layers': {}}

        for layer, w in weights.items():
            # Dynamic quantization to 8-bit
            min_val = np.min(w)
            max_val = np.max(w)
            scale = (max_val - min_val) / 255

            if scale == 0:
                quantized = np.zeros(w.shape, dtype=np.uint8)
            else:
                quantized = ((w - min_val) / scale).astype(np.uint8)

            compressed[layer] = quantized
            info['layers'][layer] = {
                'shape': w.shape,
                'min': float(min_val),
                'max': float(max_val),
                'scale': float(scale)
            }

        # Serialize compressed weights
        serialized = self._serialize_weights(compressed)
        return zlib.compress(serialized), info

    def _quantization_decompress(self, compressed_data: bytes,
                                info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Decompress quantized weights."""
        decompressed_data = zlib.decompress(compressed_data)
        quantized = self._deserialize_weights(decompressed_data, info)

        weights = {}
        for layer, q in quantized.items():
            layer_info = info['layers'][layer]
            scale = layer_info['scale']
            min_val = layer_info['min']

            if scale == 0:
                weights[layer] = np.full(layer_info['shape'], min_val)
            else:
                weights[layer] = q.astype(np.float32) * scale + min_val

        return weights

    def _sparsification_compress(self, weights: Dict[str, np.ndarray]) -> Tuple[bytes, Dict[str, Any]]:
        """Compress using sparsification."""
        compressed = {}
        info = {'method': 'sparsification', 'layers': {}}

        for layer, w in weights.items():
            # Keep only top-k values
            flat = w.flatten()
            k = int(len(flat) * self.compression_ratio)
            threshold = np.partition(np.abs(flat), -k)[-k] if k > 0 else 0

            # Create sparse matrix
            mask = np.abs(w) >= threshold
            sparse_w = w * mask

            # Convert to sparse format
            sparse_matrix = sparse.csr_matrix(sparse_w)

            compressed[layer] = {
                'data': sparse_matrix.data,
                'indices': sparse_matrix.indices,
                'indptr': sparse_matrix.indptr
            }

            info['layers'][layer] = {
                'shape': w.shape,
                'nnz': sparse_matrix.nnz
            }

        serialized = self._serialize_sparse(compressed)
        return zlib.compress(serialized), info

    def _sparsification_decompress(self, compressed_data: bytes,
                                  info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Decompress sparse weights."""
        decompressed_data = zlib.decompress(compressed_data)
        sparse_weights = self._deserialize_sparse(decompressed_data, info)

        weights = {}
        for layer, sparse_data in sparse_weights.items():
            layer_info = info['layers'][layer]
            shape = layer_info['shape']

            # Reconstruct sparse matrix
            sparse_matrix = sparse.csr_matrix(
                (sparse_data['data'], sparse_data['indices'], sparse_data['indptr']),
                shape=(np.prod(shape[:-1]), shape[-1]) if len(shape) > 1 else shape
            )

            weights[layer] = sparse_matrix.toarray().reshape(shape)

        return weights

    def _low_rank_compress(self, weights: Dict[str, np.ndarray]) -> Tuple[bytes, Dict[str, Any]]:
        """Compress using low-rank approximation."""
        compressed = {}
        info = {'method': 'low_rank', 'layers': {}}

        for layer, w in weights.items():
            if len(w.shape) == 2:
                # Apply SVD for matrices
                rank = max(1, int(min(w.shape) * self.compression_ratio))
                u, s, vt = np.linalg.svd(w, full_matrices=False)

                # Keep top-k singular values
                u_k = u[:, :rank]
                s_k = s[:rank]
                vt_k = vt[:rank, :]

                compressed[layer] = {
                    'u': u_k,
                    's': s_k,
                    'vt': vt_k
                }

                info['layers'][layer] = {
                    'shape': w.shape,
                    'rank': rank
                }
            else:
                # For non-matrix layers, use quantization
                compressed[layer] = w
                info['layers'][layer] = {'shape': w.shape, 'compressed': False}

        serialized = self._serialize_low_rank(compressed)
        return zlib.compress(serialized), info

    def _low_rank_decompress(self, compressed_data: bytes,
                            info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Decompress low-rank approximation."""
        decompressed_data = zlib.decompress(compressed_data)
        low_rank = self._deserialize_low_rank(decompressed_data, info)

        weights = {}
        for layer, data in low_rank.items():
            layer_info = info['layers'][layer]

            if 'rank' in layer_info:
                # Reconstruct from SVD
                weights[layer] = data['u'] @ np.diag(data['s']) @ data['vt']
            else:
                weights[layer] = data

        return weights

    def _hybrid_compress(self, weights: Dict[str, np.ndarray]) -> Tuple[bytes, Dict[str, Any]]:
        """Hybrid compression combining multiple techniques."""
        # Apply different compression to different layers
        compressed = {}
        info = {'method': 'hybrid', 'layers': {}}

        for layer, w in weights.items():
            if 'conv' in layer.lower() or 'dense' in layer.lower():
                # Use low-rank for large layers
                if len(w.shape) == 2 and min(w.shape) > 10:
                    result, layer_info = self._compress_single_low_rank(w)
                    compressed[layer] = result
                    info['layers'][layer] = layer_info
                else:
                    # Use quantization for smaller layers
                    result, layer_info = self._compress_single_quantization(w)
                    compressed[layer] = result
                    info['layers'][layer] = layer_info
            else:
                # Use sparsification for other layers
                result, layer_info = self._compress_single_sparsification(w)
                compressed[layer] = result
                info['layers'][layer] = layer_info

        serialized = self._serialize_hybrid(compressed, info)
        return zlib.compress(serialized), info

    def _hybrid_decompress(self, compressed_data: bytes,
                          info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Decompress hybrid compression."""
        decompressed_data = zlib.decompress(compressed_data)
        return self._deserialize_hybrid(decompressed_data, info)

    def _serialize_weights(self, weights: Dict[str, np.ndarray]) -> bytes:
        """Serialize weights to bytes."""
        return pickle.dumps(weights)

    def _deserialize_weights(self, data: bytes, info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Deserialize weights from bytes."""
        return pickle.loads(data)

    def _serialize_sparse(self, sparse_data: Dict) -> bytes:
        """Serialize sparse data."""
        import pickle
        return pickle.dumps(sparse_data)

    def _deserialize_sparse(self, data: bytes, info: Dict[str, Any]) -> Dict:
        """Deserialize sparse data."""
        import pickle
        return pickle.loads(data)

    def _serialize_low_rank(self, low_rank_data: Dict) -> bytes:
        """Serialize low-rank data."""
        import pickle
        return pickle.dumps(low_rank_data)

    def _deserialize_low_rank(self, data: bytes, info: Dict[str, Any]) -> Dict:
        """Deserialize low-rank data."""
        import pickle
        return pickle.loads(data)

    def _serialize_hybrid(self, compressed: Dict, info: Dict[str, Any]) -> bytes:
        """Serialize hybrid compressed data."""
        import pickle
        return pickle.dumps(compressed)

    def _deserialize_hybrid(self, data: bytes, info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Deserialize hybrid compressed data."""
        import pickle
        compressed = pickle.loads(data)
        weights = {}

        for layer, data in compressed.items():
            layer_info = info['layers'][layer]
            method = layer_info.get('method', 'quantization')

            if method == 'low_rank':
                weights[layer] = data['u'] @ np.diag(data['s']) @ data['vt']
            elif method == 'quantization':
                scale = layer_info['scale']
                min_val = layer_info['min']
                weights[layer] = data.astype(np.float32) * scale + min_val
            elif method == 'sparsification':
                shape = layer_info['shape']
                sparse_matrix = sparse.csr_matrix(
                    (data['data'], data['indices'], data['indptr']),
                    shape=(np.prod(shape[:-1]), shape[-1]) if len(shape) > 1 else shape
                )
                weights[layer] = sparse_matrix.toarray().reshape(shape)
            else:
                weights[layer] = data

        return weights

    def _compress_single_low_rank(self, w: np.ndarray) -> Tuple[Dict, Dict]:
        """Compress single layer using low-rank."""
        rank = max(1, int(min(w.shape) * self.compression_ratio))
        u, s, vt = np.linalg.svd(w, full_matrices=False)
        u_k = u[:, :rank]
        s_k = s[:rank]
        vt_k = vt[:rank, :]

        return {'u': u_k, 's': s_k, 'vt': vt_k}, {'shape': w.shape, 'rank': rank, 'method': 'low_rank'}

    def _compress_single_quantization(self, w: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Compress single layer using quantization."""
        min_val = np.min(w)
        max_val = np.max(w)
        scale = (max_val - min_val) / 255

        if scale == 0:
            quantized = np.zeros(w.shape, dtype=np.uint8)
        else:
            quantized = ((w - min_val) / scale).astype(np.uint8)

        return quantized, {
            'shape': w.shape,
            'min': float(min_val),
            'max': float(max_val),
            'scale': float(scale),
            'method': 'quantization'
        }

    def _compress_single_sparsification(self, w: np.ndarray) -> Tuple[Dict, Dict]:
        """Compress single layer using sparsification."""
        flat = w.flatten()
        k = int(len(flat) * self.compression_ratio)
        threshold = np.partition(np.abs(flat), -k)[-k] if k > 0 else 0

        mask = np.abs(w) >= threshold
        sparse_w = w * mask
        sparse_matrix = sparse.csr_matrix(sparse_w)

        return {
            'data': sparse_matrix.data,
            'indices': sparse_matrix.indices,
            'indptr': sparse_matrix.indptr
        }, {
            'shape': w.shape,
            'nnz': sparse_matrix.nnz,
            'method': 'sparsification'
        }


class CommunicationOptimizer:
    """Optimize communication in federated learning."""

    def __init__(self):
        """Initialize communication optimizer."""
        self.communication_rounds = 0
        self.total_bytes_sent = 0
        self.total_bytes_received = 0

    def optimize_communication_round(self, clients: List[str],
                                    model_size: int) -> Dict[str, Any]:
        """Optimize communication for a round.

        Args:
            clients: List of client IDs
            model_size: Size of model in bytes

        Returns:
            Communication plan
        """
        plan = {
            'clients': clients,
            'model_size': model_size,
            'strategy': 'gradient_compression',
            'compression_ratio': self._adaptive_compression_ratio(len(clients)),
            'batch_communication': self._should_batch(len(clients)),
            'async_aggregation': len(clients) > 20
        }

        self.communication_rounds += 1
        return plan

    def _adaptive_compression_ratio(self, num_clients: int) -> float:
        """Determine compression ratio based on number of clients."""
        if num_clients < 10:
            return 0.5  # Less compression for few clients
        elif num_clients < 50:
            return 0.2  # Moderate compression
        else:
            return 0.1  # High compression for many clients

    def _should_batch(self, num_clients: int) -> bool:
        """Determine if batched communication is beneficial."""
        return num_clients > 10


class ContributionTracker:
    """Track and calculate client contributions using Shapley values."""

    def __init__(self):
        """Initialize contribution tracker."""
        self.contributions = {}
        self.round_performances = []

    def calculate_shapley_values(self, clients: List[str],
                                performance_func: callable,
                                sample_size: int = 100) -> Dict[str, float]:
        """Calculate Shapley values for client contributions.

        Args:
            clients: List of client IDs
            performance_func: Function to evaluate subset performance
            sample_size: Number of samples for Monte Carlo approximation

        Returns:
            Shapley values for each client
        """
        n = len(clients)
        shapley_values = {client: 0.0 for client in clients}

        if n > 10:
            # Use Monte Carlo sampling for large coalitions
            for _ in range(sample_size):
                # Random permutation
                permutation = np.random.permutation(clients)

                for i, client in enumerate(permutation):
                    # Coalition without client
                    coalition_without = permutation[:i].tolist()
                    # Coalition with client
                    coalition_with = permutation[:i+1].tolist()

                    # Calculate marginal contribution
                    if coalition_without:
                        perf_without = performance_func(coalition_without)
                    else:
                        perf_without = 0

                    perf_with = performance_func(coalition_with)
                    marginal = perf_with - perf_without

                    shapley_values[client] += marginal / sample_size
        else:
            # Exact calculation for small coalitions
            for client in clients:
                for size in range(n):
                    for coalition in combinations([c for c in clients if c != client], size):
                        coalition_list = list(coalition)

                        # Performance without client
                        if coalition_list:
                            perf_without = performance_func(coalition_list)
                        else:
                            perf_without = 0

                        # Performance with client
                        coalition_with = coalition_list + [client]
                        perf_with = performance_func(coalition_with)

                        # Weight for this coalition size
                        weight = np.math.factorial(size) * np.math.factorial(n - size - 1) / np.math.factorial(n)

                        shapley_values[client] += weight * (perf_with - perf_without)

        return shapley_values

    def update_contributions(self, round_num: int,
                            client_contributions: Dict[str, float],
                            round_performance: float):
        """Update contribution tracking.

        Args:
            round_num: Round number
            client_contributions: Contributions for this round
            round_performance: Overall round performance
        """
        for client, contribution in client_contributions.items():
            if client not in self.contributions:
                self.contributions[client] = []

            self.contributions[client].append({
                'round': round_num,
                'contribution': contribution,
                'performance': round_performance
            })

        self.round_performances.append({
            'round': round_num,
            'performance': round_performance,
            'num_clients': len(client_contributions)
        })

    def get_fair_rewards(self, total_reward: float) -> Dict[str, float]:
        """Calculate fair reward distribution based on contributions.

        Args:
            total_reward: Total reward to distribute

        Returns:
            Fair reward for each client
        """
        if not self.contributions:
            return {}

        # Calculate average contribution per client
        avg_contributions = {}
        for client, history in self.contributions.items():
            avg_contributions[client] = np.mean([h['contribution'] for h in history])

        # Normalize to sum to 1
        total_contribution = sum(avg_contributions.values())
        if total_contribution == 0:
            return {client: total_reward / len(avg_contributions) for client in avg_contributions}

        # Distribute reward proportionally
        rewards = {}
        for client, contribution in avg_contributions.items():
            rewards[client] = (contribution / total_contribution) * total_reward

        return rewards

# Import pickle for serialization (was missing in previous code)
import pickle