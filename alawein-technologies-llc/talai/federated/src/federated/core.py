"""
Core federated learning infrastructure for multi-institution research collaboration.

Supports horizontal and vertical federated learning with secure aggregation,
differential privacy, and Byzantine robustness.
"""

import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


class FederationType(Enum):
    """Types of federated learning supported."""
    HORIZONTAL = "horizontal"  # Same features, different samples
    VERTICAL = "vertical"      # Different features, same samples
    TRANSFER = "transfer"      # Knowledge transfer between domains
    HIERARCHICAL = "hierarchical"  # Multi-level federation


class ClientStatus(Enum):
    """Client status in federation."""
    IDLE = "idle"
    TRAINING = "training"
    AGGREGATING = "aggregating"
    VALIDATING = "validating"
    DROPPED = "dropped"
    MALICIOUS = "malicious"


@dataclass
class FederationConfig:
    """Configuration for federated learning."""
    federation_type: FederationType = FederationType.HORIZONTAL
    num_rounds: int = 100
    min_clients: int = 2
    fraction_fit: float = 0.1
    fraction_eval: float = 0.1
    min_available_clients: int = 2
    privacy_budget: float = 10.0  # Epsilon for differential privacy
    delta: float = 1e-5  # Delta for differential privacy
    secure_aggregation: bool = True
    byzantine_robust: bool = True
    compression_enabled: bool = True
    compression_ratio: float = 0.1
    client_timeout: int = 300  # seconds
    checkpoint_interval: int = 10
    audit_enabled: bool = True
    encryption_enabled: bool = True


@dataclass
class ClientMetadata:
    """Metadata for a federated client."""
    client_id: str
    institution: str
    join_time: datetime
    public_key: Optional[bytes] = None
    capabilities: Dict[str, Any] = field(default_factory=dict)
    data_statistics: Dict[str, Any] = field(default_factory=dict)
    contribution_score: float = 0.0
    trust_score: float = 1.0
    rounds_participated: int = 0
    last_active: datetime = field(default_factory=datetime.now)


@dataclass
class ModelUpdate:
    """Model update from a client."""
    client_id: str
    round_number: int
    weights: Dict[str, np.ndarray]
    metrics: Dict[str, float]
    num_samples: int
    computation_time: float
    timestamp: datetime
    signature: Optional[bytes] = None
    encrypted: bool = False


class FederationProtocol(ABC):
    """Abstract base class for federation protocols."""

    @abstractmethod
    def initialize_round(self, round_number: int) -> Dict[str, Any]:
        """Initialize a new federation round."""
        pass

    @abstractmethod
    def aggregate_updates(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Aggregate model updates from clients."""
        pass

    @abstractmethod
    def validate_update(self, update: ModelUpdate) -> bool:
        """Validate a client update."""
        pass

    @abstractmethod
    def finalize_round(self, aggregated_model: Dict[str, np.ndarray]) -> None:
        """Finalize the current round."""
        pass


class SecureAggregator:
    """Secure aggregation protocol for privacy-preserving federated learning."""

    def __init__(self, threshold: int = 3):
        """Initialize secure aggregator.

        Args:
            threshold: Minimum number of clients for secure aggregation
        """
        self.threshold = threshold
        self.client_keys = {}
        self.shared_secrets = defaultdict(dict)
        self.masked_updates = {}

    def generate_keys(self, client_id: str) -> Tuple[bytes, bytes]:
        """Generate public/private key pair for client."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        self.client_keys[client_id] = {
            'private': private_key,
            'public': public_key
        }

        return public_pem, private_pem

    def create_shared_secret(self, client1_id: str, client2_id: str) -> bytes:
        """Create shared secret between two clients."""
        # Simplified Diffie-Hellman key exchange
        shared_key = Fernet.generate_key()
        self.shared_secrets[client1_id][client2_id] = shared_key
        self.shared_secrets[client2_id][client1_id] = shared_key
        return shared_key

    def mask_weights(self, weights: Dict[str, np.ndarray],
                     client_id: str, round_number: int) -> Dict[str, np.ndarray]:
        """Mask weights with random noise for secure aggregation."""
        masked = {}
        for layer, w in weights.items():
            # Generate deterministic mask based on round and client
            seed = hash((client_id, round_number, layer)) % (2**32)
            np.random.seed(seed)
            mask = np.random.randn(*w.shape) * 0.01
            masked[layer] = w + mask
        return masked

    def unmask_aggregate(self, masked_updates: List[Dict[str, np.ndarray]],
                        client_ids: List[str], round_number: int) -> Dict[str, np.ndarray]:
        """Remove masks after aggregation."""
        if len(masked_updates) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} clients for secure aggregation")

        # Average masked updates
        aggregated = {}
        for layer in masked_updates[0].keys():
            stacked = np.stack([u[layer] for u in masked_updates])
            aggregated[layer] = np.mean(stacked, axis=0)

        # Remove masks
        for client_id in client_ids:
            for layer in aggregated.keys():
                seed = hash((client_id, round_number, layer)) % (2**32)
                np.random.seed(seed)
                mask = np.random.randn(*aggregated[layer].shape) * 0.01
                aggregated[layer] -= mask / len(client_ids)

        return aggregated


class FederatedServer:
    """Central server for federated learning coordination."""

    def __init__(self, config: FederationConfig):
        """Initialize federated server.

        Args:
            config: Federation configuration
        """
        self.config = config
        self.clients: Dict[str, ClientMetadata] = {}
        self.current_round = 0
        self.global_model: Optional[Dict[str, np.ndarray]] = None
        self.round_updates: List[ModelUpdate] = []
        self.secure_aggregator = SecureAggregator() if config.secure_aggregation else None
        self.audit_log = []
        self.model_checkpoints = {}

    def register_client(self, client_id: str, institution: str,
                       capabilities: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new client with the federation.

        Args:
            client_id: Unique client identifier
            institution: Institution name
            capabilities: Client capabilities (compute, data size, etc.)

        Returns:
            Registration confirmation with credentials
        """
        if client_id in self.clients:
            return {"status": "already_registered", "client_id": client_id}

        metadata = ClientMetadata(
            client_id=client_id,
            institution=institution,
            join_time=datetime.now(),
            capabilities=capabilities
        )

        response = {"status": "registered", "client_id": client_id}

        if self.config.encryption_enabled and self.secure_aggregator:
            public_key, private_key = self.secure_aggregator.generate_keys(client_id)
            metadata.public_key = public_key
            response["private_key"] = private_key.decode()
            response["public_key"] = public_key.decode()

        self.clients[client_id] = metadata

        if self.config.audit_enabled:
            self._audit_event("client_registered", {
                "client_id": client_id,
                "institution": institution,
                "timestamp": datetime.now().isoformat()
            })

        return response

    def select_clients(self, round_number: int) -> List[str]:
        """Select clients for current round.

        Args:
            round_number: Current round number

        Returns:
            List of selected client IDs
        """
        available_clients = [
            cid for cid, meta in self.clients.items()
            if meta.trust_score > 0.5 and
            (datetime.now() - meta.last_active).seconds < self.config.client_timeout
        ]

        if len(available_clients) < self.config.min_available_clients:
            logger.warning(f"Not enough clients available: {len(available_clients)}")
            return []

        # Client selection strategy (can be customized)
        num_clients = max(
            self.config.min_clients,
            int(self.config.fraction_fit * len(available_clients))
        )

        # Weighted selection based on trust score and contribution
        weights = []
        for cid in available_clients:
            meta = self.clients[cid]
            weight = meta.trust_score * (1 + meta.contribution_score)
            weights.append(weight)

        weights = np.array(weights)
        weights = weights / weights.sum()

        selected = np.random.choice(
            available_clients,
            size=min(num_clients, len(available_clients)),
            replace=False,
            p=weights
        ).tolist()

        if self.config.audit_enabled:
            self._audit_event("clients_selected", {
                "round": round_number,
                "selected": selected,
                "total_available": len(available_clients)
            })

        return selected

    def receive_update(self, update: ModelUpdate) -> bool:
        """Receive and validate model update from client.

        Args:
            update: Model update from client

        Returns:
            True if update is valid and accepted
        """
        # Validate update
        if update.client_id not in self.clients:
            logger.error(f"Unknown client: {update.client_id}")
            return False

        if update.round_number != self.current_round:
            logger.error(f"Wrong round: expected {self.current_round}, got {update.round_number}")
            return False

        # Check for Byzantine behavior
        if self.config.byzantine_robust and self._is_byzantine(update):
            logger.warning(f"Byzantine behavior detected from {update.client_id}")
            self.clients[update.client_id].trust_score *= 0.5
            return False

        # Update client metadata
        meta = self.clients[update.client_id]
        meta.last_active = datetime.now()
        meta.rounds_participated += 1

        # Store update
        self.round_updates.append(update)

        if self.config.audit_enabled:
            self._audit_event("update_received", {
                "client_id": update.client_id,
                "round": update.round_number,
                "num_samples": update.num_samples,
                "metrics": update.metrics
            })

        return True

    def aggregate_round(self) -> Optional[Dict[str, np.ndarray]]:
        """Aggregate updates for current round.

        Returns:
            Aggregated model weights or None if aggregation fails
        """
        if len(self.round_updates) < self.config.min_clients:
            logger.error(f"Not enough updates: {len(self.round_updates)}")
            return None

        # Filter out malicious updates if Byzantine robust
        valid_updates = self.round_updates
        if self.config.byzantine_robust:
            valid_updates = self._filter_byzantine_updates(self.round_updates)

        # Aggregate based on federation type
        if self.config.federation_type == FederationType.HORIZONTAL:
            aggregated = self._fedavg_aggregate(valid_updates)
        elif self.config.federation_type == FederationType.VERTICAL:
            aggregated = self._vertical_aggregate(valid_updates)
        else:
            aggregated = self._fedavg_aggregate(valid_updates)

        # Apply differential privacy if enabled
        if self.config.privacy_budget > 0:
            aggregated = self._apply_differential_privacy(aggregated)

        # Update global model
        self.global_model = aggregated

        # Checkpoint if needed
        if self.current_round % self.config.checkpoint_interval == 0:
            self._save_checkpoint()

        # Calculate contribution scores
        self._update_contribution_scores(valid_updates)

        # Clear round updates
        self.round_updates = []
        self.current_round += 1

        return aggregated

    def _fedavg_aggregate(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """FedAvg aggregation strategy."""
        total_samples = sum(u.num_samples for u in updates)
        aggregated = {}

        for layer in updates[0].weights.keys():
            weighted_sum = np.zeros_like(updates[0].weights[layer])
            for update in updates:
                weight = update.num_samples / total_samples
                weighted_sum += update.weights[layer] * weight
            aggregated[layer] = weighted_sum

        return aggregated

    def _vertical_aggregate(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Vertical federated learning aggregation."""
        # In vertical FL, different clients have different features
        # Aggregate by concatenating feature representations
        aggregated = {}

        for update in updates:
            for layer, weights in update.weights.items():
                client_layer = f"{layer}_{update.client_id}"
                aggregated[client_layer] = weights

        return aggregated

    def _is_byzantine(self, update: ModelUpdate) -> bool:
        """Detect potential Byzantine behavior."""
        if not self.global_model:
            return False

        # Check for extreme deviations from global model
        for layer, weights in update.weights.items():
            if layer not in self.global_model:
                continue

            deviation = np.linalg.norm(weights - self.global_model[layer])
            avg_norm = np.linalg.norm(self.global_model[layer])

            if deviation > 10 * avg_norm:  # Threshold can be tuned
                return True

        return False

    def _filter_byzantine_updates(self, updates: List[ModelUpdate]) -> List[ModelUpdate]:
        """Filter out Byzantine updates using statistical methods."""
        if len(updates) < 3:
            return updates

        # Use median-based filtering
        filtered = []
        for i, update in enumerate(updates):
            distances = []
            for j, other in enumerate(updates):
                if i != j:
                    dist = 0
                    for layer in update.weights.keys():
                        if layer in other.weights:
                            dist += np.linalg.norm(
                                update.weights[layer] - other.weights[layer]
                            )
                    distances.append(dist)

            median_dist = np.median(distances)
            if median_dist < np.percentile(distances, 75) * 2:
                filtered.append(update)

        return filtered

    def _apply_differential_privacy(self, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply differential privacy to aggregated weights."""
        noisy_weights = {}

        for layer, w in weights.items():
            # Calculate sensitivity (simplified)
            sensitivity = np.max(np.abs(w)) / len(self.round_updates)

            # Add Gaussian noise
            noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / self.config.delta)) / self.config.privacy_budget
            noise = np.random.normal(0, noise_scale, w.shape)

            noisy_weights[layer] = w + noise

        return noisy_weights

    def _update_contribution_scores(self, updates: List[ModelUpdate]):
        """Update client contribution scores using Shapley values."""
        # Simplified Shapley value calculation
        for update in updates:
            client = self.clients[update.client_id]

            # Contribution based on data quantity and quality
            data_contribution = update.num_samples / sum(u.num_samples for u in updates)

            # Quality based on validation metrics
            quality = np.mean(list(update.metrics.values())) if update.metrics else 0.5

            # Update contribution score (exponential moving average)
            alpha = 0.3
            client.contribution_score = (
                alpha * (data_contribution * quality) +
                (1 - alpha) * client.contribution_score
            )

    def _save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint = {
            'round': self.current_round,
            'model': self.global_model,
            'clients': {cid: {
                'institution': meta.institution,
                'contribution_score': meta.contribution_score,
                'trust_score': meta.trust_score,
                'rounds_participated': meta.rounds_participated
            } for cid, meta in self.clients.items()},
            'timestamp': datetime.now().isoformat()
        }
        self.model_checkpoints[self.current_round] = checkpoint

    def _audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log audit event."""
        self.audit_log.append({
            'event': event_type,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })

    def get_audit_trail(self, start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get audit trail for specified time period."""
        filtered_log = self.audit_log

        if start_time:
            filtered_log = [
                e for e in filtered_log
                if datetime.fromisoformat(e['timestamp']) >= start_time
            ]

        if end_time:
            filtered_log = [
                e for e in filtered_log
                if datetime.fromisoformat(e['timestamp']) <= end_time
            ]

        return filtered_log


class FederatedClient:
    """Client in federated learning system."""

    def __init__(self, client_id: str, institution: str,
                 server_url: str, data_loader: Any = None):
        """Initialize federated client.

        Args:
            client_id: Unique client identifier
            institution: Institution name
            server_url: URL of federation server
            data_loader: Data loader for local training
        """
        self.client_id = client_id
        self.institution = institution
        self.server_url = server_url
        self.data_loader = data_loader
        self.local_model = None
        self.private_key = None
        self.public_key = None
        self.status = ClientStatus.IDLE

    def register(self, capabilities: Dict[str, Any]) -> bool:
        """Register with federation server.

        Args:
            capabilities: Client capabilities

        Returns:
            True if registration successful
        """
        # In production, this would make an API call to server
        # For now, we'll simulate the registration
        logger.info(f"Client {self.client_id} registering with server")
        self.status = ClientStatus.IDLE
        return True

    def train_round(self, global_weights: Dict[str, np.ndarray],
                   config: Dict[str, Any]) -> ModelUpdate:
        """Train local model for one round.

        Args:
            global_weights: Global model weights
            config: Training configuration

        Returns:
            Model update to send to server
        """
        self.status = ClientStatus.TRAINING
        start_time = time.time()

        # Update local model with global weights
        self.local_model = global_weights.copy()

        # Simulate local training
        num_epochs = config.get('num_epochs', 1)
        batch_size = config.get('batch_size', 32)

        # Train on local data (simplified)
        for epoch in range(num_epochs):
            # In practice, this would use the data_loader
            for layer in self.local_model.keys():
                # Simulate gradient update
                gradient = np.random.randn(*self.local_model[layer].shape) * 0.01
                self.local_model[layer] -= 0.01 * gradient

        # Calculate metrics
        metrics = {
            'loss': np.random.random() * 0.5,
            'accuracy': 0.8 + np.random.random() * 0.15
        }

        computation_time = time.time() - start_time

        # Create update
        update = ModelUpdate(
            client_id=self.client_id,
            round_number=config.get('round_number', 0),
            weights=self.local_model,
            metrics=metrics,
            num_samples=1000,  # Would come from data_loader
            computation_time=computation_time,
            timestamp=datetime.now()
        )

        self.status = ClientStatus.IDLE
        return update

    def validate_model(self, model_weights: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Validate model on local test data.

        Args:
            model_weights: Model weights to validate

        Returns:
            Validation metrics
        """
        self.status = ClientStatus.VALIDATING

        # Simulate validation
        metrics = {
            'val_loss': np.random.random() * 0.3,
            'val_accuracy': 0.85 + np.random.random() * 0.1,
            'val_f1': 0.8 + np.random.random() * 0.15
        }

        self.status = ClientStatus.IDLE
        return metrics