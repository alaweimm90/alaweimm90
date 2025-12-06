"""
Privacy-preserving mechanisms for federated learning.

Implements differential privacy, secure multiparty computation,
homomorphic encryption, and privacy accounting.
"""

import hashlib
import logging
import secrets
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


@dataclass
class PrivacyParameters:
    """Privacy parameters for federated learning."""
    epsilon: float  # Privacy budget
    delta: float    # Privacy parameter
    noise_multiplier: float
    max_grad_norm: float
    num_microbatches: int
    mechanism: str  # 'gaussian', 'laplace', 'exponential'


class DifferentialPrivacy:
    """Differential privacy mechanisms for federated learning."""

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5,
                 max_grad_norm: float = 1.0):
        """Initialize differential privacy.

        Args:
            epsilon: Privacy budget
            delta: Privacy parameter
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.privacy_spent = 0.0
        self.noise_history = []

    def add_gaussian_noise(self, tensor: np.ndarray,
                          sensitivity: float,
                          sampling_rate: float = 1.0) -> np.ndarray:
        """Add Gaussian noise for differential privacy.

        Args:
            tensor: Input tensor
            sensitivity: Sensitivity of the query
            sampling_rate: Sampling rate for subsampling

        Returns:
            Noisy tensor
        """
        # Calculate noise scale using Gaussian mechanism
        noise_scale = self._calculate_gaussian_noise_scale(
            sensitivity, sampling_rate
        )

        # Add noise
        noise = np.random.normal(0, noise_scale, tensor.shape)
        noisy_tensor = tensor + noise

        # Track privacy spending
        self._update_privacy_spent(noise_scale, sampling_rate)

        return noisy_tensor

    def add_laplace_noise(self, tensor: np.ndarray,
                         sensitivity: float) -> np.ndarray:
        """Add Laplace noise for differential privacy.

        Args:
            tensor: Input tensor
            sensitivity: Sensitivity of the query

        Returns:
            Noisy tensor
        """
        noise_scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, noise_scale, tensor.shape)
        return tensor + noise

    def clip_gradients(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Clip gradients to bound sensitivity.

        Args:
            gradients: Dictionary of gradients

        Returns:
            Clipped gradients
        """
        # Calculate total norm
        total_norm = 0
        for grad in gradients.values():
            total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)

        # Clip if necessary
        if total_norm > self.max_grad_norm:
            scale = self.max_grad_norm / total_norm
            clipped = {}
            for key, grad in gradients.items():
                clipped[key] = grad * scale
            return clipped

        return gradients

    def _calculate_gaussian_noise_scale(self, sensitivity: float,
                                       sampling_rate: float) -> float:
        """Calculate noise scale for Gaussian mechanism."""
        if sampling_rate == 0:
            return 0

        # Using the analytical Gaussian mechanism
        c = np.sqrt(2 * np.log(1.25 / self.delta))
        noise_scale = c * sensitivity / (self.epsilon * sampling_rate)

        return noise_scale

    def _update_privacy_spent(self, noise_scale: float,
                             sampling_rate: float):
        """Update privacy spent using RDP accounting."""
        # Simplified privacy accounting
        # In production, use tools like Opacus or TensorFlow Privacy
        if noise_scale > 0:
            spent = sampling_rate * (1 / noise_scale)
            self.privacy_spent += spent
            self.noise_history.append({
                'noise_scale': noise_scale,
                'sampling_rate': sampling_rate,
                'spent': spent
            })

    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get total privacy spent.

        Returns:
            Tuple of (epsilon_spent, delta_spent)
        """
        return self.privacy_spent, self.delta

    def compose_privacy(self, mechanisms: List['DifferentialPrivacy']) -> 'DifferentialPrivacy':
        """Compose multiple DP mechanisms.

        Args:
            mechanisms: List of DP mechanisms to compose

        Returns:
            Composed DP mechanism
        """
        total_epsilon = sum(m.epsilon for m in mechanisms)
        total_delta = sum(m.delta for m in mechanisms)

        return DifferentialPrivacy(
            epsilon=total_epsilon,
            delta=total_delta,
            max_grad_norm=self.max_grad_norm
        )


class SecureMultipartyComputation:
    """Secure multiparty computation for privacy-preserving aggregation."""

    def __init__(self, num_parties: int, threshold: int):
        """Initialize secure MPC.

        Args:
            num_parties: Number of parties
            threshold: Threshold for reconstruction
        """
        self.num_parties = num_parties
        self.threshold = threshold
        self.shares = {}

    def secret_share(self, secret: np.ndarray,
                    num_shares: int) -> List[np.ndarray]:
        """Create secret shares using Shamir's secret sharing.

        Args:
            secret: Secret value to share
            num_shares: Number of shares to create

        Returns:
            List of secret shares
        """
        if num_shares < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares")

        shares = []
        shape = secret.shape

        # Flatten for processing
        flat_secret = secret.flatten()

        for i in range(num_shares):
            if i < num_shares - 1:
                # Random shares
                share = np.random.randn(flat_secret.shape[0])
            else:
                # Last share ensures sum equals secret
                share = flat_secret - sum(shares)

            shares.append(share.reshape(shape))

        return shares

    def reconstruct_secret(self, shares: List[np.ndarray]) -> np.ndarray:
        """Reconstruct secret from shares.

        Args:
            shares: List of secret shares

        Returns:
            Reconstructed secret
        """
        if len(shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares for reconstruction")

        # Simple additive reconstruction
        return np.sum(shares, axis=0)

    def secure_aggregation(self, values: List[np.ndarray],
                          masks: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """Perform secure aggregation.

        Args:
            values: Values from different parties
            masks: Optional masks for additional privacy

        Returns:
            Aggregated result
        """
        if len(values) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} values")

        # Apply masks if provided
        if masks:
            masked_values = [v + m for v, m in zip(values, masks)]
        else:
            masked_values = values

        # Aggregate
        aggregated = np.mean(masked_values, axis=0)

        # Remove masks if used
        if masks:
            mask_sum = np.sum(masks, axis=0) / len(masks)
            aggregated -= mask_sum

        return aggregated

    def generate_beaver_triples(self, num_triples: int) -> List[Tuple[int, int, int]]:
        """Generate Beaver triples for secure multiplication.

        Args:
            num_triples: Number of triples to generate

        Returns:
            List of Beaver triples (a, b, c) where c = a * b
        """
        triples = []
        for _ in range(num_triples):
            a = secrets.randbits(256)
            b = secrets.randbits(256)
            c = a * b
            triples.append((a, b, c))
        return triples

    def secure_comparison(self, x_shares: List[int],
                         y_shares: List[int]) -> bool:
        """Secure comparison of shared values.

        Args:
            x_shares: Shares of value x
            y_shares: Shares of value y

        Returns:
            True if x > y
        """
        # Simplified secure comparison
        # In production, use protocols like Yao's garbled circuits
        x = sum(x_shares)
        y = sum(y_shares)
        return x > y


class HomomorphicEncryption:
    """Homomorphic encryption for computation on encrypted data."""

    def __init__(self, key_size: int = 2048):
        """Initialize homomorphic encryption.

        Args:
            key_size: Size of encryption key
        """
        self.key_size = key_size
        self.public_key = None
        self.private_key = None
        self._generate_keys()

    def _generate_keys(self):
        """Generate public and private keys."""
        # Simplified key generation
        # In production, use libraries like SEAL or TenSEAL
        self.private_key = secrets.randbits(self.key_size)
        self.public_key = pow(2, self.private_key, 2**self.key_size - 1)

    def encrypt(self, plaintext: np.ndarray) -> np.ndarray:
        """Encrypt plaintext.

        Args:
            plaintext: Data to encrypt

        Returns:
            Encrypted data
        """
        # Simplified encryption (not cryptographically secure)
        # In production, use proper HE libraries
        flat = plaintext.flatten()
        encrypted = []

        for value in flat:
            # Add noise for semantic security
            noise = secrets.randbits(64)
            encrypted_value = (int(value * 1e6) + noise * self.public_key) % (2**self.key_size)
            encrypted.append(encrypted_value)

        return np.array(encrypted).reshape(plaintext.shape)

    def decrypt(self, ciphertext: np.ndarray) -> np.ndarray:
        """Decrypt ciphertext.

        Args:
            ciphertext: Encrypted data

        Returns:
            Decrypted data
        """
        flat = ciphertext.flatten()
        decrypted = []

        for value in flat:
            # Remove noise using private key
            decrypted_value = (value % self.private_key) / 1e6
            decrypted.append(decrypted_value)

        return np.array(decrypted).reshape(ciphertext.shape)

    def add_encrypted(self, ciphertext1: np.ndarray,
                     ciphertext2: np.ndarray) -> np.ndarray:
        """Add two encrypted values.

        Args:
            ciphertext1: First encrypted value
            ciphertext2: Second encrypted value

        Returns:
            Encrypted sum
        """
        return (ciphertext1 + ciphertext2) % (2**self.key_size)

    def multiply_encrypted_scalar(self, ciphertext: np.ndarray,
                                 scalar: float) -> np.ndarray:
        """Multiply encrypted value by plaintext scalar.

        Args:
            ciphertext: Encrypted value
            scalar: Plaintext scalar

        Returns:
            Encrypted product
        """
        return (ciphertext * int(scalar * 1e6)) % (2**self.key_size)


class PrivacyAccountant:
    """Track and manage privacy budget across federated learning rounds."""

    def __init__(self, total_epsilon: float, total_delta: float,
                 num_rounds: int):
        """Initialize privacy accountant.

        Args:
            total_epsilon: Total privacy budget
            total_delta: Total delta parameter
            num_rounds: Number of training rounds
        """
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.num_rounds = num_rounds
        self.epsilon_per_round = total_epsilon / num_rounds
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.round_history = []

    def allocate_privacy_budget(self, round_number: int) -> PrivacyParameters:
        """Allocate privacy budget for a round.

        Args:
            round_number: Current round number

        Returns:
            Privacy parameters for the round
        """
        remaining_rounds = self.num_rounds - round_number
        if remaining_rounds <= 0:
            raise ValueError("No remaining rounds")

        remaining_epsilon = self.total_epsilon - self.spent_epsilon

        # Adaptive allocation based on remaining budget
        round_epsilon = min(
            self.epsilon_per_round * 1.2,  # Allow slight overspend early
            remaining_epsilon / remaining_rounds
        )

        round_delta = self.total_delta / self.num_rounds

        # Calculate noise multiplier for target epsilon
        noise_multiplier = np.sqrt(2 * np.log(1.25 / round_delta)) / round_epsilon

        params = PrivacyParameters(
            epsilon=round_epsilon,
            delta=round_delta,
            noise_multiplier=noise_multiplier,
            max_grad_norm=1.0,
            num_microbatches=32,
            mechanism='gaussian'
        )

        return params

    def log_privacy_spending(self, round_number: int,
                            epsilon_spent: float,
                            delta_spent: float,
                            metrics: Dict[str, float]):
        """Log privacy spending for a round.

        Args:
            round_number: Round number
            epsilon_spent: Epsilon spent in round
            delta_spent: Delta spent in round
            metrics: Additional metrics
        """
        self.spent_epsilon += epsilon_spent
        self.spent_delta += delta_spent

        self.round_history.append({
            'round': round_number,
            'epsilon_spent': epsilon_spent,
            'delta_spent': delta_spent,
            'cumulative_epsilon': self.spent_epsilon,
            'cumulative_delta': self.spent_delta,
            'metrics': metrics
        })

    def check_privacy_budget(self) -> bool:
        """Check if privacy budget is exhausted.

        Returns:
            True if budget remains
        """
        return self.spent_epsilon < self.total_epsilon

    def get_privacy_guarantee(self) -> Dict[str, float]:
        """Get current privacy guarantee.

        Returns:
            Dictionary with privacy parameters
        """
        return {
            'epsilon_spent': self.spent_epsilon,
            'delta_spent': self.spent_delta,
            'epsilon_remaining': self.total_epsilon - self.spent_epsilon,
            'delta_remaining': self.total_delta - self.spent_delta,
            'rounds_completed': len(self.round_history),
            'privacy_amplification': self._calculate_amplification()
        }

    def _calculate_amplification(self) -> float:
        """Calculate privacy amplification from subsampling."""
        if not self.round_history:
            return 1.0

        # Simplified amplification calculation
        # In practice, use tighter bounds
        avg_sampling_rate = 0.1  # Assumed average
        amplification = np.sqrt(avg_sampling_rate)
        return amplification

    def compose_mechanisms(self, mechanisms: List[PrivacyParameters]) -> PrivacyParameters:
        """Compose multiple privacy mechanisms.

        Args:
            mechanisms: List of privacy mechanisms

        Returns:
            Composed privacy parameters
        """
        # Advanced composition theorem
        total_epsilon = np.sqrt(sum(m.epsilon**2 for m in mechanisms))
        total_delta = sum(m.delta for m in mechanisms)

        # Use the most conservative noise multiplier
        max_noise = max(m.noise_multiplier for m in mechanisms)

        return PrivacyParameters(
            epsilon=total_epsilon,
            delta=total_delta,
            noise_multiplier=max_noise,
            max_grad_norm=1.0,
            num_microbatches=32,
            mechanism='gaussian'
        )

    def get_renyi_privacy(self, alpha: float) -> float:
        """Calculate Rényi differential privacy.

        Args:
            alpha: Rényi parameter

        Returns:
            Rényi privacy parameter
        """
        if alpha == 1:
            return self.spent_epsilon

        # Simplified RDP calculation
        rdp = self.spent_epsilon * np.sqrt(alpha / 2)
        return rdp