"""
Federated Learning Research Platform for Multi-Institution Collaboration

A comprehensive platform enabling secure, privacy-preserving federated learning
across research institutions with support for horizontal/vertical FL, differential
privacy, secure aggregation, and Byzantine robustness.
"""

from .core import (
    FederatedServer,
    FederatedClient,
    SecureAggregator,
    FederationProtocol,
)
from .privacy import (
    DifferentialPrivacy,
    SecureMultipartyComputation,
    HomomorphicEncryption,
    PrivacyAccountant,
)
from .aggregation import (
    FedAvgAggregator,
    ByzantineRobustAggregator,
    PersonalizedAggregator,
    AdaptiveAggregator,
)
from .strategies import (
    ClientSelector,
    ModelCompressor,
    CommunicationOptimizer,
    ContributionTracker,
)

__version__ = "1.0.0"
__all__ = [
    "FederatedServer",
    "FederatedClient",
    "SecureAggregator",
    "FederationProtocol",
    "DifferentialPrivacy",
    "SecureMultipartyComputation",
    "HomomorphicEncryption",
    "PrivacyAccountant",
    "FedAvgAggregator",
    "ByzantineRobustAggregator",
    "PersonalizedAggregator",
    "AdaptiveAggregator",
    "ClientSelector",
    "ModelCompressor",
    "CommunicationOptimizer",
    "ContributionTracker",
]