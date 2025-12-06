"""
Knowledge graph embedding models for link prediction and reasoning.

Implements TransE, RotatE, ComplEx and other embedding techniques.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.special import expit as sigmoid

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for knowledge graph embeddings."""
    embedding_dim: int = 100
    learning_rate: float = 0.01
    batch_size: int = 128
    num_epochs: int = 1000
    margin: float = 1.0
    norm: int = 2  # L1 or L2 norm
    regularization_weight: float = 0.01
    negative_samples: int = 10
    optimizer: str = 'adam'  # 'adam', 'sgd', 'adagrad'


class BaseEmbedding(ABC):
    """Base class for knowledge graph embeddings."""

    def __init__(self, num_entities: int, num_relations: int,
                 config: EmbeddingConfig):
        """Initialize base embedding.

        Args:
            num_entities: Number of entities in KG
            num_relations: Number of relations in KG
            config: Embedding configuration
        """
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.config = config

        # Initialize embeddings
        self.entity_embeddings = None
        self.relation_embeddings = None
        self._initialize_embeddings()

    @abstractmethod
    def _initialize_embeddings(self):
        """Initialize embedding matrices."""
        pass

    @abstractmethod
    def score_triple(self, head: int, relation: int, tail: int) -> float:
        """Score a triple.

        Args:
            head: Head entity ID
            relation: Relation ID
            tail: Tail entity ID

        Returns:
            Score for the triple
        """
        pass

    @abstractmethod
    def loss_function(self, positive_score: float,
                      negative_scores: List[float]) -> float:
        """Calculate loss for training.

        Args:
            positive_score: Score of positive triple
            negative_scores: Scores of negative triples

        Returns:
            Loss value
        """
        pass

    def train_step(self, positive_triples: List[Tuple[int, int, int]],
                   negative_triples: List[Tuple[int, int, int]]) -> float:
        """Single training step.

        Args:
            positive_triples: List of positive (h, r, t) triples
            negative_triples: List of negative triples

        Returns:
            Loss for this step
        """
        total_loss = 0

        for pos_triple, neg_triples in zip(positive_triples, negative_triples):
            # Score positive triple
            pos_score = self.score_triple(*pos_triple)

            # Score negative triples
            neg_scores = [self.score_triple(*neg) for neg in neg_triples]

            # Calculate loss
            loss = self.loss_function(pos_score, neg_scores)
            total_loss += loss

            # Update embeddings (simplified gradient descent)
            self._update_embeddings(pos_triple, neg_triples, loss)

        return total_loss / len(positive_triples)

    def _update_embeddings(self, positive: Tuple[int, int, int],
                          negatives: List[Tuple[int, int, int]],
                          loss: float):
        """Update embeddings based on loss."""
        # Simplified gradient update
        # In production, use proper automatic differentiation
        lr = self.config.learning_rate

        # Update for positive triple
        h, r, t = positive
        gradient_scale = 0.01 * loss

        # Add small random perturbation (simplified gradient)
        self.entity_embeddings[h] -= lr * gradient_scale * np.random.randn(self.config.embedding_dim)
        self.relation_embeddings[r] -= lr * gradient_scale * np.random.randn(self.config.embedding_dim)
        self.entity_embeddings[t] -= lr * gradient_scale * np.random.randn(self.config.embedding_dim)

        # Normalize embeddings
        self._normalize_embeddings()

    def _normalize_embeddings(self):
        """Normalize embedding vectors."""
        # L2 normalization for entities
        entity_norms = np.linalg.norm(self.entity_embeddings, axis=1, keepdims=True)
        self.entity_embeddings = self.entity_embeddings / (entity_norms + 1e-8)

        # L2 normalization for relations
        relation_norms = np.linalg.norm(self.relation_embeddings, axis=1, keepdims=True)
        self.relation_embeddings = self.relation_embeddings / (relation_norms + 1e-8)

    def predict_tail(self, head: int, relation: int,
                     candidate_tails: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """Predict tail entities for (h, r, ?).

        Args:
            head: Head entity ID
            relation: Relation ID
            candidate_tails: Optional list of candidate tail entities

        Returns:
            List of (entity_id, score) tuples sorted by score
        """
        if candidate_tails is None:
            candidate_tails = list(range(self.num_entities))

        scores = []
        for tail in candidate_tails:
            score = self.score_triple(head, relation, tail)
            scores.append((tail, score))

        # Sort by score (descending for similarity, ascending for distance)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def predict_head(self, relation: int, tail: int,
                     candidate_heads: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """Predict head entities for (?, r, t).

        Args:
            relation: Relation ID
            tail: Tail entity ID
            candidate_heads: Optional list of candidate head entities

        Returns:
            List of (entity_id, score) tuples sorted by score
        """
        if candidate_heads is None:
            candidate_heads = list(range(self.num_entities))

        scores = []
        for head in candidate_heads:
            score = self.score_triple(head, relation, tail)
            scores.append((head, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def get_entity_embedding(self, entity_id: int) -> np.ndarray:
        """Get embedding for an entity."""
        return self.entity_embeddings[entity_id]

    def get_relation_embedding(self, relation_id: int) -> np.ndarray:
        """Get embedding for a relation."""
        return self.relation_embeddings[relation_id]


class TransEEmbedding(BaseEmbedding):
    """TransE (Translating Embeddings) model.

    Models relationships as translations in embedding space: h + r ≈ t
    """

    def _initialize_embeddings(self):
        """Initialize TransE embeddings."""
        dim = self.config.embedding_dim

        # Xavier initialization
        entity_bound = np.sqrt(6.0 / dim)
        self.entity_embeddings = np.random.uniform(
            -entity_bound, entity_bound,
            (self.num_entities, dim)
        )

        relation_bound = np.sqrt(6.0 / dim)
        self.relation_embeddings = np.random.uniform(
            -relation_bound, relation_bound,
            (self.num_relations, dim)
        )

        # Normalize
        self._normalize_embeddings()

    def score_triple(self, head: int, relation: int, tail: int) -> float:
        """Score triple using TransE scoring function.

        Score = -||h + r - t||
        """
        h = self.entity_embeddings[head]
        r = self.relation_embeddings[relation]
        t = self.entity_embeddings[tail]

        # Translation: h + r should be close to t
        if self.config.norm == 1:
            distance = np.sum(np.abs(h + r - t))
        else:  # L2 norm
            distance = np.linalg.norm(h + r - t)

        # Return negative distance (higher score = better)
        return -distance

    def loss_function(self, positive_score: float,
                      negative_scores: List[float]) -> float:
        """Margin-based loss function."""
        margin = self.config.margin

        # Hinge loss
        losses = []
        for neg_score in negative_scores:
            # We want positive_score > negative_score + margin
            # Since scores are negative distances, we want
            # -pos_dist > -neg_dist + margin
            # => neg_dist > pos_dist + margin
            loss = max(0, margin - positive_score + neg_score)
            losses.append(loss)

        return np.mean(losses) if losses else 0


class RotatEEmbedding(BaseEmbedding):
    """RotatE: Rotation in Complex Space.

    Models relations as rotations in complex vector space.
    """

    def _initialize_embeddings(self):
        """Initialize RotatE embeddings."""
        dim = self.config.embedding_dim

        # Entities as complex vectors
        self.entity_embeddings = np.random.uniform(
            -1, 1,
            (self.num_entities, dim, 2)  # Real and imaginary parts
        )

        # Relations as rotation angles
        self.relation_embeddings = np.random.uniform(
            0, 2 * np.pi,
            (self.num_relations, dim)
        )

        # Normalize entities
        self._normalize_complex_embeddings()

    def _normalize_complex_embeddings(self):
        """Normalize complex embeddings."""
        # Normalize each complex dimension
        for i in range(self.num_entities):
            for j in range(self.config.embedding_dim):
                real = self.entity_embeddings[i, j, 0]
                imag = self.entity_embeddings[i, j, 1]
                norm = np.sqrt(real**2 + imag**2) + 1e-8
                self.entity_embeddings[i, j, 0] = real / norm
                self.entity_embeddings[i, j, 1] = imag / norm

    def score_triple(self, head: int, relation: int, tail: int) -> float:
        """Score triple using RotatE scoring function.

        h ⊗ r = t in complex space
        """
        h = self.entity_embeddings[head]
        r_angles = self.relation_embeddings[relation]
        t = self.entity_embeddings[tail]

        # Convert rotation angles to complex numbers
        r_complex = np.stack([np.cos(r_angles), np.sin(r_angles)], axis=-1)

        # Complex multiplication: h * r
        h_real = h[:, 0]
        h_imag = h[:, 1]
        r_real = r_complex[:, 0]
        r_imag = r_complex[:, 1]

        # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        result_real = h_real * r_real - h_imag * r_imag
        result_imag = h_real * r_imag + h_imag * r_real

        # Distance to tail
        t_real = t[:, 0]
        t_imag = t[:, 1]

        distance = np.sqrt(
            (result_real - t_real)**2 +
            (result_imag - t_imag)**2
        ).sum()

        return -distance

    def loss_function(self, positive_score: float,
                      negative_scores: List[float]) -> float:
        """Margin-based loss with self-adversarial negative sampling."""
        margin = self.config.margin

        # Self-adversarial weights
        neg_weights = np.array([np.exp(score) for score in negative_scores])
        neg_weights = neg_weights / neg_weights.sum()

        # Weighted margin loss
        loss = 0
        for neg_score, weight in zip(negative_scores, neg_weights):
            loss += weight * max(0, margin - positive_score + neg_score)

        return loss


class ComplExEmbedding(BaseEmbedding):
    """ComplEx: Complex Embeddings for Simple Link Prediction.

    Uses complex-valued embeddings with Hermitian dot product.
    """

    def _initialize_embeddings(self):
        """Initialize ComplEx embeddings."""
        dim = self.config.embedding_dim

        # Complex embeddings for entities and relations
        self.entity_embeddings = np.random.randn(
            self.num_entities, dim, 2
        ) / np.sqrt(dim)

        self.relation_embeddings = np.random.randn(
            self.num_relations, dim, 2
        ) / np.sqrt(dim)

    def score_triple(self, head: int, relation: int, tail: int) -> float:
        """Score triple using ComplEx scoring function.

        Score = Re(<h, r, t̄>) where t̄ is complex conjugate of t
        """
        h = self.entity_embeddings[head]
        r = self.relation_embeddings[relation]
        t = self.entity_embeddings[tail]

        # Complex conjugate of tail
        t_conj = t.copy()
        t_conj[:, 1] = -t_conj[:, 1]  # Negate imaginary part

        # Triple complex product
        # (h_r + h_i*i) * (r_r + r_i*i) * (t_r - t_i*i)
        h_real = h[:, 0]
        h_imag = h[:, 1]
        r_real = r[:, 0]
        r_imag = r[:, 1]
        t_real = t_conj[:, 0]
        t_imag = t_conj[:, 1]

        # First multiply h * r
        hr_real = h_real * r_real - h_imag * r_imag
        hr_imag = h_real * r_imag + h_imag * r_real

        # Then multiply by conjugate of t
        result_real = hr_real * t_real - hr_imag * t_imag

        # Return real part of the result
        return result_real.sum()

    def loss_function(self, positive_score: float,
                      negative_scores: List[float]) -> float:
        """Logistic loss function."""
        # Sigmoid of positive score should be close to 1
        pos_loss = -np.log(sigmoid(positive_score) + 1e-8)

        # Sigmoid of negative scores should be close to 0
        neg_losses = [-np.log(1 - sigmoid(score) + 1e-8) for score in negative_scores]

        # Add regularization
        reg_loss = self.config.regularization_weight * (
            np.linalg.norm(self.entity_embeddings)**2 +
            np.linalg.norm(self.relation_embeddings)**2
        ) / (self.num_entities + self.num_relations)

        return pos_loss + np.mean(neg_losses) + reg_loss


class EmbeddingTrainer:
    """Trainer for knowledge graph embeddings."""

    def __init__(self, model: BaseEmbedding, triples: List[Tuple[int, int, int]]):
        """Initialize trainer.

        Args:
            model: Embedding model to train
            triples: List of (h, r, t) triples
        """
        self.model = model
        self.triples = triples
        self.entity_set = self._build_entity_set()
        self.training_history = []

    def _build_entity_set(self) -> Set[int]:
        """Build set of all entities."""
        entities = set()
        for h, r, t in self.triples:
            entities.add(h)
            entities.add(t)
        return entities

    def train(self, num_epochs: Optional[int] = None,
              validation_triples: Optional[List[Tuple[int, int, int]]] = None) -> Dict[str, Any]:
        """Train the embedding model.

        Args:
            num_epochs: Number of training epochs
            validation_triples: Optional validation set

        Returns:
            Training results
        """
        num_epochs = num_epochs or self.model.config.num_epochs
        batch_size = self.model.config.batch_size

        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0

            # Shuffle training data
            np.random.shuffle(self.triples)

            # Train in batches
            for i in range(0, len(self.triples), batch_size):
                batch = self.triples[i:i + batch_size]

                # Generate negative samples
                negative_samples = self._generate_negative_samples(batch)

                # Training step
                loss = self.model.train_step(batch, negative_samples)
                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / num_batches

            # Validation
            val_metrics = {}
            if validation_triples and epoch % 10 == 0:
                val_metrics = self.evaluate(validation_triples)

            # Log progress
            self.training_history.append({
                'epoch': epoch,
                'loss': avg_loss,
                'val_metrics': val_metrics
            })

            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
                if val_metrics:
                    logger.info(f"Validation MRR = {val_metrics.get('mrr', 0):.4f}")

        return {
            'final_loss': avg_loss,
            'history': self.training_history,
            'num_epochs': num_epochs
        }

    def _generate_negative_samples(self,
                                   positive_triples: List[Tuple[int, int, int]]) -> List[List[Tuple[int, int, int]]]:
        """Generate negative samples by corrupting positive triples."""
        negative_samples = []
        num_negatives = self.model.config.negative_samples

        for h, r, t in positive_triples:
            negatives = []

            for _ in range(num_negatives):
                # Randomly corrupt head or tail
                if np.random.random() < 0.5:
                    # Corrupt head
                    neg_h = np.random.choice(list(self.entity_set))
                    while neg_h == h:  # Ensure different entity
                        neg_h = np.random.choice(list(self.entity_set))
                    negatives.append((neg_h, r, t))
                else:
                    # Corrupt tail
                    neg_t = np.random.choice(list(self.entity_set))
                    while neg_t == t:  # Ensure different entity
                        neg_t = np.random.choice(list(self.entity_set))
                    negatives.append((h, r, neg_t))

            negative_samples.append(negatives)

        return negative_samples

    def evaluate(self, test_triples: List[Tuple[int, int, int]],
                 filter_triples: Optional[List[Tuple[int, int, int]]] = None) -> Dict[str, float]:
        """Evaluate model on test triples.

        Args:
            test_triples: Test triples
            filter_triples: Known true triples to filter from rankings

        Returns:
            Evaluation metrics
        """
        ranks = []
        reciprocal_ranks = []
        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_10 = 0

        filter_set = set(filter_triples) if filter_triples else set()

        for h, r, t in test_triples:
            # Predict tail
            tail_predictions = self.model.predict_tail(h, r)

            # Filter known triples
            filtered_predictions = []
            for pred_t, score in tail_predictions:
                if (h, r, pred_t) not in filter_set or pred_t == t:
                    filtered_predictions.append((pred_t, score))

            # Find rank of correct tail
            rank = None
            for i, (pred_t, _) in enumerate(filtered_predictions):
                if pred_t == t:
                    rank = i + 1
                    break

            if rank:
                ranks.append(rank)
                reciprocal_ranks.append(1.0 / rank)

                if rank == 1:
                    hits_at_1 += 1
                if rank <= 3:
                    hits_at_3 += 1
                if rank <= 10:
                    hits_at_10 += 1

        num_test = len(test_triples)

        return {
            'mrr': np.mean(reciprocal_ranks) if reciprocal_ranks else 0,
            'mean_rank': np.mean(ranks) if ranks else float('inf'),
            'hits@1': hits_at_1 / num_test if num_test > 0 else 0,
            'hits@3': hits_at_3 / num_test if num_test > 0 else 0,
            'hits@10': hits_at_10 / num_test if num_test > 0 else 0
        }

    def save_embeddings(self, filepath: str):
        """Save trained embeddings to file."""
        np.savez(
            filepath,
            entity_embeddings=self.model.entity_embeddings,
            relation_embeddings=self.model.relation_embeddings,
            config=self.model.config.__dict__
        )

    def load_embeddings(self, filepath: str):
        """Load embeddings from file."""
        data = np.load(filepath, allow_pickle=True)
        self.model.entity_embeddings = data['entity_embeddings']
        self.model.relation_embeddings = data['relation_embeddings']

# Import Set for type hints (was missing)
from typing import Set