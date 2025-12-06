"""
Similarity matcher - Finds similar past failures
"""

from typing import List, Tuple
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from hall_of_failures.models import Failure


class SimilarityMatcher:
    """
    Find similar failures using multiple similarity metrics

    Uses:
    - TF-IDF cosine similarity for semantic matching
    - Exact hash matching
    - Pattern matching
    """

    def __init__(self):
        """Initialize similarity matcher"""
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.corpus_failures = []
        self.corpus_vectors = None

    def build_index(self, failures: List[Failure]):
        """
        Build similarity index from corpus

        Args:
            failures: List of failures to index
        """
        if not failures:
            return

        self.corpus_failures = failures

        # Build TF-IDF vectors
        texts = [
            f"{f.hypothesis} {f.description} {' '.join(f.root_causes)}"
            for f in failures
        ]

        try:
            self.corpus_vectors = self.vectorizer.fit_transform(texts)
        except Exception:
            # Fallback if TF-IDF fails
            self.corpus_vectors = None

    def find_similar(
        self,
        query_failure: Failure,
        corpus: List[Failure],
        top_k: int = 5,
        min_similarity: float = 0.3
    ) -> List[Tuple[Failure, float]]:
        """
        Find similar failures

        Args:
            query_failure: Failure to match against
            corpus: List of failures to search
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of (failure, similarity_score) tuples
        """
        if not corpus:
            return []

        # 1. Exact hash matching
        exact_matches = [
            (f, 1.0) for f in corpus
            if f.similarity_hash == query_failure.similarity_hash
            and f.id != query_failure.id
        ]

        if exact_matches:
            return exact_matches[:top_k]

        # 2. TF-IDF semantic similarity
        semantic_matches = self._semantic_similarity(query_failure, corpus)

        # 3. Pattern matching
        pattern_matches = self._pattern_matching(query_failure, corpus)

        # 4. Context similarity
        context_matches = self._context_similarity(query_failure, corpus)

        # Combine scores
        combined_scores = {}
        for f in corpus:
            if f.id == query_failure.id:
                continue

            score = 0.0
            # Weighted combination
            score += 0.5 * semantic_matches.get(f.id, 0.0)
            score += 0.3 * pattern_matches.get(f.id, 0.0)
            score += 0.2 * context_matches.get(f.id, 0.0)

            combined_scores[f.id] = score

        # Filter and sort
        results = [
            (f, combined_scores[f.id])
            for f in corpus
            if f.id in combined_scores and combined_scores[f.id] >= min_similarity
        ]

        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def _semantic_similarity(
        self,
        query: Failure,
        corpus: List[Failure]
    ) -> dict:
        """TF-IDF cosine similarity"""
        scores = {}

        # Build corpus if needed
        if self.corpus_vectors is None or len(self.corpus_failures) != len(corpus):
            self.build_index(corpus)

        if self.corpus_vectors is None:
            return scores

        try:
            # Vectorize query
            query_text = f"{query.hypothesis} {query.description} {' '.join(query.root_causes)}"
            query_vector = self.vectorizer.transform([query_text])

            # Compute similarities
            similarities = cosine_similarity(query_vector, self.corpus_vectors)[0]

            for i, failure in enumerate(corpus):
                scores[failure.id] = float(similarities[i])
        except Exception:
            pass

        return scores

    def _pattern_matching(self, query: Failure, corpus: List[Failure]) -> dict:
        """Keyword and pattern matching"""
        scores = {}

        query_text = f"{query.hypothesis} {query.description}".lower()
        query_words = set(re.findall(r'\w+', query_text))

        for failure in corpus:
            failure_text = f"{failure.hypothesis} {failure.description}".lower()
            failure_words = set(re.findall(r'\w+', failure_text))

            # Jaccard similarity
            intersection = len(query_words & failure_words)
            union = len(query_words | failure_words)

            scores[failure.id] = intersection / union if union > 0 else 0.0

        return scores

    def _context_similarity(self, query: Failure, corpus: List[Failure]) -> dict:
        """Context field similarity"""
        scores = {}

        for failure in corpus:
            score = 0.0

            # Same failure type
            if query.failure_type == failure.failure_type:
                score += 0.5

            # Same severity
            if query.severity == failure.severity:
                score += 0.2

            # Overlapping root causes
            if query.root_causes and failure.root_causes:
                overlap = len(set(query.root_causes) & set(failure.root_causes))
                score += 0.3 * (overlap / len(query.root_causes))

            scores[failure.id] = min(1.0, score)

        return scores
