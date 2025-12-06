"""
Cross-Domain Analogy Finder

Finds deep structural similarities between domains.

Example:
- "Quantum tunneling" <-> "Psychological defense mechanisms"
- Both involve: entity passes through normally impenetrable barrier
"""

from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json


class CrossDomainAnalogyFinder:
    """
    Finds structural analogies between two domains.

    Uses semantic embeddings to find concepts that have similar
    abstract structure but different concrete implementations.
    """

    def __init__(self, domains_file: str = "database/domains.json"):
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.domains = self._load_domains(domains_file)

        # Pre-compute embeddings for all concepts
        self._precompute_embeddings()

    def _load_domains(self, filename: str) -> Dict:
        """Load domain database."""
        with open(filename, 'r') as f:
            return json.load(f)

    def _precompute_embeddings(self):
        """Encode all concepts as vectors for fast comparison."""
        for domain_name, domain_data in self.domains.items():
            for concept in domain_data.get('key_concepts', []):
                if 'embedding' not in concept:
                    concept['embedding'] = self.model.encode(
                        concept['description']
                    )

    def find_analogies(
        self,
        domain_a: str,
        domain_b: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Find structural analogies between two domains.

        Returns list of analogies with:
        - concept_a: Concept from domain A
        - concept_b: Concept from domain B
        - similarity_score: 0-1
        - analogy_type: 'structural', 'functional', 'causal'
        - explanation: Natural language explanation
        """

        concepts_a = self.domains[domain_a]['key_concepts']
        concepts_b = self.domains[domain_b]['key_concepts']

        # Get embeddings
        embeddings_a = np.array([c['embedding'] for c in concepts_a])
        embeddings_b = np.array([c['embedding'] for c in concepts_b])

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings_a, embeddings_b)

        analogies = []

        for i, concept_a in enumerate(concepts_a):
            for j, concept_b in enumerate(concepts_b):
                similarity = similarity_matrix[i, j]

                # Sweet spot: not too obvious (>0.8), not too random (<0.4)
                if 0.4 < similarity < 0.8:
                    analogy = {
                        'concept_a': concept_a['name'],
                        'concept_a_description': concept_a['description'],
                        'concept_b': concept_b['name'],
                        'concept_b_description': concept_b['description'],
                        'similarity_score': float(similarity),
                        'domain_a': domain_a,
                        'domain_b': domain_b,
                        'interestingness': self._calculate_interestingness(
                            similarity, domain_a, domain_b
                        )
                    }
                    analogies.append(analogy)

        # Sort by interestingness
        analogies.sort(key=lambda x: x['interestingness'], reverse=True)

        return analogies[:top_k]

    def _calculate_interestingness(
        self,
        similarity: float,
        domain_a: str,
        domain_b: str
    ) -> float:
        """
        Calculate how interesting an analogy is.

        Interesting = medium similarity × high domain distance

        We want:
        - Domains that are very different (physics vs poetry)
        - Concepts that are moderately similar (not obvious, not nonsense)
        """

        domain_distance = self._get_domain_distance(domain_a, domain_b)

        # Reward medium similarity (peak at 0.6)
        similarity_score = 1.0 - abs(similarity - 0.6) / 0.4

        # Interestingness = similarity sweetspot × domain distance
        interestingness = similarity_score * domain_distance

        return interestingness

    def _get_domain_distance(self, domain_a: str, domain_b: str) -> float:
        """
        Calculate how different two domains are.

        Returns 0-1:
        - 0 = same domain (boring)
        - 1 = maximally different (interesting)

        Uses domain categories and ontological distance.
        """

        category_a = self.domains[domain_a]['category']
        category_b = self.domains[domain_b]['category']

        # Same category = lower distance
        if category_a == category_b:
            return 0.3

        # Related categories (e.g., Physics & Chemistry)
        related_categories = {
            'Physics': ['Chemistry', 'Engineering'],
            'Biology': ['Medicine', 'Chemistry'],
            'Mathematics': ['Computer Science', 'Physics'],
            'Psychology': ['Sociology', 'Neuroscience'],
            'Art': ['Music', 'Literature'],
        }

        if category_b in related_categories.get(category_a, []):
            return 0.6

        # Unrelated = maximum distance
        return 1.0

    def generate_hypothesis_from_analogy(
        self,
        analogy: Dict
    ) -> str:
        """
        Generate testable hypothesis from analogy.

        Uses GPT-4 to articulate the deep connection.
        """

        prompt = f"""
        You've found a deep structural analogy between two concepts:

        Concept A: {analogy['concept_a']} from {analogy['domain_a']}
        Description: {analogy['concept_a_description']}

        Concept B: {analogy['concept_b']} from {analogy['domain_b']}
        Description: {analogy['concept_b_description']}

        They have a similarity score of {analogy['similarity_score']:.2f},
        suggesting they share abstract structure.

        Generate a testable hypothesis that applies insights from
        {analogy['domain_a']} to solve problems in {analogy['domain_b']}.

        Requirements:
        1. Be specific and testable
        2. Explain the analogy clearly
        3. Suggest an experiment or analysis
        4. Be creative but rigorous

        Format:
        **Hypothesis:** [One sentence]
        **Analogy:** [How the concepts map to each other]
        **Test:** [How to verify experimentally]
        **Impact:** [Why this matters]
        """

        # TODO: Call GPT-4
        # For now, return template

        return f"""
        **Hypothesis:** {analogy['concept_a']} dynamics in {analogy['domain_a']}
        can be applied to understand {analogy['concept_b']} in {analogy['domain_b']}.

        **Analogy:** Both involve [abstract pattern that connects them].

        **Test:** Compare predictions from {analogy['domain_a']} model with
        real-world {analogy['domain_b']} data.

        **Impact:** Could reveal hidden mechanisms in {analogy['domain_b']}.
        """


# Example usage
def example():
    """Test analogy finder."""

    # Mock domain database for testing
    mock_domains = {
        "Quantum Mechanics": {
            "category": "Physics",
            "key_concepts": [
                {
                    "name": "Superposition",
                    "description": "A system exists in multiple states simultaneously until observed"
                },
                {
                    "name": "Entanglement",
                    "description": "Two particles become correlated such that measuring one affects the other"
                }
            ]
        },
        "Social Media": {
            "category": "Technology",
            "key_concepts": [
                {
                    "name": "Filter Bubbles",
                    "description": "Users see different realities based on their viewing history"
                },
                {
                    "name": "Viral Spread",
                    "description": "Content spreads through network effects and replication"
                }
            ]
        }
    }

    # Save mock database
    with open('database/domains.json', 'w') as f:
        json.dump(mock_domains, f, indent=2)

    # Test analogy finder
    finder = CrossDomainAnalogyFinder('database/domains.json')

    analogies = finder.find_analogies(
        "Quantum Mechanics",
        "Social Media"
    )

    print(f"Found {len(analogies)} analogies:\n")

    for i, analogy in enumerate(analogies[:3], 1):
        print(f"{i}. {analogy['concept_a']} <-> {analogy['concept_b']}")
        print(f"   Similarity: {analogy['similarity_score']:.2f}")
        print(f"   Interestingness: {analogy['interestingness']:.2f}")
        print()

        # Generate hypothesis
        hypothesis = finder.generate_hypothesis_from_analogy(analogy)
        print(hypothesis)
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    example()
