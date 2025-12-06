"""
Domain Classification and Quality Assessment for TalAI

Machine learning-based classification for research domains, quality levels,
and validation categories.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from pathlib import Path
import re

logger = logging.getLogger(__name__)

class DomainClassifier:
    """
    Advanced domain classification and categorization system.

    Features:
    - Multi-class domain classification
    - Hierarchical domain taxonomy
    - Evidence quality classification
    - Research type identification
    - Confidence scoring
    - Active learning support
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize domain classifier."""
        self.model_path = Path(model_path or '/tmp/talai_classifiers')
        self.model_path.mkdir(parents=True, exist_ok=True)

        self.vectorizer = None
        self.domain_classifier = None
        self.quality_classifier = None
        self.label_encoder = LabelEncoder()

        # Define domain taxonomy
        self.domain_hierarchy = {
            'natural_sciences': ['physics', 'chemistry', 'biology', 'astronomy', 'geology'],
            'formal_sciences': ['mathematics', 'logic', 'computer_science', 'statistics'],
            'social_sciences': ['psychology', 'sociology', 'economics', 'political_science'],
            'applied_sciences': ['engineering', 'medicine', 'agriculture', 'architecture'],
            'humanities': ['philosophy', 'history', 'literature', 'linguistics', 'arts']
        }

        # Research types
        self.research_types = [
            'experimental', 'observational', 'theoretical', 'computational',
            'review', 'meta-analysis', 'case_study', 'survey'
        ]

        self._initialize_models()

    def _initialize_models(self):
        """Initialize classification models."""
        try:
            # Initialize vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 3),
                min_df=2,
                stop_words='english'
            )

            # Initialize classifiers
            self.domain_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

            self.quality_classifier = SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )

            # Try to load pre-trained models
            self._load_models()

        except Exception as e:
            logger.error(f"Error initializing models: {e}")

    async def classify_domain(self,
                            text: str,
                            return_probabilities: bool = True) -> Dict[str, Any]:
        """
        Classify text into research domain.

        Args:
            text: Input text to classify
            return_probabilities: Whether to return probability scores

        Returns:
            Domain classification results
        """
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)

            # If no trained model, use rule-based classification
            if self.domain_classifier is None or self.vectorizer is None:
                return await self._rule_based_classification(processed_text)

            # Vectorize text
            features = self.vectorizer.transform([processed_text])

            # Predict domain
            prediction = self.domain_classifier.predict(features)[0]
            confidence = 0.0

            if return_probabilities:
                probabilities = self.domain_classifier.predict_proba(features)[0]
                confidence = float(np.max(probabilities))

                # Get top 3 predictions
                top_indices = np.argsort(probabilities)[::-1][:3]
                top_domains = []

                for idx in top_indices:
                    domain = self.label_encoder.inverse_transform([idx])[0]
                    top_domains.append({
                        'domain': domain,
                        'probability': float(probabilities[idx])
                    })
            else:
                top_domains = [{'domain': prediction, 'probability': 1.0}]

            # Determine parent category
            parent_category = self._get_parent_category(prediction)

            # Extract domain-specific keywords
            keywords = self._extract_domain_keywords(processed_text, prediction)

            return {
                'primary_domain': prediction,
                'confidence': confidence,
                'parent_category': parent_category,
                'top_domains': top_domains,
                'domain_keywords': keywords,
                'classification_method': 'ml_model'
            }

        except Exception as e:
            logger.error(f"Error classifying domain: {e}")
            # Fallback to rule-based
            return await self._rule_based_classification(text)

    async def _rule_based_classification(self, text: str) -> Dict[str, Any]:
        """Rule-based domain classification fallback."""
        text_lower = text.lower()

        # Domain keyword dictionaries
        domain_keywords = {
            'physics': ['quantum', 'particle', 'energy', 'force', 'gravity', 'relativity'],
            'chemistry': ['molecule', 'reaction', 'compound', 'element', 'catalyst', 'synthesis'],
            'biology': ['cell', 'gene', 'protein', 'organism', 'evolution', 'dna', 'rna'],
            'computer_science': ['algorithm', 'software', 'database', 'network', 'ai', 'machine learning'],
            'mathematics': ['theorem', 'proof', 'equation', 'algebra', 'calculus', 'topology'],
            'medicine': ['patient', 'treatment', 'diagnosis', 'disease', 'clinical', 'therapy'],
            'psychology': ['behavior', 'cognition', 'mental', 'perception', 'consciousness', 'personality'],
            'economics': ['market', 'inflation', 'gdp', 'investment', 'trade', 'finance']
        }

        # Count keyword matches
        scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[domain] = score

        if not scores:
            return {
                'primary_domain': 'general',
                'confidence': 0.0,
                'parent_category': 'unclassified',
                'classification_method': 'rule_based_fallback'
            }

        # Get best match
        best_domain = max(scores.items(), key=lambda x: x[1])

        return {
            'primary_domain': best_domain[0],
            'confidence': min(best_domain[1] / 5, 1.0),  # Normalize score
            'parent_category': self._get_parent_category(best_domain[0]),
            'top_domains': [{'domain': d, 'score': s} for d, s in sorted(
                scores.items(), key=lambda x: x[1], reverse=True)[:3]],
            'classification_method': 'rule_based'
        }

    def _get_parent_category(self, domain: str) -> str:
        """Get parent category for a domain."""
        for parent, children in self.domain_hierarchy.items():
            if domain in children:
                return parent
        return 'interdisciplinary'

    def _extract_domain_keywords(self, text: str, domain: str) -> List[str]:
        """Extract domain-specific keywords from text."""
        # Domain-specific technical terms
        domain_terms = {
            'physics': r'\b(quantum|photon|electron|neutron|quark|boson|fermion)\b',
            'chemistry': r'\b(mol|catalyst|oxidation|reduction|synthesis|polymer)\b',
            'biology': r'\b(cell|protein|gene|enzyme|metabol|organism|species)\b',
            'computer_science': r'\b(algorithm|data structure|complexity|runtime|cache)\b',
            'mathematics': r'\b(theorem|lemma|proof|axiom|conjecture|corollary)\b'
        }

        keywords = []
        if domain in domain_terms:
            pattern = domain_terms[domain]
            matches = re.findall(pattern, text.lower())
            keywords = list(set(matches))[:10]  # Limit to 10 keywords

        return keywords

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for classification."""
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.lower()

    async def classify_quality(self,
                              hypothesis_data: Dict) -> Dict[str, Any]:
        """
        Classify the quality level of a hypothesis.

        Args:
            hypothesis_data: Hypothesis information

        Returns:
            Quality classification results
        """
        try:
            # Extract quality features
            features = self._extract_quality_features(hypothesis_data)

            # Rule-based quality classification
            quality_score = 0.0

            # Check evidence quality
            evidence = hypothesis_data.get('evidence', [])
            if evidence:
                peer_reviewed = sum(1 for e in evidence
                                  if e.get('source', {}).get('peer_reviewed'))
                quality_score += (peer_reviewed / len(evidence)) * 0.3

            # Check methodology
            if hypothesis_data.get('methodology'):
                quality_score += 0.2

            # Check statistical rigor
            if hypothesis_data.get('statistical_analysis'):
                quality_score += 0.2

            # Check reproducibility
            if hypothesis_data.get('reproducible'):
                quality_score += 0.15

            # Check sample size
            sample_size = hypothesis_data.get('sample_size', 0)
            if sample_size >= 100:
                quality_score += 0.15
            elif sample_size >= 30:
                quality_score += 0.1

            # Determine quality level
            if quality_score >= 0.8:
                quality_level = 'excellent'
                confidence = 0.9
            elif quality_score >= 0.6:
                quality_level = 'good'
                confidence = 0.8
            elif quality_score >= 0.4:
                quality_level = 'fair'
                confidence = 0.7
            else:
                quality_level = 'poor'
                confidence = 0.6

            # Get improvement suggestions
            suggestions = self._get_quality_improvements(hypothesis_data, quality_score)

            return {
                'quality_level': quality_level,
                'quality_score': float(quality_score),
                'confidence': float(confidence),
                'quality_factors': {
                    'evidence_quality': float(peer_reviewed / len(evidence))
                                      if evidence else 0.0,
                    'has_methodology': bool(hypothesis_data.get('methodology')),
                    'statistical_rigor': bool(hypothesis_data.get('statistical_analysis')),
                    'reproducible': bool(hypothesis_data.get('reproducible')),
                    'adequate_sample': sample_size >= 30
                },
                'improvement_suggestions': suggestions
            }

        except Exception as e:
            logger.error(f"Error classifying quality: {e}")
            return {
                'quality_level': 'unknown',
                'error': str(e)
            }

    def _extract_quality_features(self, hypothesis_data: Dict) -> np.ndarray:
        """Extract numerical features for quality classification."""
        features = []

        # Evidence features
        evidence = hypothesis_data.get('evidence', [])
        features.append(len(evidence))
        features.append(sum(1 for e in evidence
                          if e.get('source', {}).get('peer_reviewed')))

        # Methodology features
        features.append(1 if hypothesis_data.get('methodology') else 0)
        features.append(1 if hypothesis_data.get('statistical_analysis') else 0)

        # Sample features
        features.append(hypothesis_data.get('sample_size', 0))

        # Assumption features
        assumptions = hypothesis_data.get('assumptions', [])
        features.append(len(assumptions))

        return np.array(features)

    def _get_quality_improvements(self,
                                 hypothesis_data: Dict,
                                 current_score: float) -> List[str]:
        """Generate quality improvement suggestions."""
        suggestions = []

        if current_score < 0.8:
            # Check evidence
            evidence = hypothesis_data.get('evidence', [])
            if not evidence:
                suggestions.append("Add supporting evidence from credible sources")
            elif not any(e.get('source', {}).get('peer_reviewed') for e in evidence):
                suggestions.append("Include peer-reviewed evidence")

            # Check methodology
            if not hypothesis_data.get('methodology'):
                suggestions.append("Document research methodology")

            # Check statistical analysis
            if not hypothesis_data.get('statistical_analysis'):
                suggestions.append("Include statistical analysis plan")

            # Check sample size
            if hypothesis_data.get('sample_size', 0) < 30:
                suggestions.append("Increase sample size for statistical validity")

            # Check reproducibility
            if not hypothesis_data.get('reproducible'):
                suggestions.append("Ensure methodology is reproducible")

        return suggestions[:5]  # Limit to 5 suggestions

    async def classify_research_type(self, text: str) -> Dict[str, Any]:
        """
        Classify the type of research.

        Args:
            text: Research description

        Returns:
            Research type classification
        """
        try:
            text_lower = text.lower()

            # Research type indicators
            type_indicators = {
                'experimental': ['experiment', 'control group', 'treatment', 'randomized', 'trial'],
                'observational': ['observe', 'survey', 'cohort', 'cross-sectional', 'naturalistic'],
                'theoretical': ['theory', 'model', 'framework', 'conceptual', 'hypothesis'],
                'computational': ['simulation', 'algorithm', 'compute', 'model', 'numerical'],
                'review': ['review', 'meta-analysis', 'systematic', 'literature', 'synthesis'],
                'case_study': ['case study', 'case report', 'individual', 'specific instance'],
                'survey': ['questionnaire', 'survey', 'poll', 'interview', 'respondents']
            }

            # Score each type
            scores = {}
            for research_type, indicators in type_indicators.items():
                score = sum(1 for indicator in indicators if indicator in text_lower)
                if score > 0:
                    scores[research_type] = score

            if not scores:
                return {
                    'research_type': 'unspecified',
                    'confidence': 0.0
                }

            # Get primary type
            primary_type = max(scores.items(), key=lambda x: x[1])

            return {
                'research_type': primary_type[0],
                'confidence': min(primary_type[1] / 3, 1.0),
                'all_types': list(scores.keys()),
                'type_scores': scores
            }

        except Exception as e:
            logger.error(f"Error classifying research type: {e}")
            return {'research_type': 'unknown', 'error': str(e)}

    async def train_domain_classifier(self,
                                     training_data: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Train or update the domain classifier.

        Args:
            training_data: List of (text, domain) tuples

        Returns:
            Training results
        """
        try:
            if len(training_data) < 10:
                return {'error': 'Need at least 10 training examples'}

            # Prepare data
            texts, labels = zip(*training_data)

            # Preprocess texts
            processed_texts = [self._preprocess_text(text) for text in texts]

            # Fit vectorizer if needed
            if not hasattr(self.vectorizer, 'vocabulary_'):
                self.vectorizer.fit(processed_texts)

            # Transform texts
            X = self.vectorizer.transform(processed_texts)

            # Encode labels
            y = self.label_encoder.fit_transform(labels)

            # Train classifier
            self.domain_classifier.fit(X, y)

            # Cross-validation
            cv_scores = cross_val_score(self.domain_classifier, X, y, cv=min(5, len(labels)))

            # Save models
            await self._save_models()

            return {
                'training_samples': len(training_data),
                'unique_domains': len(set(labels)),
                'cv_accuracy': float(np.mean(cv_scores)),
                'cv_std': float(np.std(cv_scores)),
                'model_saved': True
            }

        except Exception as e:
            logger.error(f"Error training classifier: {e}")
            return {'error': str(e)}

    async def _save_models(self):
        """Save trained models to disk."""
        try:
            # Save vectorizer
            vectorizer_path = self.model_path / 'tfidf_vectorizer.pkl'
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)

            # Save domain classifier
            domain_model_path = self.model_path / 'domain_classifier.pkl'
            with open(domain_model_path, 'wb') as f:
                pickle.dump(self.domain_classifier, f)

            # Save label encoder
            encoder_path = self.model_path / 'label_encoder.pkl'
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)

            logger.info("Models saved successfully")

        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def _load_models(self):
        """Load pre-trained models from disk."""
        try:
            vectorizer_path = self.model_path / 'tfidf_vectorizer.pkl'
            domain_model_path = self.model_path / 'domain_classifier.pkl'
            encoder_path = self.model_path / 'label_encoder.pkl'

            if vectorizer_path.exists():
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)

            if domain_model_path.exists():
                with open(domain_model_path, 'rb') as f:
                    self.domain_classifier = pickle.load(f)

            if encoder_path.exists():
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)

                logger.info("Pre-trained models loaded successfully")

        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")

    async def batch_classify(self,
                           texts: List[str],
                           classification_type: str = 'domain') -> List[Dict]:
        """
        Classify multiple texts in batch.

        Args:
            texts: List of texts to classify
            classification_type: Type of classification ('domain', 'quality', 'research_type')

        Returns:
            List of classification results
        """
        results = []

        for text in texts:
            if classification_type == 'domain':
                result = await self.classify_domain(text)
            elif classification_type == 'research_type':
                result = await self.classify_research_type(text)
            else:
                result = {'error': f'Unknown classification type: {classification_type}'}

            results.append(result)

        return results