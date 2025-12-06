"""
Recommendation Engine for TalAI

Intelligent recommendation system for suggesting similar validations,
validation modes, evidence sources, and expert reviewers.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import hashlib
import json

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """
    Generate intelligent recommendations for research validation.

    Features:
    - Similar validation suggestions
    - Validation mode recommendations
    - Evidence source suggestions
    - Expert reviewer matching
    - Citation recommendations
    - Collaborative filtering
    - Content-based filtering
    - Hybrid recommendations
    """

    def __init__(self):
        """Initialize recommendation engine."""
        self.validation_history = []
        self.user_preferences = defaultdict(dict)
        self.expert_profiles = {}
        self.citation_graph = defaultdict(set)
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

    async def recommend_similar_validations(self,
                                           current_hypothesis: Dict,
                                           validation_database: List[Dict],
                                           top_k: int = 5) -> List[Dict]:
        """
        Recommend similar past validations.

        Args:
            current_hypothesis: Current hypothesis data
            validation_database: Historical validation data
            top_k: Number of recommendations

        Returns:
            List of similar validation recommendations
        """
        try:
            if not validation_database:
                return []

            # Extract features from current hypothesis
            current_features = await self._extract_hypothesis_features(current_hypothesis)

            # Calculate similarities with past validations
            similarities = []
            for validation in validation_database:
                # Skip if same hypothesis
                if validation.get('hypothesis_id') == current_hypothesis.get('id'):
                    continue

                # Extract features
                val_features = await self._extract_hypothesis_features(validation)

                # Calculate multi-dimensional similarity
                similarity = await self._calculate_similarity(current_features, val_features)

                similarities.append({
                    'validation': validation,
                    'similarity_score': similarity['overall'],
                    'similarity_breakdown': similarity
                })

            # Sort by similarity
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)

            # Get top-k recommendations
            recommendations = []
            for sim in similarities[:top_k]:
                validation = sim['validation']
                recommendation = {
                    'validation_id': validation.get('id', self._generate_id(str(validation))),
                    'hypothesis': validation.get('statement', '')[:200],
                    'domain': validation.get('domain', 'unknown'),
                    'success_rate': validation.get('success_rate', 0.0),
                    'similarity_score': sim['similarity_score'],
                    'similarity_type': self._determine_similarity_type(sim['similarity_breakdown']),
                    'insights': await self._extract_insights(validation, current_hypothesis),
                    'validation_date': validation.get('timestamp', ''),
                    'validation_mode': validation.get('mode', 'standard')
                }
                recommendations.append(recommendation)

            return recommendations

        except Exception as e:
            logger.error(f"Error recommending similar validations: {e}")
            return []

    async def _extract_hypothesis_features(self, hypothesis: Dict) -> Dict[str, Any]:
        """Extract multi-dimensional features from hypothesis."""
        features = {
            'text': hypothesis.get('statement', ''),
            'domain': hypothesis.get('domain', 'general'),
            'evidence_count': len(hypothesis.get('evidence', [])),
            'methodology': hypothesis.get('methodology', {}),
            'assumptions': hypothesis.get('assumptions', []),
            'keywords': self._extract_keywords(hypothesis.get('statement', '')),
            'complexity': len(hypothesis.get('statement', '').split()),
            'validation_mode': hypothesis.get('validation_mode', 'standard')
        }
        return features

    async def _calculate_similarity(self,
                                  features1: Dict,
                                  features2: Dict) -> Dict[str, float]:
        """Calculate multi-dimensional similarity between features."""
        similarities = {}

        # Text similarity
        if features1['text'] and features2['text']:
            vectorizer = TfidfVectorizer(max_features=100)
            try:
                tfidf_matrix = vectorizer.fit_transform([features1['text'], features2['text']])
                text_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                similarities['text'] = float(text_sim)
            except:
                similarities['text'] = 0.0
        else:
            similarities['text'] = 0.0

        # Domain similarity
        similarities['domain'] = 1.0 if features1['domain'] == features2['domain'] else 0.3

        # Evidence similarity
        evidence_diff = abs(features1['evidence_count'] - features2['evidence_count'])
        similarities['evidence'] = max(0, 1 - evidence_diff / 10)

        # Keyword overlap
        keywords1 = set(features1['keywords'])
        keywords2 = set(features2['keywords'])
        if keywords1 and keywords2:
            overlap = len(keywords1 & keywords2) / len(keywords1 | keywords2)
            similarities['keywords'] = overlap
        else:
            similarities['keywords'] = 0.0

        # Complexity similarity
        complexity_diff = abs(features1['complexity'] - features2['complexity'])
        similarities['complexity'] = max(0, 1 - complexity_diff / 50)

        # Calculate weighted overall similarity
        weights = {
            'text': 0.4,
            'domain': 0.2,
            'keywords': 0.2,
            'evidence': 0.1,
            'complexity': 0.1
        }

        overall = sum(similarities.get(key, 0) * weight
                     for key, weight in weights.items())

        similarities['overall'] = overall

        return similarities

    def _determine_similarity_type(self, similarity_breakdown: Dict) -> str:
        """Determine the primary type of similarity."""
        # Remove 'overall' from consideration
        similarities = {k: v for k, v in similarity_breakdown.items() if k != 'overall'}

        if not similarities:
            return 'general'

        # Find highest similarity dimension
        max_dim = max(similarities.items(), key=lambda x: x[1])

        if max_dim[0] == 'text':
            return 'content_based'
        elif max_dim[0] == 'domain':
            return 'domain_specific'
        elif max_dim[0] == 'keywords':
            return 'topic_based'
        else:
            return 'structural'

    async def _extract_insights(self,
                               past_validation: Dict,
                               current_hypothesis: Dict) -> List[str]:
        """Extract actionable insights from similar validation."""
        insights = []

        # Success rate insight
        past_success = past_validation.get('success_rate', 0.5)
        if past_success > 0.8:
            insights.append(f"Similar hypothesis achieved {past_success:.0%} success rate")
        elif past_success < 0.3:
            insights.append(f"Similar hypothesis had challenges (only {past_success:.0%} success)")

        # Evidence insight
        past_evidence = len(past_validation.get('evidence', []))
        current_evidence = len(current_hypothesis.get('evidence', []))
        if past_evidence > current_evidence + 3:
            insights.append(f"Consider adding more evidence (similar used {past_evidence} sources)")

        # Methodology insight
        if past_validation.get('methodology') and not current_hypothesis.get('methodology'):
            insights.append("Similar validation benefited from clear methodology")

        # Duration insight
        duration = past_validation.get('duration_seconds', 0)
        if duration > 0:
            insights.append(f"Expected duration: ~{duration/60:.1f} minutes")

        return insights[:3]

    def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract keywords from text using TF-IDF."""
        try:
            # Simple keyword extraction
            words = text.lower().split()
            # Remove common words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                         'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were'}
            keywords = [w for w in words if len(w) > 3 and w not in stop_words]
            # Count frequency
            word_counts = Counter(keywords)
            return [word for word, _ in word_counts.most_common(top_n)]
        except:
            return []

    def _generate_id(self, content: str) -> str:
        """Generate unique ID from content."""
        return hashlib.md5(content.encode()).hexdigest()[:12]

    async def recommend_validation_mode(self,
                                       hypothesis_data: Dict,
                                       user_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Recommend optimal validation mode.

        Args:
            hypothesis_data: Hypothesis information
            user_history: User's validation history

        Returns:
            Validation mode recommendation
        """
        try:
            scores = {
                'quick': 0.0,
                'standard': 0.5,  # Default bias
                'comprehensive': 0.0,
                'peer_reviewed': 0.0
            }

            reasons = defaultdict(list)

            # Analyze hypothesis complexity
            statement_length = len(hypothesis_data.get('statement', ''))
            if statement_length < 100:
                scores['quick'] += 0.3
                reasons['quick'].append("Simple hypothesis statement")
            elif statement_length > 500:
                scores['comprehensive'] += 0.3
                reasons['comprehensive'].append("Complex hypothesis requires thorough validation")

            # Check evidence availability
            evidence_count = len(hypothesis_data.get('evidence', []))
            if evidence_count < 3:
                scores['quick'] += 0.2
                reasons['quick'].append("Limited evidence available")
            elif evidence_count > 10:
                scores['comprehensive'] += 0.2
                reasons['comprehensive'].append("Extensive evidence to analyze")

            # Check domain criticality
            critical_domains = ['medicine', 'healthcare', 'safety', 'security']
            domain = hypothesis_data.get('domain', '').lower()
            if any(cd in domain for cd in critical_domains):
                scores['peer_reviewed'] += 0.4
                scores['comprehensive'] += 0.2
                reasons['peer_reviewed'].append(f"Critical domain ({domain}) requires peer review")

            # Check assumptions risk
            assumptions = hypothesis_data.get('assumptions', [])
            high_risk = sum(1 for a in assumptions if a.get('risk_level') == 'high')
            if high_risk > 2:
                scores['comprehensive'] += 0.3
                scores['peer_reviewed'] += 0.2
                reasons['comprehensive'].append(f"{high_risk} high-risk assumptions need validation")

            # User preference learning
            if user_history:
                mode_counts = Counter(val.get('mode', 'standard') for val in user_history[-10:])
                preferred_mode = mode_counts.most_common(1)[0][0] if mode_counts else 'standard'
                scores[preferred_mode] += 0.2
                reasons[preferred_mode].append("Aligns with your recent preferences")

            # Time constraints
            if hypothesis_data.get('urgent', False):
                scores['quick'] += 0.3
                reasons['quick'].append("Urgent validation requested")

            # Determine recommendation
            recommended_mode = max(scores.items(), key=lambda x: x[1])[0]

            # Calculate confidence
            score_values = list(scores.values())
            confidence = (max(score_values) - np.mean(score_values)) / max(score_values)

            return {
                'recommended_mode': recommended_mode,
                'confidence': float(confidence),
                'mode_scores': scores,
                'reasons': dict(reasons),
                'alternative_modes': sorted(
                    [(mode, score) for mode, score in scores.items() if mode != recommended_mode],
                    key=lambda x: x[1],
                    reverse=True
                ),
                'estimated_duration': self._estimate_mode_duration(recommended_mode, hypothesis_data)
            }

        except Exception as e:
            logger.error(f"Error recommending validation mode: {e}")
            return {
                'recommended_mode': 'standard',
                'confidence': 0.5,
                'error': str(e)
            }

    def _estimate_mode_duration(self, mode: str, hypothesis_data: Dict) -> Dict[str, float]:
        """Estimate validation duration for different modes."""
        base_duration = 120  # 2 minutes base
        evidence_factor = len(hypothesis_data.get('evidence', [])) * 10

        durations = {
            'quick': (base_duration + evidence_factor) * 0.5,
            'standard': base_duration + evidence_factor,
            'comprehensive': (base_duration + evidence_factor) * 2,
            'peer_reviewed': (base_duration + evidence_factor) * 3
        }

        return {
            'seconds': durations.get(mode, base_duration),
            'minutes': durations.get(mode, base_duration) / 60
        }

    async def suggest_evidence_sources(self,
                                      hypothesis_data: Dict,
                                      existing_sources: List[Dict] = None) -> List[Dict]:
        """
        Suggest additional evidence sources.

        Args:
            hypothesis_data: Hypothesis information
            existing_sources: Already collected evidence

        Returns:
            Evidence source suggestions
        """
        try:
            suggestions = []
            existing_sources = existing_sources or hypothesis_data.get('evidence', [])

            # Extract domain and keywords
            domain = hypothesis_data.get('domain', 'general')
            keywords = self._extract_keywords(hypothesis_data.get('statement', ''))

            # Domain-specific source databases
            domain_sources = {
                'physics': ['arXiv Physics', 'Physical Review Letters', 'Nature Physics'],
                'chemistry': ['Journal of American Chemical Society', 'Chemical Reviews', 'Nature Chemistry'],
                'biology': ['Nature', 'Science', 'Cell', 'PLOS Biology'],
                'medicine': ['PubMed', 'The Lancet', 'NEJM', 'JAMA'],
                'computer_science': ['arXiv CS', 'ACM Digital Library', 'IEEE Xplore'],
                'mathematics': ['arXiv Math', 'Annals of Mathematics', 'Journal of AMS'],
                'economics': ['NBER', 'Journal of Economic Theory', 'Econometrica']
            }

            # Get domain-specific sources
            recommended_sources = domain_sources.get(domain, ['Google Scholar', 'arXiv'])

            # Check existing source types
            existing_types = set(s.get('source', {}).get('type', '') for s in existing_sources)

            # Suggest diverse source types
            source_types = {
                'journal': 'Peer-reviewed journal articles',
                'conference': 'Conference proceedings',
                'book': 'Academic books and monographs',
                'preprint': 'Preprint servers (arXiv, bioRxiv)',
                'dataset': 'Public datasets and repositories',
                'patent': 'Patent databases',
                'thesis': 'Doctoral dissertations'
            }

            for source_type, description in source_types.items():
                if source_type not in existing_types:
                    suggestions.append({
                        'type': source_type,
                        'description': description,
                        'priority': 'high' if source_type in ['journal', 'conference'] else 'medium',
                        'search_terms': keywords[:5]
                    })

            # Add specific database suggestions
            for source in recommended_sources[:3]:
                suggestions.append({
                    'database': source,
                    'domain': domain,
                    'search_query': ' OR '.join(keywords[:3]),
                    'expected_results': 'high' if domain in source.lower() else 'medium'
                })

            # Check for missing evidence types
            if not any(s.get('source', {}).get('peer_reviewed') for s in existing_sources):
                suggestions.insert(0, {
                    'priority': 'critical',
                    'type': 'peer_reviewed',
                    'recommendation': 'Add at least 3 peer-reviewed sources for credibility',
                    'databases': recommended_sources
                })

            # Suggest recent sources if all are old
            if existing_sources:
                dates = [s.get('source', {}).get('publication_date', '') for s in existing_sources]
                recent_count = sum(1 for d in dates if d and '202' in d)
                if recent_count < len(existing_sources) / 2:
                    suggestions.append({
                        'priority': 'high',
                        'type': 'recent',
                        'recommendation': 'Include more recent publications (2020 or later)',
                        'search_filter': 'publication_date:[2020 TO *]'
                    })

            return suggestions[:10]

        except Exception as e:
            logger.error(f"Error suggesting evidence sources: {e}")
            return []

    async def recommend_expert_reviewers(self,
                                        hypothesis_data: Dict,
                                        available_experts: List[Dict] = None) -> List[Dict]:
        """
        Recommend expert reviewers for hypothesis.

        Args:
            hypothesis_data: Hypothesis information
            available_experts: Pool of available experts

        Returns:
            Expert reviewer recommendations
        """
        try:
            if not available_experts:
                # Generate mock expert profiles
                available_experts = self._generate_mock_experts()

            recommendations = []
            domain = hypothesis_data.get('domain', 'general')
            keywords = set(self._extract_keywords(hypothesis_data.get('statement', '')))

            for expert in available_experts:
                # Calculate match score
                match_score = 0.0
                match_reasons = []

                # Domain match
                if domain in expert.get('domains', []):
                    match_score += 0.4
                    match_reasons.append(f"Domain expert in {domain}")
                elif any(d in expert.get('domains', []) for d in [domain]):
                    match_score += 0.2
                    match_reasons.append("Related domain expertise")

                # Keyword overlap
                expert_keywords = set(expert.get('expertise_keywords', []))
                overlap = len(keywords & expert_keywords)
                if overlap > 0:
                    match_score += min(overlap * 0.1, 0.3)
                    match_reasons.append(f"{overlap} matching expertise areas")

                # Publication relevance
                if expert.get('publication_count', 0) > 50:
                    match_score += 0.1
                    match_reasons.append("Highly published researcher")

                # Availability
                if expert.get('available', True):
                    match_score += 0.1
                else:
                    match_score *= 0.5  # Penalize unavailable experts

                # Review history
                if expert.get('avg_review_quality', 0) > 4.0:
                    match_score += 0.1
                    match_reasons.append("Excellent review history")

                if match_score > 0.3:  # Threshold for recommendation
                    recommendations.append({
                        'expert_id': expert.get('id', self._generate_id(str(expert))),
                        'name': expert.get('name', 'Expert'),
                        'match_score': float(match_score),
                        'primary_domain': expert.get('domains', ['general'])[0],
                        'expertise_keywords': list(expert_keywords)[:5],
                        'match_reasons': match_reasons,
                        'availability': expert.get('available', True),
                        'response_time_hours': expert.get('avg_response_time', 48),
                        'review_quality': expert.get('avg_review_quality', 3.5)
                    })

            # Sort by match score
            recommendations.sort(key=lambda x: x['match_score'], reverse=True)

            return recommendations[:5]

        except Exception as e:
            logger.error(f"Error recommending experts: {e}")
            return []

    def _generate_mock_experts(self) -> List[Dict]:
        """Generate mock expert profiles for demonstration."""
        return [
            {
                'id': 'exp001',
                'name': 'Dr. Sarah Chen',
                'domains': ['physics', 'quantum computing'],
                'expertise_keywords': ['quantum', 'entanglement', 'qubits', 'algorithms'],
                'publication_count': 87,
                'available': True,
                'avg_response_time': 24,
                'avg_review_quality': 4.7
            },
            {
                'id': 'exp002',
                'name': 'Prof. Michael Johnson',
                'domains': ['biology', 'genetics'],
                'expertise_keywords': ['gene', 'dna', 'crispr', 'protein', 'cell'],
                'publication_count': 156,
                'available': True,
                'avg_response_time': 36,
                'avg_review_quality': 4.5
            },
            {
                'id': 'exp003',
                'name': 'Dr. Emily Williams',
                'domains': ['computer_science', 'ai'],
                'expertise_keywords': ['machine learning', 'neural networks', 'nlp', 'vision'],
                'publication_count': 45,
                'available': False,
                'avg_response_time': 72,
                'avg_review_quality': 4.2
            },
            {
                'id': 'exp004',
                'name': 'Prof. Robert Zhang',
                'domains': ['chemistry', 'materials'],
                'expertise_keywords': ['synthesis', 'catalysis', 'polymers', 'nanoparticles'],
                'publication_count': 203,
                'available': True,
                'avg_response_time': 48,
                'avg_review_quality': 4.8
            },
            {
                'id': 'exp005',
                'name': 'Dr. Lisa Anderson',
                'domains': ['medicine', 'neuroscience'],
                'expertise_keywords': ['brain', 'cognition', 'disorders', 'treatment', 'clinical'],
                'publication_count': 92,
                'available': True,
                'avg_response_time': 60,
                'avg_review_quality': 4.6
            }
        ]

    async def recommend_citations(self,
                                 hypothesis_data: Dict,
                                 citation_database: List[Dict] = None) -> List[Dict]:
        """
        Recommend relevant citations.

        Args:
            hypothesis_data: Hypothesis information
            citation_database: Available citations

        Returns:
            Citation recommendations
        """
        try:
            if not citation_database:
                # Generate mock citations
                citation_database = self._generate_mock_citations(
                    hypothesis_data.get('domain', 'general')
                )

            recommendations = []
            statement = hypothesis_data.get('statement', '')
            keywords = set(self._extract_keywords(statement))

            for citation in citation_database:
                # Calculate relevance score
                relevance_score = 0.0
                relevance_factors = []

                # Title relevance
                title_keywords = set(self._extract_keywords(citation.get('title', '')))
                title_overlap = len(keywords & title_keywords)
                if title_overlap > 0:
                    relevance_score += min(title_overlap * 0.15, 0.5)
                    relevance_factors.append(f"{title_overlap} matching keywords in title")

                # Abstract relevance
                abstract_keywords = set(self._extract_keywords(citation.get('abstract', '')))
                abstract_overlap = len(keywords & abstract_keywords)
                if abstract_overlap > 0:
                    relevance_score += min(abstract_overlap * 0.1, 0.3)

                # Recency bonus
                year = citation.get('year', 2020)
                if year >= 2022:
                    relevance_score += 0.1
                    relevance_factors.append("Recent publication")

                # Impact factor
                if citation.get('citations', 0) > 100:
                    relevance_score += 0.1
                    relevance_factors.append("Highly cited")

                if relevance_score > 0.2:  # Threshold
                    recommendations.append({
                        'citation_id': citation.get('id', self._generate_id(str(citation))),
                        'title': citation.get('title', ''),
                        'authors': citation.get('authors', [])[:3],
                        'year': citation.get('year', 2020),
                        'journal': citation.get('journal', 'Unknown'),
                        'relevance_score': float(relevance_score),
                        'relevance_factors': relevance_factors,
                        'citation_count': citation.get('citations', 0),
                        'doi': citation.get('doi', ''),
                        'abstract_snippet': citation.get('abstract', '')[:200]
                    })

            # Sort by relevance
            recommendations.sort(key=lambda x: x['relevance_score'], reverse=True)

            return recommendations[:10]

        except Exception as e:
            logger.error(f"Error recommending citations: {e}")
            return []

    def _generate_mock_citations(self, domain: str) -> List[Dict]:
        """Generate mock citations for demonstration."""
        domain_citations = {
            'physics': [
                {
                    'id': 'cit001',
                    'title': 'Quantum Entanglement in Many-Body Systems',
                    'authors': ['A. Einstein', 'B. Podolsky', 'N. Rosen'],
                    'year': 2023,
                    'journal': 'Physical Review Letters',
                    'citations': 342,
                    'doi': '10.1103/PhysRevLett.130.050401',
                    'abstract': 'We investigate quantum entanglement properties in many-body systems...'
                }
            ],
            'biology': [
                {
                    'id': 'cit002',
                    'title': 'CRISPR-Cas9 Gene Editing: Recent Advances',
                    'authors': ['J. Doudna', 'E. Charpentier'],
                    'year': 2022,
                    'journal': 'Science',
                    'citations': 567,
                    'doi': '10.1126/science.abc123',
                    'abstract': 'Recent advances in CRISPR technology enable precise genetic modifications...'
                }
            ]
        }

        # Return domain-specific or general citations
        return domain_citations.get(domain, [
            {
                'id': 'cit_gen',
                'title': 'Advances in Scientific Research Methodology',
                'authors': ['Various Authors'],
                'year': 2023,
                'journal': 'Nature Reviews',
                'citations': 189,
                'doi': '10.1038/nrm.2023.1',
                'abstract': 'A comprehensive review of modern research methodologies...'
            }
        ])

    async def get_collaborative_recommendations(self,
                                               user_id: str,
                                               hypothesis_data: Dict) -> Dict[str, Any]:
        """
        Generate collaborative filtering recommendations.

        Args:
            user_id: User identifier
            hypothesis_data: Current hypothesis

        Returns:
            Collaborative recommendations
        """
        try:
            # Get user preferences
            user_prefs = self.user_preferences.get(user_id, {})

            # Find similar users (simplified)
            similar_users = []
            for other_id, other_prefs in self.user_preferences.items():
                if other_id != user_id:
                    similarity = self._calculate_user_similarity(user_prefs, other_prefs)
                    if similarity > 0.5:
                        similar_users.append((other_id, similarity))

            # Sort by similarity
            similar_users.sort(key=lambda x: x[1], reverse=True)

            # Get recommendations from similar users
            recommendations = {
                'validation_modes': [],
                'domains': [],
                'methodologies': []
            }

            for similar_id, similarity in similar_users[:5]:
                similar_prefs = self.user_preferences[similar_id]

                # Aggregate preferences
                if 'preferred_mode' in similar_prefs:
                    recommendations['validation_modes'].append({
                        'mode': similar_prefs['preferred_mode'],
                        'weight': similarity
                    })

                if 'domains' in similar_prefs:
                    for domain in similar_prefs['domains']:
                        recommendations['domains'].append({
                            'domain': domain,
                            'weight': similarity
                        })

            return {
                'similar_users_found': len(similar_users),
                'recommendations': recommendations,
                'confidence': min(len(similar_users) / 10, 1.0)
            }

        except Exception as e:
            logger.error(f"Error in collaborative recommendations: {e}")
            return {'error': str(e)}

    def _calculate_user_similarity(self, prefs1: Dict, prefs2: Dict) -> float:
        """Calculate similarity between user preferences."""
        similarity = 0.0
        comparisons = 0

        # Compare domains
        if 'domains' in prefs1 and 'domains' in prefs2:
            domains1 = set(prefs1['domains'])
            domains2 = set(prefs2['domains'])
            if domains1 and domains2:
                similarity += len(domains1 & domains2) / len(domains1 | domains2)
                comparisons += 1

        # Compare validation modes
        if 'preferred_mode' in prefs1 and 'preferred_mode' in prefs2:
            similarity += 1.0 if prefs1['preferred_mode'] == prefs2['preferred_mode'] else 0.0
            comparisons += 1

        return similarity / comparisons if comparisons > 0 else 0.0

    async def update_user_preferences(self,
                                     user_id: str,
                                     interaction_data: Dict):
        """Update user preferences based on interactions."""
        try:
            if user_id not in self.user_preferences:
                self.user_preferences[user_id] = {
                    'domains': [],
                    'validation_modes': [],
                    'interaction_count': 0
                }

            prefs = self.user_preferences[user_id]

            # Update domains
            if 'domain' in interaction_data:
                prefs['domains'].append(interaction_data['domain'])
                # Keep only recent domains
                prefs['domains'] = prefs['domains'][-20:]

            # Update validation modes
            if 'validation_mode' in interaction_data:
                prefs['validation_modes'].append(interaction_data['validation_mode'])
                prefs['validation_modes'] = prefs['validation_modes'][-20:]

                # Calculate preferred mode
                mode_counts = Counter(prefs['validation_modes'])
                prefs['preferred_mode'] = mode_counts.most_common(1)[0][0]

            # Update interaction count
            prefs['interaction_count'] += 1

            logger.info(f"Updated preferences for user {user_id}")

        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")