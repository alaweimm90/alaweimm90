"""
Quality Scoring Module for TalAI

Comprehensive scoring algorithms for hypothesis quality, evidence assessment,
and research validation metrics.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class QualityScore:
    """Comprehensive quality score with breakdown."""
    overall_score: float
    components: Dict[str, float]
    confidence: float
    feedback: List[str]
    timestamp: datetime

class QualityScorer:
    """
    Advanced quality scoring system for research validation.

    Features:
    - Multi-dimensional scoring
    - Evidence quality assessment
    - Assumption risk scoring
    - Hypothesis complexity analysis
    - Research impact estimation
    - Weighted composite scoring
    """

    def __init__(self):
        """Initialize quality scorer."""
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scoring_weights = {
            'evidence_quality': 0.25,
            'methodology_rigor': 0.20,
            'assumption_validity': 0.15,
            'logical_consistency': 0.15,
            'novelty': 0.10,
            'feasibility': 0.10,
            'impact_potential': 0.05
        }

    async def score_hypothesis(self, hypothesis_data: Dict) -> QualityScore:
        """
        Score a hypothesis across multiple dimensions.

        Args:
            hypothesis_data: Hypothesis information

        Returns:
            Comprehensive quality score
        """
        try:
            scores = {}
            feedback = []

            # Score evidence quality
            evidence_score = await self._score_evidence_quality(
                hypothesis_data.get('evidence', [])
            )
            scores['evidence_quality'] = evidence_score['score']
            feedback.extend(evidence_score['feedback'])

            # Score methodology rigor
            methodology_score = await self._score_methodology(
                hypothesis_data.get('methodology', {})
            )
            scores['methodology_rigor'] = methodology_score['score']
            feedback.extend(methodology_score['feedback'])

            # Score assumption validity
            assumption_score = await self._score_assumptions(
                hypothesis_data.get('assumptions', [])
            )
            scores['assumption_validity'] = assumption_score['score']
            feedback.extend(assumption_score['feedback'])

            # Score logical consistency
            consistency_score = await self._score_logical_consistency(
                hypothesis_data
            )
            scores['logical_consistency'] = consistency_score

            # Score novelty
            novelty_score = await self._score_novelty(
                hypothesis_data.get('statement', ''),
                hypothesis_data.get('domain', 'general')
            )
            scores['novelty'] = novelty_score

            # Score feasibility
            feasibility_score = await self._score_feasibility(
                hypothesis_data
            )
            scores['feasibility'] = feasibility_score

            # Score impact potential
            impact_score = await self._score_impact_potential(
                hypothesis_data
            )
            scores['impact_potential'] = impact_score

            # Calculate weighted overall score
            overall_score = sum(
                scores[key] * self.scoring_weights[key]
                for key in scores
            )

            # Calculate confidence based on data completeness
            confidence = self._calculate_confidence(hypothesis_data)

            # Add overall feedback
            if overall_score > 0.8:
                feedback.insert(0, "Excellent hypothesis quality - ready for validation")
            elif overall_score > 0.6:
                feedback.insert(0, "Good hypothesis quality - minor improvements recommended")
            elif overall_score > 0.4:
                feedback.insert(0, "Fair hypothesis quality - significant improvements needed")
            else:
                feedback.insert(0, "Poor hypothesis quality - major revisions required")

            return QualityScore(
                overall_score=overall_score,
                components=scores,
                confidence=confidence,
                feedback=feedback,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Error scoring hypothesis: {e}")
            return QualityScore(
                overall_score=0.0,
                components={},
                confidence=0.0,
                feedback=[f"Error in scoring: {str(e)}"],
                timestamp=datetime.now()
            )

    async def _score_evidence_quality(self, evidence: List[Dict]) -> Dict[str, Any]:
        """Score the quality of evidence provided."""
        if not evidence:
            return {
                'score': 0.0,
                'feedback': ["No evidence provided"]
            }

        feedback = []
        quality_scores = []

        for e in evidence:
            e_score = 0.0

            # Check source reliability
            source = e.get('source', {})
            if source.get('peer_reviewed', False):
                e_score += 0.3
            elif source.get('type') in ['journal', 'conference']:
                e_score += 0.2
            else:
                e_score += 0.1

            # Check recency
            pub_date = source.get('publication_date')
            if pub_date:
                try:
                    date = datetime.fromisoformat(pub_date)
                    years_old = (datetime.now() - date).days / 365
                    if years_old < 2:
                        e_score += 0.2
                    elif years_old < 5:
                        e_score += 0.15
                    elif years_old < 10:
                        e_score += 0.1
                except:
                    pass

            # Check relevance
            relevance = e.get('relevance_score', 0.5)
            e_score += relevance * 0.3

            # Check statistical significance
            if e.get('statistical_significance'):
                p_value = e.get('p_value', 1.0)
                if p_value < 0.001:
                    e_score += 0.2
                elif p_value < 0.05:
                    e_score += 0.15
                else:
                    e_score += 0.05

            quality_scores.append(min(e_score, 1.0))

        avg_quality = np.mean(quality_scores)

        # Generate feedback
        if avg_quality < 0.5:
            feedback.append("Evidence quality is low - consider peer-reviewed sources")
        if len(evidence) < 3:
            feedback.append("Limited evidence - consider adding more sources")
        if all(not e.get('source', {}).get('peer_reviewed') for e in evidence):
            feedback.append("No peer-reviewed evidence - credibility may be questioned")

        return {
            'score': avg_quality,
            'feedback': feedback,
            'individual_scores': quality_scores
        }

    async def _score_methodology(self, methodology: Dict) -> Dict[str, Any]:
        """Score the research methodology."""
        score = 0.0
        feedback = []

        if not methodology:
            return {
                'score': 0.0,
                'feedback': ["No methodology specified"]
            }

        # Check for key methodology components
        if methodology.get('research_design'):
            score += 0.2
        else:
            feedback.append("Research design not specified")

        if methodology.get('sample_size'):
            sample_size = methodology['sample_size']
            if sample_size >= 100:
                score += 0.2
            elif sample_size >= 30:
                score += 0.15
            else:
                score += 0.05
                feedback.append("Small sample size may limit generalizability")

        if methodology.get('control_variables'):
            score += 0.15
        else:
            feedback.append("Control variables not identified")

        if methodology.get('statistical_methods'):
            methods = methodology['statistical_methods']
            if len(methods) >= 3:
                score += 0.2
            else:
                score += 0.1
        else:
            feedback.append("Statistical methods not described")

        if methodology.get('validation_approach'):
            score += 0.15
        else:
            feedback.append("Validation approach not defined")

        if methodology.get('limitations_acknowledged'):
            score += 0.1

        return {
            'score': min(score, 1.0),
            'feedback': feedback
        }

    async def _score_assumptions(self, assumptions: List[Dict]) -> Dict[str, Any]:
        """Score the validity and risk of assumptions."""
        if not assumptions:
            return {
                'score': 0.5,  # Neutral score for no assumptions
                'feedback': ["No assumptions explicitly stated"]
            }

        feedback = []
        risk_scores = []

        for assumption in assumptions:
            risk = assumption.get('risk_level', 'medium')
            justification = assumption.get('justification', '')

            # Score based on risk level and justification
            if risk == 'low' and justification:
                risk_scores.append(0.9)
            elif risk == 'low':
                risk_scores.append(0.7)
            elif risk == 'medium' and justification:
                risk_scores.append(0.6)
            elif risk == 'medium':
                risk_scores.append(0.4)
            elif risk == 'high' and justification:
                risk_scores.append(0.3)
            else:  # high risk, no justification
                risk_scores.append(0.1)

        avg_score = np.mean(risk_scores)

        # Generate feedback
        high_risk_count = sum(1 for a in assumptions if a.get('risk_level') == 'high')
        if high_risk_count > 2:
            feedback.append(f"{high_risk_count} high-risk assumptions may compromise validity")

        unjustified = sum(1 for a in assumptions if not a.get('justification'))
        if unjustified > 0:
            feedback.append(f"{unjustified} assumptions lack justification")

        return {
            'score': avg_score,
            'feedback': feedback,
            'total_assumptions': len(assumptions),
            'high_risk_count': high_risk_count
        }

    async def _score_logical_consistency(self, hypothesis_data: Dict) -> float:
        """Score the logical consistency of the hypothesis."""
        score = 0.5  # Base score

        # Check for contradictions
        statement = hypothesis_data.get('statement', '')
        evidence = hypothesis_data.get('evidence', [])

        # Check if evidence supports or contradicts
        supporting = sum(1 for e in evidence if e.get('supports_hypothesis', False))
        contradicting = sum(1 for e in evidence if not e.get('supports_hypothesis', True))

        if evidence:
            support_ratio = supporting / len(evidence)
            score = 0.3 + (support_ratio * 0.7)

            # Penalty for contradictions
            if contradicting > supporting:
                score *= 0.5

        # Check for circular reasoning indicators
        if 'therefore' in statement.lower() and 'because' in statement.lower():
            score *= 0.9  # Small penalty for potential circular reasoning

        return min(score, 1.0)

    async def _score_novelty(self, statement: str, domain: str) -> float:
        """Score the novelty of the hypothesis."""
        # Simple novelty scoring based on uniqueness indicators
        novelty_keywords = [
            'novel', 'new', 'innovative', 'unprecedented', 'unique',
            'first', 'revolutionary', 'breakthrough', 'paradigm'
        ]

        score = 0.3  # Base novelty

        statement_lower = statement.lower()
        for keyword in novelty_keywords:
            if keyword in statement_lower:
                score += 0.1

        # Domain-specific novelty boost
        emerging_domains = ['quantum', 'ai', 'synthetic biology', 'nanotechnology']
        if any(d in domain.lower() for d in emerging_domains):
            score += 0.2

        # Hash-based uniqueness (simplified)
        statement_hash = hashlib.md5(statement.encode()).hexdigest()
        hash_value = int(statement_hash[:4], 16) / 0xFFFF
        score += hash_value * 0.2

        return min(score, 1.0)

    async def _score_feasibility(self, hypothesis_data: Dict) -> float:
        """Score the feasibility of testing the hypothesis."""
        score = 0.5  # Base feasibility

        # Check resource requirements
        resources = hypothesis_data.get('required_resources', {})
        if resources:
            if resources.get('budget', float('inf')) < 10000:
                score += 0.2
            if resources.get('time_months', float('inf')) < 6:
                score += 0.15
            if resources.get('team_size', float('inf')) < 5:
                score += 0.15

        # Check for methodology clarity
        if hypothesis_data.get('methodology'):
            score += 0.2

        # Check for measurable outcomes
        if hypothesis_data.get('success_criteria'):
            score += 0.1

        return min(score, 1.0)

    async def _score_impact_potential(self, hypothesis_data: Dict) -> float:
        """Score the potential impact of the hypothesis."""
        score = 0.3  # Base impact

        # Check citation potential (simplified)
        domain = hypothesis_data.get('domain', '').lower()
        high_impact_domains = ['medicine', 'climate', 'energy', 'ai safety']
        if any(d in domain for d in high_impact_domains):
            score += 0.3

        # Check scope
        scope = hypothesis_data.get('scope', 'narrow')
        if scope == 'broad':
            score += 0.2
        elif scope == 'moderate':
            score += 0.1

        # Check applications
        applications = hypothesis_data.get('potential_applications', [])
        if len(applications) > 5:
            score += 0.2
        elif len(applications) > 2:
            score += 0.1

        return min(score, 1.0)

    def _calculate_confidence(self, hypothesis_data: Dict) -> float:
        """Calculate confidence based on data completeness."""
        required_fields = [
            'statement', 'evidence', 'methodology', 'assumptions',
            'domain', 'success_criteria'
        ]

        present_fields = sum(1 for field in required_fields
                           if hypothesis_data.get(field))

        return present_fields / len(required_fields)

    async def compare_hypotheses(self,
                                hypotheses: List[Dict]) -> Dict[str, Any]:
        """
        Compare multiple hypotheses and rank them.

        Args:
            hypotheses: List of hypothesis data

        Returns:
            Comparative analysis and ranking
        """
        try:
            scored_hypotheses = []

            for h in hypotheses:
                score = await self.score_hypothesis(h)
                scored_hypotheses.append({
                    'hypothesis_id': h.get('id', hashlib.md5(
                        str(h).encode()).hexdigest()[:8]),
                    'statement': h.get('statement', '')[:100],
                    'overall_score': score.overall_score,
                    'components': score.components,
                    'confidence': score.confidence
                })

            # Sort by overall score
            scored_hypotheses.sort(key=lambda x: x['overall_score'], reverse=True)

            # Statistical analysis
            scores = [h['overall_score'] for h in scored_hypotheses]

            return {
                'rankings': scored_hypotheses,
                'statistics': {
                    'mean_score': float(np.mean(scores)),
                    'std_score': float(np.std(scores)),
                    'min_score': float(np.min(scores)),
                    'max_score': float(np.max(scores)),
                    'median_score': float(np.median(scores))
                },
                'best_hypothesis': scored_hypotheses[0] if scored_hypotheses else None,
                'improvement_opportunities': self._identify_improvement_areas(
                    scored_hypotheses
                )
            }

        except Exception as e:
            logger.error(f"Error comparing hypotheses: {e}")
            return {'error': str(e)}

    def _identify_improvement_areas(self,
                                   scored_hypotheses: List[Dict]) -> List[str]:
        """Identify common areas for improvement across hypotheses."""
        improvements = []

        # Aggregate component scores
        all_components = {}
        for h in scored_hypotheses:
            for component, score in h['components'].items():
                if component not in all_components:
                    all_components[component] = []
                all_components[component].append(score)

        # Find weak areas
        for component, scores in all_components.items():
            avg_score = np.mean(scores)
            if avg_score < 0.5:
                improvements.append(
                    f"Improve {component.replace('_', ' ')} (avg: {avg_score:.2f})"
                )

        return improvements