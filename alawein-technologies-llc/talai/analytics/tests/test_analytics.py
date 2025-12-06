"""
Test suite for TalAI Analytics System
"""

import pytest
import asyncio
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analytics.main import TalAIAnalytics
from analytics.engine.analytics_engine import AnalyticsEngine, ValidationMetrics
from analytics.engine.scoring import QualityScorer
from analytics.ml_insights.embeddings import EmbeddingProcessor
from analytics.ml_insights.classification import DomainClassifier
from analytics.ml_insights.prediction import OutcomePredictor
from analytics.recommendations.recommendation_engine import RecommendationEngine
from analytics.visualization.dashboard import DashboardGenerator


@pytest.fixture
def sample_hypothesis():
    """Sample hypothesis for testing."""
    return {
        'id': 'test_hyp_001',
        'statement': 'Machine learning models can predict protein folding with high accuracy',
        'domain': 'biology',
        'evidence': [
            {
                'source': {'peer_reviewed': True, 'publication_date': '2023-01-15'},
                'quality_score': 0.9,
                'supports_hypothesis': True
            },
            {
                'source': {'peer_reviewed': True, 'publication_date': '2022-08-20'},
                'quality_score': 0.85,
                'supports_hypothesis': True
            },
            {
                'source': {'peer_reviewed': False, 'publication_date': '2023-03-10'},
                'quality_score': 0.7,
                'supports_hypothesis': True
            }
        ],
        'assumptions': [
            {'risk_level': 'low', 'justification': 'Based on AlphaFold success'},
            {'risk_level': 'medium', 'justification': 'Assumes sufficient training data'}
        ],
        'methodology': {
            'research_design': 'computational',
            'sample_size': 1000,
            'statistical_analysis': True
        }
    }


@pytest.fixture
def sample_validation():
    """Sample validation data for testing."""
    return {
        'hypothesis_id': 'test_hyp_001',
        'start_time': datetime.now().isoformat(),
        'end_time': datetime.now().isoformat(),
        'evidence': [
            {'supports_hypothesis': True, 'quality_score': 0.8},
            {'supports_hypothesis': True, 'quality_score': 0.9},
            {'supports_hypothesis': False, 'quality_score': 0.7}
        ],
        'success_rate': 0.67,
        'confidence': 0.75,
        'domain': 'biology',
        'validation_mode': 'standard',
        'duration_seconds': 120
    }


@pytest.mark.asyncio
async def test_analytics_initialization():
    """Test TalAI Analytics initialization."""
    analytics = TalAIAnalytics()
    assert analytics.analytics_engine is not None
    assert analytics.quality_scorer is not None
    assert analytics.embedding_processor is not None
    assert analytics.domain_classifier is not None
    assert analytics.outcome_predictor is not None
    assert analytics.recommendation_engine is not None
    assert analytics.dashboard_generator is not None


@pytest.mark.asyncio
async def test_hypothesis_analysis(sample_hypothesis):
    """Test comprehensive hypothesis analysis."""
    analytics = TalAIAnalytics()
    results = await analytics.analyze_hypothesis(sample_hypothesis)

    assert 'analyses' in results
    assert 'quality' in results['analyses']
    assert 'domain' in results['analyses']
    assert 'prediction' in results['analyses']
    assert 'risk' in results['analyses']

    # Check quality scoring
    quality = results['analyses']['quality']
    assert 'overall_score' in quality
    assert 0 <= quality['overall_score'] <= 1
    assert 'feedback' in quality

    # Check prediction
    prediction = results['analyses']['prediction']
    assert 'success_probability' in prediction or 'will_succeed' in prediction


@pytest.mark.asyncio
async def test_quality_scoring(sample_hypothesis):
    """Test quality scoring functionality."""
    scorer = QualityScorer()
    score = await scorer.score_hypothesis(sample_hypothesis)

    assert score.overall_score >= 0 and score.overall_score <= 1
    assert len(score.components) > 0
    assert score.confidence >= 0 and score.confidence <= 1
    assert isinstance(score.feedback, list)


@pytest.mark.asyncio
async def test_domain_classification():
    """Test domain classification."""
    classifier = DomainClassifier()

    text = "Quantum computing algorithms for cryptography and secure communications"
    result = await classifier.classify_domain(text)

    assert 'primary_domain' in result
    assert result['primary_domain'] in ['physics', 'computer_science', 'general']
    assert 'confidence' in result or 'top_domains' in result


@pytest.mark.asyncio
async def test_outcome_prediction(sample_hypothesis):
    """Test outcome prediction."""
    predictor = OutcomePredictor()
    result = await predictor.predict_hypothesis_success(sample_hypothesis)

    assert 'will_succeed' in result or 'success_probability' in result
    if 'success_probability' in result:
        assert 0 <= result['success_probability'] <= 1
    assert 'recommendations' in result


@pytest.mark.asyncio
async def test_validation_tracking(sample_validation):
    """Test validation event tracking."""
    analytics = TalAIAnalytics()
    result = await analytics.track_validation_event(sample_validation)

    assert 'metrics' in result
    assert 'anomaly_detection' in result
    metrics = result['metrics']
    assert metrics['hypothesis_id'] == sample_validation['hypothesis_id']


@pytest.mark.asyncio
async def test_recommendation_engine(sample_hypothesis):
    """Test recommendation engine."""
    engine = RecommendationEngine()

    # Test validation mode recommendation
    mode_rec = await engine.recommend_validation_mode(sample_hypothesis)
    assert 'recommended_mode' in mode_rec
    assert mode_rec['recommended_mode'] in ['quick', 'standard', 'comprehensive', 'peer_reviewed']

    # Test evidence suggestions
    suggestions = await engine.suggest_evidence_sources(sample_hypothesis)
    assert isinstance(suggestions, list)


@pytest.mark.asyncio
async def test_embedding_similarity():
    """Test embedding-based similarity detection."""
    processor = EmbeddingProcessor()

    hypotheses = [
        {'statement': 'AI can solve climate change', 'id': '1'},
        {'statement': 'Machine learning helps environmental research', 'id': '2'},
        {'statement': 'Quantum physics explains consciousness', 'id': '3'}
    ]

    similar = await processor.find_similar_hypotheses(
        'Artificial intelligence for environmental problems',
        hypotheses,
        top_k=2
    )

    assert len(similar) <= 2
    if similar:
        assert 'similarity_score' in similar[0]
        assert similar[0]['similarity_score'] >= 0 and similar[0]['similarity_score'] <= 1


@pytest.mark.asyncio
async def test_dashboard_generation():
    """Test dashboard generation."""
    generator = DashboardGenerator()

    validation_data = [
        {
            'timestamp': datetime.now().isoformat(),
            'success_rate': 0.8,
            'domain': 'physics',
            'validation_mode': 'standard',
            'duration_seconds': 120,
            'evidence_count': 5,
            'avg_evidence_quality': 0.75
        }
    ]

    dashboard = await generator.generate_validation_dashboard(validation_data)

    assert 'visualizations' in dashboard
    assert 'key_metrics' in dashboard
    assert dashboard['key_metrics']['total_validations'] == 1


@pytest.mark.asyncio
async def test_ab_testing():
    """Test A/B testing functionality."""
    engine = AnalyticsEngine()

    variant_a = [{'success': True}, {'success': False}, {'success': True}]
    variant_b = [{'success': True}, {'success': True}, {'success': True}]

    result = await engine.run_ab_test('test_ab', variant_a, variant_b)

    assert 'winner' in result
    assert 'p_value' in result
    assert 'variant_a' in result
    assert 'variant_b' in result


@pytest.mark.asyncio
async def test_time_series_analysis():
    """Test time series analysis."""
    from analytics.engine.time_series import TimeSeriesAnalyzer

    analyzer = TimeSeriesAnalyzer()

    data = [
        {'timestamp': datetime.now().isoformat(), 'success_rate': 0.7},
        {'timestamp': datetime.now().isoformat(), 'success_rate': 0.8},
        {'timestamp': datetime.now().isoformat(), 'success_rate': 0.75}
    ]

    result = await analyzer.analyze_validation_trends(data)

    assert 'statistics' in result
    assert 'trend' in result


@pytest.mark.asyncio
async def test_insights_report_generation():
    """Test insights report generation."""
    analytics = TalAIAnalytics()

    # Add some test data
    validation = {
        'hypothesis_id': 'test_001',
        'start_time': datetime.now().isoformat(),
        'end_time': datetime.now().isoformat(),
        'evidence': [],
        'success_rate': 0.7,
        'domain': 'test',
        'validation_duration_seconds': 100
    }

    await analytics.track_validation_event(validation)

    report = await analytics.generate_insights_report(time_range_days=7)

    assert 'sections' in report
    assert 'generated_at' in report


@pytest.mark.asyncio
async def test_export_analytics():
    """Test analytics export functionality."""
    analytics = TalAIAnalytics()

    export = await analytics.export_analytics(format='json', include_visualizations=False)

    assert 'content' in export
    assert 'format' in export
    assert export['format'] == 'json'


def test_sync():
    """Test that all async functions work properly."""
    loop = asyncio.get_event_loop()

    async def run_all_tests():
        analytics = TalAIAnalytics()
        hypothesis = {
            'statement': 'Test hypothesis',
            'domain': 'test',
            'evidence': []
        }

        # Test main functions
        await analytics.analyze_hypothesis(hypothesis)
        await analytics.generate_insights_report()

        return True

    result = loop.run_until_complete(run_all_tests())
    assert result is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])