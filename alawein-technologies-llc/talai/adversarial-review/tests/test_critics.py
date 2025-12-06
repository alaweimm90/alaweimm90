"""
Comprehensive test suite for all 6 AdversarialReview critics

Tests cover:
- Initialization
- Review output structure
- Score calculation logic
- Severity assignment
- Mode-dependent behavior
- Business logic rules
"""
import pytest
from adversarial_review.main import (
    StatisticalCritic,
    MethodologicalCritic,
    LogicalCritic,
    HistoricalCritic,
    EthicalCritic,
    EconomicCritic,
    CriticFeedback,
    AdversarialCritic
)


# ============================================================================
# StatisticalCritic Tests
# ============================================================================

class TestStatisticalCritic:
    """Tests for StatisticalCritic"""

    def test_initialization(self):
        """Test critic initializes with correct name and dimension"""
        critic = StatisticalCritic()
        assert critic.name == "Statistical Skeptic"
        assert critic.dimension == "statistical_validity"

    def test_review_returns_critic_feedback(self, sample_paper_text, normal_mode):
        """Test review returns CriticFeedback object"""
        critic = StatisticalCritic()
        result = critic.review(sample_paper_text, normal_mode)
        assert isinstance(result, CriticFeedback)

    def test_review_includes_required_fields(self, sample_paper_text, normal_mode):
        """Test review includes all required feedback fields"""
        critic = StatisticalCritic()
        result = critic.review(sample_paper_text, normal_mode)

        assert result.critic_name == "Statistical Skeptic"
        assert result.dimension == "statistical_validity"
        assert result.severity in ["CRITICAL", "MAJOR", "MINOR"]
        assert isinstance(result.issues, list)
        assert isinstance(result.score, float)
        assert isinstance(result.recommendations, list)

    def test_review_normal_mode_issue_count(self, sample_paper_text, normal_mode):
        """Test normal mode generates expected number of issues (3)"""
        critic = StatisticalCritic()
        result = critic.review(sample_paper_text, normal_mode)
        assert len(result.issues) == 3
        assert len(result.recommendations) == 3

    def test_review_nightmare_mode_issue_count(self, sample_paper_text, nightmare_mode):
        """Test nightmare mode generates more issues (5)"""
        critic = StatisticalCritic()
        result = critic.review(sample_paper_text, nightmare_mode)
        assert len(result.issues) == 5
        assert len(result.recommendations) == 5

    def test_score_within_valid_range(self, sample_paper_text, normal_mode):
        """Test score is between 0 and 10"""
        critic = StatisticalCritic()
        result = critic.review(sample_paper_text, normal_mode)
        assert 0.0 <= result.score <= 10.0

    def test_severity_matches_score_critical(self, sample_paper_text):
        """Test CRITICAL severity when score < 4"""
        critic = StatisticalCritic()
        # Nightmare mode typically produces low scores
        result = critic.review(sample_paper_text, "nightmare")
        if result.score < 4:
            assert result.severity == "CRITICAL"

    def test_severity_matches_score_major(self, sample_paper_text):
        """Test MAJOR severity when 4 <= score < 6"""
        critic = StatisticalCritic()
        result = critic.review(sample_paper_text, "standard")
        if 4 <= result.score < 6:
            assert result.severity == "MAJOR"

    def test_severity_matches_score_minor(self, sample_paper_text):
        """Test MINOR severity when score >= 6"""
        critic = StatisticalCritic()
        result = critic.review(sample_paper_text, "standard")
        if result.score >= 6:
            assert result.severity == "MINOR"

    def test_issues_are_strings(self, sample_paper_text, normal_mode):
        """Test all issues are non-empty strings"""
        critic = StatisticalCritic()
        result = critic.review(sample_paper_text, normal_mode)
        assert all(isinstance(issue, str) and len(issue) > 0 for issue in result.issues)

    def test_recommendations_are_strings(self, sample_paper_text, normal_mode):
        """Test all recommendations are non-empty strings"""
        critic = StatisticalCritic()
        result = critic.review(sample_paper_text, normal_mode)
        assert all(isinstance(rec, str) and len(rec) > 0 for rec in result.recommendations)


# ============================================================================
# MethodologicalCritic Tests
# ============================================================================

class TestMethodologicalCritic:
    """Tests for MethodologicalCritic"""

    def test_initialization(self):
        """Test critic initializes with correct name and dimension"""
        critic = MethodologicalCritic()
        assert critic.name == "Methodology Maverick"
        assert critic.dimension == "methodological_rigor"

    def test_review_returns_critic_feedback(self, sample_paper_text, normal_mode):
        """Test review returns CriticFeedback object"""
        critic = MethodologicalCritic()
        result = critic.review(sample_paper_text, normal_mode)
        assert isinstance(result, CriticFeedback)

    def test_review_normal_mode_issue_count(self, sample_paper_text, normal_mode):
        """Test normal mode generates 4 issues"""
        critic = MethodologicalCritic()
        result = critic.review(sample_paper_text, normal_mode)
        assert len(result.issues) == 4
        assert len(result.recommendations) == 4

    def test_review_nightmare_mode_issue_count(self, sample_paper_text, nightmare_mode):
        """Test nightmare mode generates 6 issues"""
        critic = MethodologicalCritic()
        result = critic.review(sample_paper_text, nightmare_mode)
        assert len(result.issues) == 6
        assert len(result.recommendations) == 6

    def test_score_decreases_with_more_issues(self, sample_paper_text):
        """Test nightmare mode produces lower score than normal mode"""
        critic = MethodologicalCritic()
        normal_result = critic.review(sample_paper_text, "standard")
        nightmare_result = critic.review(sample_paper_text, "nightmare")

        # Nightmare mode should generally have lower scores due to more issues
        assert nightmare_result.score <= normal_result.score + 0.1  # Small tolerance for randomness

    def test_score_minimum_threshold(self, sample_paper_text, nightmare_mode):
        """Test score respects minimum threshold (1.5)"""
        critic = MethodologicalCritic()
        result = critic.review(sample_paper_text, nightmare_mode)
        assert result.score >= 1.5

    def test_required_fields_present(self, sample_paper_text, normal_mode):
        """Test all required fields are present"""
        critic = MethodologicalCritic()
        result = critic.review(sample_paper_text, normal_mode)

        assert hasattr(result, 'critic_name')
        assert hasattr(result, 'dimension')
        assert hasattr(result, 'severity')
        assert hasattr(result, 'issues')
        assert hasattr(result, 'score')
        assert hasattr(result, 'recommendations')


# ============================================================================
# LogicalCritic Tests
# ============================================================================

class TestLogicalCritic:
    """Tests for LogicalCritic"""

    def test_initialization(self):
        """Test critic initializes with correct name and dimension"""
        critic = LogicalCritic()
        assert critic.name == "Logic Enforcer"
        assert critic.dimension == "logical_consistency"

    def test_review_returns_critic_feedback(self, sample_paper_text, normal_mode):
        """Test review returns CriticFeedback object"""
        critic = LogicalCritic()
        result = critic.review(sample_paper_text, normal_mode)
        assert isinstance(result, CriticFeedback)

    def test_review_normal_mode_issue_count(self, sample_paper_text, normal_mode):
        """Test normal mode generates 2 issues"""
        critic = LogicalCritic()
        result = critic.review(sample_paper_text, normal_mode)
        assert len(result.issues) == 2
        assert len(result.recommendations) == 2

    def test_review_nightmare_mode_issue_count(self, sample_paper_text, nightmare_mode):
        """Test nightmare mode generates 4 issues"""
        critic = LogicalCritic()
        result = critic.review(sample_paper_text, nightmare_mode)
        assert len(result.issues) == 4
        assert len(result.recommendations) == 4

    def test_score_minimum_threshold(self, sample_paper_text, nightmare_mode):
        """Test score respects minimum threshold (2.0)"""
        critic = LogicalCritic()
        result = critic.review(sample_paper_text, nightmare_mode)
        assert result.score >= 2.0

    def test_score_calculation_logic(self, sample_paper_text, normal_mode):
        """Test score calculation follows expected formula"""
        critic = LogicalCritic()
        result = critic.review(sample_paper_text, normal_mode)

        # Score should be 8.5 - (num_issues * 1.5), but at least 2.0
        expected_score = max(2.0, 8.5 - len(result.issues) * 1.5)
        assert result.score == round(expected_score, 1)

    def test_issues_match_recommendations_count(self, sample_paper_text, normal_mode):
        """Test issues and recommendations have same count"""
        critic = LogicalCritic()
        result = critic.review(sample_paper_text, normal_mode)
        assert len(result.issues) == len(result.recommendations)


# ============================================================================
# HistoricalCritic Tests
# ============================================================================

class TestHistoricalCritic:
    """Tests for HistoricalCritic"""

    def test_initialization(self):
        """Test critic initializes with correct name and dimension"""
        critic = HistoricalCritic()
        assert critic.name == "History Hunter"
        assert critic.dimension == "historical_context"

    def test_review_returns_critic_feedback(self, sample_paper_text, normal_mode):
        """Test review returns CriticFeedback object"""
        critic = HistoricalCritic()
        result = critic.review(sample_paper_text, normal_mode)
        assert isinstance(result, CriticFeedback)

    def test_review_normal_mode_issue_count(self, sample_paper_text, normal_mode):
        """Test normal mode generates 3 issues"""
        critic = HistoricalCritic()
        result = critic.review(sample_paper_text, normal_mode)
        assert len(result.issues) == 3
        assert len(result.recommendations) == 3

    def test_review_nightmare_mode_issue_count(self, sample_paper_text, nightmare_mode):
        """Test nightmare mode generates 4 issues"""
        critic = HistoricalCritic()
        result = critic.review(sample_paper_text, nightmare_mode)
        assert len(result.issues) == 4
        assert len(result.recommendations) == 4

    def test_score_minimum_threshold(self, sample_paper_text, nightmare_mode):
        """Test score respects minimum threshold (2.5)"""
        critic = HistoricalCritic()
        result = critic.review(sample_paper_text, nightmare_mode)
        assert result.score >= 2.5

    def test_dimension_is_historical_context(self, sample_paper_text, normal_mode):
        """Test dimension is correctly set"""
        critic = HistoricalCritic()
        result = critic.review(sample_paper_text, normal_mode)
        assert result.dimension == "historical_context"


# ============================================================================
# EthicalCritic Tests
# ============================================================================

class TestEthicalCritic:
    """Tests for EthicalCritic"""

    def test_initialization(self):
        """Test critic initializes with correct name and dimension"""
        critic = EthicalCritic()
        assert critic.name == "Ethics Enforcer"
        assert critic.dimension == "ethical_implications"

    def test_review_returns_critic_feedback(self, sample_paper_text, normal_mode):
        """Test review returns CriticFeedback object"""
        critic = EthicalCritic()
        result = critic.review(sample_paper_text, normal_mode)
        assert isinstance(result, CriticFeedback)

    def test_review_normal_mode_issue_count(self, sample_paper_text, normal_mode):
        """Test normal mode generates 2 issues"""
        critic = EthicalCritic()
        result = critic.review(sample_paper_text, normal_mode)
        assert len(result.issues) == 2
        assert len(result.recommendations) == 2

    def test_review_nightmare_mode_issue_count(self, sample_paper_text, nightmare_mode):
        """Test nightmare mode generates 3 issues"""
        critic = EthicalCritic()
        result = critic.review(sample_paper_text, nightmare_mode)
        assert len(result.issues) == 3
        assert len(result.recommendations) == 3

    def test_score_minimum_threshold(self, sample_paper_text, nightmare_mode):
        """Test score respects minimum threshold (3.0)"""
        critic = EthicalCritic()
        result = critic.review(sample_paper_text, nightmare_mode)
        assert result.score >= 3.0

    def test_score_calculation_with_issues(self, sample_paper_text, nightmare_mode):
        """Test score decreases appropriately with issues"""
        critic = EthicalCritic()
        result = critic.review(sample_paper_text, nightmare_mode)

        # Score should be 8.0 - (num_issues * 1.5), but at least 3.0
        expected_score = max(3.0, 8.0 - len(result.issues) * 1.5)
        assert result.score == round(expected_score, 1)

    def test_severity_assignment(self, sample_paper_text, normal_mode):
        """Test severity is assigned correctly based on score"""
        critic = EthicalCritic()
        result = critic.review(sample_paper_text, normal_mode)

        if result.score < 4:
            assert result.severity == "CRITICAL"
        elif result.score < 6:
            assert result.severity == "MAJOR"
        else:
            assert result.severity == "MINOR"


# ============================================================================
# EconomicCritic Tests
# ============================================================================

class TestEconomicCritic:
    """Tests for EconomicCritic"""

    def test_initialization(self):
        """Test critic initializes with correct name and dimension"""
        critic = EconomicCritic()
        assert critic.name == "Economic Realist"
        assert critic.dimension == "economic_feasibility"

    def test_review_returns_critic_feedback(self, sample_paper_text, normal_mode):
        """Test review returns CriticFeedback object"""
        critic = EconomicCritic()
        result = critic.review(sample_paper_text, normal_mode)
        assert isinstance(result, CriticFeedback)

    def test_review_normal_mode_issue_count(self, sample_paper_text, normal_mode):
        """Test normal mode generates 2 issues"""
        critic = EconomicCritic()
        result = critic.review(sample_paper_text, normal_mode)
        assert len(result.issues) == 2
        assert len(result.recommendations) == 2

    def test_review_nightmare_mode_issue_count(self, sample_paper_text, nightmare_mode):
        """Test nightmare mode generates 3 issues"""
        critic = EconomicCritic()
        result = critic.review(sample_paper_text, nightmare_mode)
        assert len(result.issues) == 3
        assert len(result.recommendations) == 3

    def test_score_minimum_threshold(self, sample_paper_text, nightmare_mode):
        """Test score respects minimum threshold (3.5)"""
        critic = EconomicCritic()
        result = critic.review(sample_paper_text, nightmare_mode)
        assert result.score >= 3.5

    def test_score_calculation_logic(self, sample_paper_text, normal_mode):
        """Test score calculation follows expected formula"""
        critic = EconomicCritic()
        result = critic.review(sample_paper_text, normal_mode)

        # Score should be 7.5 - (num_issues * 1.0), but at least 3.5
        expected_score = max(3.5, 7.5 - len(result.issues) * 1.0)
        assert result.score == round(expected_score, 1)

    def test_different_paper_texts_produce_valid_results(self, short_paper_text, long_paper_text, normal_mode):
        """Test critic works with different paper lengths"""
        critic = EconomicCritic()

        short_result = critic.review(short_paper_text, normal_mode)
        long_result = critic.review(long_paper_text, normal_mode)

        assert isinstance(short_result, CriticFeedback)
        assert isinstance(long_result, CriticFeedback)
        assert short_result.score >= 3.5
        assert long_result.score >= 3.5


# ============================================================================
# Cross-Critic Tests
# ============================================================================

class TestCrossCriticBehavior:
    """Tests that verify consistent behavior across all critics"""

    @pytest.mark.parametrize("critic_class,expected_name,expected_dimension", [
        (StatisticalCritic, "Statistical Skeptic", "statistical_validity"),
        (MethodologicalCritic, "Methodology Maverick", "methodological_rigor"),
        (LogicalCritic, "Logic Enforcer", "logical_consistency"),
        (HistoricalCritic, "History Hunter", "historical_context"),
        (EthicalCritic, "Ethics Enforcer", "ethical_implications"),
        (EconomicCritic, "Economic Realist", "economic_feasibility"),
    ])
    def test_all_critics_initialize_correctly(self, critic_class, expected_name, expected_dimension):
        """Test all critics initialize with correct name and dimension"""
        critic = critic_class()
        assert critic.name == expected_name
        assert critic.dimension == expected_dimension

    @pytest.mark.parametrize("critic_class", [
        StatisticalCritic,
        MethodologicalCritic,
        LogicalCritic,
        HistoricalCritic,
        EthicalCritic,
        EconomicCritic,
    ])
    def test_all_critics_return_valid_feedback(self, critic_class, sample_paper_text, normal_mode):
        """Test all critics return valid CriticFeedback"""
        critic = critic_class()
        result = critic.review(sample_paper_text, normal_mode)

        assert isinstance(result, CriticFeedback)
        assert isinstance(result.issues, list)
        assert isinstance(result.recommendations, list)
        assert len(result.issues) > 0
        assert len(result.recommendations) > 0

    @pytest.mark.parametrize("critic_class", [
        StatisticalCritic,
        MethodologicalCritic,
        LogicalCritic,
        HistoricalCritic,
        EthicalCritic,
        EconomicCritic,
    ])
    def test_all_critics_respect_score_bounds(self, critic_class, sample_paper_text, nightmare_mode):
        """Test all critics produce scores between 0 and 10"""
        critic = critic_class()
        result = critic.review(sample_paper_text, nightmare_mode)
        assert 0.0 <= result.score <= 10.0

    @pytest.mark.parametrize("critic_class", [
        StatisticalCritic,
        MethodologicalCritic,
        LogicalCritic,
        HistoricalCritic,
        EthicalCritic,
        EconomicCritic,
    ])
    def test_all_critics_nightmare_mode_more_issues(self, critic_class, sample_paper_text):
        """Test nightmare mode produces more or equal issues than normal mode"""
        critic = critic_class()
        normal_result = critic.review(sample_paper_text, "standard")
        nightmare_result = critic.review(sample_paper_text, "nightmare")

        assert len(nightmare_result.issues) >= len(normal_result.issues)

    @pytest.mark.parametrize("critic_class", [
        StatisticalCritic,
        MethodologicalCritic,
        LogicalCritic,
        HistoricalCritic,
        EthicalCritic,
        EconomicCritic,
    ])
    def test_all_critics_issue_recommendation_parity(self, critic_class, sample_paper_text, normal_mode):
        """Test all critics produce equal number of issues and recommendations"""
        critic = critic_class()
        result = critic.review(sample_paper_text, normal_mode)
        assert len(result.issues) == len(result.recommendations)
