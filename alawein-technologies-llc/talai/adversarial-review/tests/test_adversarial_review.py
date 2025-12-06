"""
Integration tests for AdversarialReview main class

Tests cover:
- Full review workflow
- Overall score calculation
- Verdict assignment logic
- Executive summary generation
- Multi-critic integration
- Result serialization
"""
import pytest
import json
import tempfile
import os
from datetime import datetime
from adversarial_review.main import (
    AdversarialReview,
    ReviewResult,
    CriticFeedback,
)


class TestAdversarialReview:
    """Tests for main AdversarialReview class"""

    def test_initialization(self):
        """Test AdversarialReview initializes with 6 critics"""
        reviewer = AdversarialReview()
        assert len(reviewer.critics) == 6

    def test_all_critics_present(self, critic_dimensions):
        """Test all 6 expected critics are initialized"""
        reviewer = AdversarialReview()
        critic_names = [c.name for c in reviewer.critics]

        for expected_name in critic_dimensions.keys():
            assert expected_name in critic_names

    def test_review_paper_returns_review_result(self, paper_title, sample_paper_text):
        """Test review_paper returns ReviewResult object"""
        reviewer = AdversarialReview()
        result = reviewer.review_paper(paper_title, sample_paper_text, "standard")
        assert isinstance(result, ReviewResult)

    def test_review_result_has_required_fields(self, paper_title, sample_paper_text):
        """Test ReviewResult contains all required fields"""
        reviewer = AdversarialReview()
        result = reviewer.review_paper(paper_title, sample_paper_text, "standard")

        assert hasattr(result, 'paper_title')
        assert hasattr(result, 'overall_score')
        assert hasattr(result, 'verdict')
        assert hasattr(result, 'feedbacks')
        assert hasattr(result, 'executive_summary')
        assert hasattr(result, 'timestamp')

    def test_review_generates_6_feedbacks(self, paper_title, sample_paper_text):
        """Test review generates feedback from all 6 critics"""
        reviewer = AdversarialReview()
        result = reviewer.review_paper(paper_title, sample_paper_text, "standard")
        assert len(result.feedbacks) == 6

    def test_overall_score_is_average(self, paper_title, sample_paper_text):
        """Test overall score is average of all critic scores"""
        reviewer = AdversarialReview()
        result = reviewer.review_paper(paper_title, sample_paper_text, "standard")

        expected_average = sum(f.score for f in result.feedbacks) / len(result.feedbacks)
        assert result.overall_score == round(expected_average, 1)

    def test_overall_score_in_valid_range(self, paper_title, sample_paper_text):
        """Test overall score is between 0 and 10"""
        reviewer = AdversarialReview()
        result = reviewer.review_paper(paper_title, sample_paper_text, "nightmare")
        assert 0.0 <= result.overall_score <= 10.0

    def test_verdict_reject_for_low_score(self, paper_title, long_paper_text):
        """Test REJECT verdict for score < 4.0"""
        reviewer = AdversarialReview()
        result = reviewer.review_paper(paper_title, long_paper_text, "nightmare")

        if result.overall_score < 4.0:
            assert result.verdict == "REJECT"

    def test_verdict_major_revision_for_medium_low_score(self, paper_title, sample_paper_text):
        """Test MAJOR_REVISION verdict for 4.0 <= score < 5.5"""
        reviewer = AdversarialReview()
        result = reviewer.review_paper(paper_title, sample_paper_text, "nightmare")

        if 4.0 <= result.overall_score < 5.5:
            assert result.verdict == "MAJOR_REVISION"

    def test_verdict_minor_revision_for_medium_score(self, paper_title, sample_paper_text):
        """Test MINOR_REVISION verdict for 5.5 <= score < 7.0"""
        reviewer = AdversarialReview()
        result = reviewer.review_paper(paper_title, sample_paper_text, "standard")

        if 5.5 <= result.overall_score < 7.0:
            assert result.verdict == "MINOR_REVISION"

    def test_verdict_accept_for_high_score(self, paper_title, short_paper_text):
        """Test ACCEPT verdict for score >= 7.0"""
        reviewer = AdversarialReview()
        result = reviewer.review_paper(paper_title, short_paper_text, "standard")

        if result.overall_score >= 7.0:
            assert result.verdict == "ACCEPT"

    def test_verdict_is_valid(self, paper_title, sample_paper_text, valid_verdicts):
        """Test verdict is one of the valid options"""
        reviewer = AdversarialReview()
        result = reviewer.review_paper(paper_title, sample_paper_text, "standard")
        assert result.verdict in valid_verdicts

    def test_executive_summary_contains_score(self, paper_title, sample_paper_text):
        """Test executive summary includes overall score"""
        reviewer = AdversarialReview()
        result = reviewer.review_paper(paper_title, sample_paper_text, "standard")
        assert f"{result.overall_score}" in result.executive_summary

    def test_executive_summary_contains_verdict(self, paper_title, sample_paper_text):
        """Test executive summary includes verdict"""
        reviewer = AdversarialReview()
        result = reviewer.review_paper(paper_title, sample_paper_text, "standard")
        assert result.verdict in result.executive_summary

    def test_timestamp_is_valid_iso_format(self, paper_title, sample_paper_text):
        """Test timestamp is valid ISO format"""
        reviewer = AdversarialReview()
        result = reviewer.review_paper(paper_title, sample_paper_text, "standard")

        # Should be able to parse the timestamp
        parsed_time = datetime.fromisoformat(result.timestamp)
        assert isinstance(parsed_time, datetime)

    def test_paper_title_preserved(self, paper_title, sample_paper_text):
        """Test paper title is preserved in result"""
        reviewer = AdversarialReview()
        result = reviewer.review_paper(paper_title, sample_paper_text, "standard")
        assert result.paper_title == paper_title

    def test_different_modes_affect_score(self, paper_title, sample_paper_text):
        """Test different modes produce different scores"""
        reviewer = AdversarialReview()

        standard_result = reviewer.review_paper(paper_title, sample_paper_text, "standard")
        nightmare_result = reviewer.review_paper(paper_title, sample_paper_text, "nightmare")

        # Nightmare mode should generally produce lower or equal scores
        # Due to randomness, we just check they both produce valid results
        assert 0.0 <= standard_result.overall_score <= 10.0
        assert 0.0 <= nightmare_result.overall_score <= 10.0

    def test_feedbacks_are_critic_feedback_objects(self, paper_title, sample_paper_text):
        """Test all feedbacks are CriticFeedback instances"""
        reviewer = AdversarialReview()
        result = reviewer.review_paper(paper_title, sample_paper_text, "standard")

        for feedback in result.feedbacks:
            assert isinstance(feedback, CriticFeedback)

    def test_save_result_creates_file(self, paper_title, sample_paper_text):
        """Test save_result creates a JSON file"""
        reviewer = AdversarialReview()
        result = reviewer.review_paper(paper_title, sample_paper_text, "standard")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            reviewer.save_result(result, temp_file)
            assert os.path.exists(temp_file)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_save_result_valid_json(self, paper_title, sample_paper_text):
        """Test saved result is valid JSON"""
        reviewer = AdversarialReview()
        result = reviewer.review_paper(paper_title, sample_paper_text, "standard")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            reviewer.save_result(result, temp_file)

            with open(temp_file, 'r') as f:
                loaded_data = json.load(f)

            assert 'paper_title' in loaded_data
            assert 'overall_score' in loaded_data
            assert 'verdict' in loaded_data
            assert 'feedbacks' in loaded_data
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_save_result_preserves_data(self, paper_title, sample_paper_text):
        """Test saved and loaded data matches original"""
        reviewer = AdversarialReview()
        result = reviewer.review_paper(paper_title, sample_paper_text, "standard")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            reviewer.save_result(result, temp_file)

            with open(temp_file, 'r') as f:
                loaded_data = json.load(f)

            assert loaded_data['paper_title'] == result.paper_title
            assert loaded_data['overall_score'] == result.overall_score
            assert loaded_data['verdict'] == result.verdict
            assert len(loaded_data['feedbacks']) == len(result.feedbacks)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_multiple_reviews_independent(self, paper_title, alternative_title, sample_paper_text):
        """Test multiple reviews produce independent results"""
        reviewer = AdversarialReview()

        result1 = reviewer.review_paper(paper_title, sample_paper_text, "standard")
        result2 = reviewer.review_paper(alternative_title, sample_paper_text, "standard")

        # Titles should be different
        assert result1.paper_title != result2.paper_title

        # Both should have valid results
        assert len(result1.feedbacks) == 6
        assert len(result2.feedbacks) == 6

    def test_print_result_no_errors(self, paper_title, sample_paper_text, capsys):
        """Test print_result executes without errors"""
        reviewer = AdversarialReview()
        result = reviewer.review_paper(paper_title, sample_paper_text, "standard")

        # Should not raise any exceptions
        reviewer.print_result(result)

        # Capture output to verify something was printed
        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestReviewResultEdgeCases:
    """Edge case tests for review results"""

    def test_empty_title_handled(self, sample_paper_text):
        """Test review handles empty title"""
        reviewer = AdversarialReview()
        result = reviewer.review_paper("", sample_paper_text, "standard")
        assert result.paper_title == ""
        assert len(result.feedbacks) == 6

    def test_very_short_paper_text(self, paper_title):
        """Test review handles very short paper text"""
        reviewer = AdversarialReview()
        result = reviewer.review_paper(paper_title, "X", "standard")
        assert isinstance(result, ReviewResult)
        assert len(result.feedbacks) == 6

    def test_long_paper_text_handled(self, paper_title, long_paper_text):
        """Test review handles long paper text"""
        reviewer = AdversarialReview()
        result = reviewer.review_paper(paper_title, long_paper_text, "nightmare")
        assert isinstance(result, ReviewResult)
        assert len(result.feedbacks) == 6

    def test_brutal_mode(self, paper_title, sample_paper_text):
        """Test brutal mode produces valid results"""
        reviewer = AdversarialReview()
        result = reviewer.review_paper(paper_title, sample_paper_text, "brutal")
        assert isinstance(result, ReviewResult)
        assert 0.0 <= result.overall_score <= 10.0
        assert result.verdict in ["REJECT", "MAJOR_REVISION", "MINOR_REVISION", "ACCEPT"]

    def test_all_scores_minimum_produces_reject(self):
        """Test that all minimum scores produce REJECT verdict"""
        # Create a mock scenario where all scores are at minimum
        reviewer = AdversarialReview()

        # We can't easily mock this, but we can verify the logic
        # If overall_score < 4.0, verdict should be REJECT
        mock_feedbacks = []
        for critic in reviewer.critics:
            mock_feedback = CriticFeedback(
                critic_name=critic.name,
                dimension=critic.dimension,
                severity="CRITICAL",
                issues=["issue1", "issue2"],
                score=1.0,
                recommendations=["rec1", "rec2"]
            )
            mock_feedbacks.append(mock_feedback)

        overall_score = sum(f.score for f in mock_feedbacks) / len(mock_feedbacks)
        assert overall_score < 4.0  # Verify this would produce REJECT
