"""
Tests for ROI calculation functionality
"""
import pytest
from research_pricer.main import ROIPrediction, ResearchPricer


class TestROICalculation:
    """Test ROI calculation logic"""

    def test_calculate_roi_returns_prediction(self, pricer, sample_proposal):
        """Test that calculate_roi returns a ROIPrediction object"""
        prediction = pricer.calculate_roi(sample_proposal.proposal_id)

        assert isinstance(prediction, ROIPrediction)
        assert prediction.proposal_id == sample_proposal.proposal_id

    def test_calculate_roi_nonexistent_proposal_raises_error(self, pricer):
        """Test that calculating ROI for non-existent proposal raises ValueError"""
        with pytest.raises(ValueError, match="Proposal 999 not found"):
            pricer.calculate_roi(999)

    def test_roi_prediction_stored(self, pricer, sample_proposal):
        """Test that ROI predictions are stored in predictions dict"""
        prediction = pricer.calculate_roi(sample_proposal.proposal_id)

        assert sample_proposal.proposal_id in pricer.predictions
        assert pricer.predictions[sample_proposal.proposal_id] == prediction

    def test_roi_persistence(self, temp_data_file, sample_proposal_params):
        """Test that ROI predictions persist across sessions"""
        pricer1 = ResearchPricer(data_file=temp_data_file)
        proposal = pricer1.submit_proposal(**sample_proposal_params)
        prediction1 = pricer1.calculate_roi(proposal.proposal_id)

        # Load in new instance
        pricer2 = ResearchPricer(data_file=temp_data_file)

        assert proposal.proposal_id in pricer2.predictions
        assert pricer2.predictions[proposal.proposal_id].total_roi_percent == prediction1.total_roi_percent

    def test_publication_metrics_calculated(self, pricer, sample_proposal):
        """Test that publication metrics are calculated"""
        prediction = pricer.calculate_roi(sample_proposal.proposal_id)

        assert prediction.expected_publications > 0
        assert prediction.journal_quality_avg > 0
        assert prediction.conf_presentations > 0
        assert prediction.time_to_first_pub > 0

    def test_citation_metrics_calculated(self, pricer, sample_proposal):
        """Test that citation metrics are calculated"""
        prediction = pricer.calculate_roi(sample_proposal.proposal_id)

        assert prediction.citations_1yr >= 0
        assert prediction.citations_3yr >= 0
        assert prediction.citations_5yr >= 0
        assert prediction.h_index_increase >= 0

    def test_citation_metrics_increase_over_time(self, pricer, sample_proposal):
        """Test that citations increase over time periods"""
        prediction = pricer.calculate_roi(sample_proposal.proposal_id)

        # 5 year citations should be higher than 3 year, which should be higher than 1 year
        assert prediction.citations_5yr >= prediction.citations_3yr
        assert prediction.citations_3yr >= prediction.citations_1yr

    def test_career_impact_calculated(self, pricer, sample_proposal):
        """Test that career impact metrics are calculated"""
        prediction = pricer.calculate_roi(sample_proposal.proposal_id)

        assert 0 <= prediction.tenure_probability <= 1
        assert 0 <= prediction.promotion_probability <= 1
        assert prediction.future_grants_expected > 0
        assert prediction.collaborations_formed > 0

    def test_economic_impact_calculated(self, pricer, sample_proposal):
        """Test that economic impact metrics are calculated"""
        prediction = pricer.calculate_roi(sample_proposal.proposal_id)

        assert prediction.total_roi_percent is not None
        assert prediction.patent_value >= 0
        assert prediction.commercialization_potential >= 0
        assert prediction.industry_partnerships >= 0

    def test_risk_assessment_calculated(self, pricer, sample_proposal):
        """Test that risk assessment is calculated"""
        prediction = pricer.calculate_roi(sample_proposal.proposal_id)

        assert 0 <= prediction.failure_probability <= 1
        assert prediction.risk_level in ['low', 'medium', 'high']

    def test_overall_scores_in_valid_range(self, pricer, sample_proposal):
        """Test that overall scores are within valid range"""
        prediction = pricer.calculate_roi(sample_proposal.proposal_id)

        assert 0 <= prediction.academic_value_score <= 100
        assert 0 <= prediction.commercial_value_score <= 100
        assert 0 <= prediction.career_value_score <= 100

    def test_recommendation_is_valid(self, pricer, sample_proposal):
        """Test that recommendation is one of valid values"""
        prediction = pricer.calculate_roi(sample_proposal.proposal_id)

        valid_recommendations = ['strongly_recommend', 'recommend', 'neutral', 'not_recommend']
        assert prediction.overall_recommendation in valid_recommendations

    def test_explanation_generated(self, pricer, sample_proposal):
        """Test that explanation text is generated"""
        prediction = pricer.calculate_roi(sample_proposal.proposal_id)

        assert isinstance(prediction.explanation, str)
        assert len(prediction.explanation) > 0

    def test_comparable_grants_included(self, pricer, sample_proposal):
        """Test that comparable grants are included"""
        prediction = pricer.calculate_roi(sample_proposal.proposal_id)

        assert isinstance(prediction.comparable_grants, list)
        assert len(prediction.comparable_grants) == 3

    def test_confidence_intervals_included(self, pricer, sample_proposal):
        """Test that confidence intervals are included"""
        prediction = pricer.calculate_roi(sample_proposal.proposal_id)

        assert 'publications_low' in prediction.confidence_interval
        assert 'publications_high' in prediction.confidence_interval
        assert 'citations_low' in prediction.confidence_interval
        assert 'citations_high' in prediction.confidence_interval
        assert 'roi_low' in prediction.confidence_interval
        assert 'roi_high' in prediction.confidence_interval

    def test_confidence_intervals_valid_range(self, pricer, sample_proposal):
        """Test that confidence intervals show proper ranges"""
        prediction = pricer.calculate_roi(sample_proposal.proposal_id)

        ci = prediction.confidence_interval
        assert ci['publications_low'] < ci['publications_high']
        assert ci['citations_low'] < ci['citations_high']
