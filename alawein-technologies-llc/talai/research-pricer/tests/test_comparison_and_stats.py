"""
Tests for proposal comparison and domain statistics
"""
import pytest
from research_pricer.main import ResearchPricer


class TestProposalComparison:
    """Test proposal comparison functionality"""

    def test_compare_proposals_returns_list(self, populated_pricer):
        """Test that compare_proposals returns a list"""
        # Calculate ROI for all proposals
        for pid in populated_pricer.proposals.keys():
            populated_pricer.calculate_roi(pid)

        comparisons = populated_pricer.compare_proposals([1, 2])

        assert isinstance(comparisons, list)
        assert len(comparisons) == 2

    def test_compare_proposals_includes_required_fields(self, populated_pricer):
        """Test that comparison results include all required fields"""
        for pid in populated_pricer.proposals.keys():
            populated_pricer.calculate_roi(pid)

        comparisons = populated_pricer.compare_proposals([1, 2])
        comparison = comparisons[0]

        required_fields = [
            'proposal_id', 'title', 'funding', 'roi_percent',
            'publications', 'citations_5yr', 'academic_value',
            'commercial_value', 'career_value', 'risk_level',
            'recommendation'
        ]

        for field in required_fields:
            assert field in comparison

    def test_compare_proposals_sorted_by_roi(self, populated_pricer):
        """Test that comparisons are sorted by ROI in descending order"""
        for pid in populated_pricer.proposals.keys():
            populated_pricer.calculate_roi(pid)

        comparisons = populated_pricer.compare_proposals([1, 2, 3])

        # Verify sorted in descending order
        for i in range(len(comparisons) - 1):
            assert comparisons[i]['roi_percent'] >= comparisons[i + 1]['roi_percent']

    def test_compare_proposals_missing_prediction_raises_error(self, populated_pricer):
        """Test that comparing proposals without predictions raises error"""
        with pytest.raises(ValueError, match="Predictions not found"):
            populated_pricer.compare_proposals([1, 2, 3])

    def test_compare_single_proposal(self, populated_pricer):
        """Test comparing a single proposal"""
        populated_pricer.calculate_roi(1)

        comparisons = populated_pricer.compare_proposals([1])

        assert len(comparisons) == 1
        assert comparisons[0]['proposal_id'] == 1

    def test_compare_proposals_with_partial_predictions(self, populated_pricer):
        """Test error when some proposals lack predictions"""
        populated_pricer.calculate_roi(1)
        # Don't calculate ROI for proposal 2

        with pytest.raises(ValueError) as exc_info:
            populated_pricer.compare_proposals([1, 2])

        assert "2" in str(exc_info.value)


class TestDomainStatistics:
    """Test domain statistics functionality"""

    def test_domain_statistics_returns_dict(self, populated_pricer):
        """Test that domain statistics returns a dictionary"""
        for pid in populated_pricer.proposals.keys():
            populated_pricer.calculate_roi(pid)

        stats = populated_pricer.get_domain_statistics('Computer Science')

        assert isinstance(stats, dict)

    def test_domain_statistics_includes_required_fields(self, populated_pricer):
        """Test that domain statistics include all required fields"""
        for pid in populated_pricer.proposals.keys():
            populated_pricer.calculate_roi(pid)

        stats = populated_pricer.get_domain_statistics('Computer Science')

        required_fields = [
            'domain', 'total_proposals', 'avg_roi', 'avg_publications',
            'avg_citations_5yr', 'avg_academic_value', 'avg_commercial_value',
            'high_risk_percent', 'strongly_recommended'
        ]

        for field in required_fields:
            assert field in stats

    def test_domain_statistics_correct_count(self, populated_pricer):
        """Test that domain statistics count proposals correctly"""
        for pid in populated_pricer.proposals.keys():
            populated_pricer.calculate_roi(pid)

        stats = populated_pricer.get_domain_statistics('Computer Science')

        # Should have 2 Computer Science proposals (from conftest)
        assert stats['total_proposals'] == 2

    def test_domain_statistics_averages_calculated(self, populated_pricer):
        """Test that averages are calculated correctly"""
        for pid in populated_pricer.proposals.keys():
            populated_pricer.calculate_roi(pid)

        stats = populated_pricer.get_domain_statistics('Computer Science')

        assert stats['avg_roi'] > 0
        assert stats['avg_publications'] > 0
        assert stats['avg_citations_5yr'] > 0
        assert stats['avg_academic_value'] > 0

    def test_domain_statistics_no_proposals_returns_error(self, populated_pricer):
        """Test that requesting stats for non-existent domain returns error"""
        stats = populated_pricer.get_domain_statistics('Nonexistent Domain')

        assert 'error' in stats
        assert 'Nonexistent Domain' in stats['error']

    def test_domain_statistics_high_risk_percentage(self, pricer, high_risk_proposal_params):
        """Test high risk percentage calculation"""
        # Submit multiple proposals with varying risk
        for i in range(3):
            params = high_risk_proposal_params.copy()
            params['title'] = f'Proposal {i}'
            params['pi_name'] = f'PI {i}'
            pricer.submit_proposal(**params)

        for pid in pricer.proposals.keys():
            pricer.calculate_roi(pid)

        stats = pricer.get_domain_statistics('Biology')

        assert 'high_risk_percent' in stats
        assert 0 <= stats['high_risk_percent'] <= 100

    def test_domain_statistics_empty_predictions(self, populated_pricer):
        """Test domain statistics when no ROI calculated yet"""
        # Don't calculate any ROI
        stats = populated_pricer.get_domain_statistics('Computer Science')

        # Should return error since no predictions exist
        assert 'error' in stats


class TestDataPersistence:
    """Test data loading and saving functionality"""

    def test_save_and_load_proposals(self, temp_data_file, sample_proposal_params):
        """Test that proposals are saved and loaded correctly"""
        pricer1 = ResearchPricer(data_file=temp_data_file)
        proposal = pricer1.submit_proposal(**sample_proposal_params)

        pricer2 = ResearchPricer(data_file=temp_data_file)

        assert len(pricer2.proposals) == 1
        loaded_proposal = pricer2.proposals[proposal.proposal_id]
        assert loaded_proposal.title == sample_proposal_params['title']
        assert loaded_proposal.funding_amount == sample_proposal_params['funding_amount']

    def test_save_and_load_predictions(self, temp_data_file, sample_proposal_params):
        """Test that predictions are saved and loaded correctly"""
        pricer1 = ResearchPricer(data_file=temp_data_file)
        proposal = pricer1.submit_proposal(**sample_proposal_params)
        prediction = pricer1.calculate_roi(proposal.proposal_id)

        pricer2 = ResearchPricer(data_file=temp_data_file)

        assert len(pricer2.predictions) == 1
        loaded_prediction = pricer2.predictions[proposal.proposal_id]
        assert loaded_prediction.total_roi_percent == prediction.total_roi_percent
        assert loaded_prediction.expected_publications == prediction.expected_publications

    def test_load_nonexistent_file(self, temp_data_file):
        """Test loading when file doesn't exist creates empty pricer"""
        import os
        if os.path.exists(temp_data_file):
            os.remove(temp_data_file)

        pricer = ResearchPricer(data_file=temp_data_file)

        assert len(pricer.proposals) == 0
        assert len(pricer.predictions) == 0

    def test_incremental_updates(self, temp_data_file, sample_proposal_params, low_impact_proposal_params):
        """Test that incremental updates work correctly"""
        pricer1 = ResearchPricer(data_file=temp_data_file)
        proposal1 = pricer1.submit_proposal(**sample_proposal_params)

        pricer2 = ResearchPricer(data_file=temp_data_file)
        proposal2 = pricer2.submit_proposal(**low_impact_proposal_params)

        pricer3 = ResearchPricer(data_file=temp_data_file)

        assert len(pricer3.proposals) == 2
        assert proposal1.proposal_id in pricer3.proposals
        assert proposal2.proposal_id in pricer3.proposals
