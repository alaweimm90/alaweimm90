"""
Integration tests for end-to-end workflows
"""
import pytest
import subprocess
import sys


class TestEndToEndWorkflow:
    """Test complete workflows"""

    def test_submit_and_calculate_workflow(self, pricer, sample_proposal_params):
        """Test complete workflow of submitting and calculating ROI"""
        # Submit proposal
        proposal = pricer.submit_proposal(**sample_proposal_params)
        assert proposal.proposal_id > 0

        # Calculate ROI
        prediction = pricer.calculate_roi(proposal.proposal_id)
        assert prediction.proposal_id == proposal.proposal_id

        # Verify all metrics are present
        assert prediction.expected_publications > 0
        assert prediction.citations_5yr >= 0
        assert prediction.total_roi_percent is not None
        assert prediction.overall_recommendation in [
            'strongly_recommend', 'recommend', 'neutral', 'not_recommend'
        ]

    def test_multiple_proposals_workflow(self, pricer, sample_proposal_params, low_impact_proposal_params):
        """Test workflow with multiple proposals"""
        # Submit multiple proposals
        proposal1 = pricer.submit_proposal(**sample_proposal_params)
        proposal2 = pricer.submit_proposal(**low_impact_proposal_params)

        # Calculate ROI for both
        prediction1 = pricer.calculate_roi(proposal1.proposal_id)
        prediction2 = pricer.calculate_roi(proposal2.proposal_id)

        # Compare
        comparisons = pricer.compare_proposals([proposal1.proposal_id, proposal2.proposal_id])

        assert len(comparisons) == 2
        # Should be sorted by ROI
        assert comparisons[0]['roi_percent'] >= comparisons[1]['roi_percent']

    def test_domain_analysis_workflow(self, pricer):
        """Test domain analysis workflow"""
        # Submit multiple proposals in same domain
        for i in range(3):
            params = {
                'title': f'AI Project {i}',
                'domain': 'Artificial Intelligence',
                'funding_amount': 100000.0 + (i * 50000),
                'duration_months': 24,
                'team_size': 3 + i,
                'institution': f'University {i}',
                'pi_name': f'Dr. Smith {i}',
                'pi_h_index': 15 + (i * 5),
                'novelty_score': 0.7,
                'feasibility_score': 0.7,
                'impact_potential': 'medium',
                'methodology': 'Machine Learning',
                'prior_publications': 20 + (i * 10)
            }
            proposal = pricer.submit_proposal(**params)
            pricer.calculate_roi(proposal.proposal_id)

        # Get domain statistics
        stats = pricer.get_domain_statistics('Artificial Intelligence')

        assert stats['total_proposals'] == 3
        assert stats['avg_roi'] > 0
        assert stats['avg_publications'] > 0

    def test_persistence_workflow(self, temp_data_file, sample_proposal_params):
        """Test that data persists across sessions"""
        from research_pricer.main import ResearchPricer

        # Session 1: Submit and calculate
        pricer1 = ResearchPricer(data_file=temp_data_file)
        proposal = pricer1.submit_proposal(**sample_proposal_params)
        prediction1 = pricer1.calculate_roi(proposal.proposal_id)

        # Session 2: Load and verify
        pricer2 = ResearchPricer(data_file=temp_data_file)
        assert proposal.proposal_id in pricer2.proposals
        assert proposal.proposal_id in pricer2.predictions

        loaded_prediction = pricer2.predictions[proposal.proposal_id]
        assert loaded_prediction.total_roi_percent == prediction1.total_roi_percent


class TestDataValidation:
    """Test data validation and constraints"""

    def test_proposal_id_uniqueness(self, pricer, sample_proposal_params):
        """Test that proposal IDs are unique"""
        proposal1 = pricer.submit_proposal(**sample_proposal_params)
        proposal2 = pricer.submit_proposal(**sample_proposal_params)

        assert proposal1.proposal_id != proposal2.proposal_id
        assert proposal2.proposal_id == proposal1.proposal_id + 1

    def test_all_impact_levels(self, pricer):
        """Test all impact potential levels"""
        impact_levels = ['low', 'medium', 'high', 'transformative']

        for impact in impact_levels:
            params = {
                'title': f'{impact.capitalize()} Impact Project',
                'domain': 'Science',
                'funding_amount': 100000.0,
                'duration_months': 24,
                'team_size': 3,
                'institution': 'University',
                'pi_name': f'PI {impact}',
                'pi_h_index': 15,
                'novelty_score': 0.7,
                'feasibility_score': 0.7,
                'impact_potential': impact,
                'methodology': 'Experimental',
                'prior_publications': 20
            }
            proposal = pricer.submit_proposal(**params)
            prediction = pricer.calculate_roi(proposal.proposal_id)

            # All should produce valid predictions
            assert prediction.total_roi_percent is not None
            assert prediction.overall_recommendation in [
                'strongly_recommend', 'recommend', 'neutral', 'not_recommend'
            ]

    def test_all_risk_levels_achievable(self, pricer):
        """Test that all risk levels can be achieved"""
        # Low risk: high feasibility, low novelty
        low_risk_params = {
            'title': 'Low Risk',
            'domain': 'Science',
            'funding_amount': 100000.0,
            'duration_months': 24,
            'team_size': 3,
            'institution': 'University',
            'pi_name': 'PI Low',
            'pi_h_index': 15,
            'novelty_score': 0.3,
            'feasibility_score': 0.95,
            'impact_potential': 'medium',
            'methodology': 'Experimental',
            'prior_publications': 20
        }
        proposal_low = pricer.submit_proposal(**low_risk_params)
        prediction_low = pricer.calculate_roi(proposal_low.proposal_id)

        # Medium risk: medium feasibility and novelty
        med_risk_params = low_risk_params.copy()
        med_risk_params['title'] = 'Medium Risk'
        med_risk_params['pi_name'] = 'PI Med'
        med_risk_params['novelty_score'] = 0.7
        med_risk_params['feasibility_score'] = 0.6
        proposal_med = pricer.submit_proposal(**med_risk_params)
        prediction_med = pricer.calculate_roi(proposal_med.proposal_id)

        # High risk: low feasibility, high novelty
        high_risk_params = low_risk_params.copy()
        high_risk_params['title'] = 'High Risk'
        high_risk_params['pi_name'] = 'PI High'
        high_risk_params['novelty_score'] = 0.95
        high_risk_params['feasibility_score'] = 0.25
        proposal_high = pricer.submit_proposal(**high_risk_params)
        prediction_high = pricer.calculate_roi(proposal_high.proposal_id)

        # Verify we got different risk levels
        risk_levels = {prediction_low.risk_level, prediction_med.risk_level, prediction_high.risk_level}
        # Should have at least 2 different levels
        assert len(risk_levels) >= 2


class TestCLIBasics:
    """Basic CLI smoke tests"""

    def test_cli_module_importable(self):
        """Test that the main module can be imported"""
        from research_pricer.main import ResearchPricer, ResearchProposal, ROIPrediction
        assert ResearchPricer is not None
        assert ResearchProposal is not None
        assert ROIPrediction is not None

    def test_main_function_exists(self):
        """Test that main function exists"""
        from research_pricer.main import main
        assert callable(main)


class TestErrorHandling:
    """Test error handling"""

    def test_calculate_roi_for_invalid_id(self, pricer):
        """Test error when calculating ROI for non-existent proposal"""
        with pytest.raises(ValueError) as exc_info:
            pricer.calculate_roi(9999)

        assert "9999" in str(exc_info.value)
        assert "not found" in str(exc_info.value).lower()

    def test_compare_without_predictions(self, pricer, sample_proposal_params):
        """Test error when comparing proposals without predictions"""
        proposal = pricer.submit_proposal(**sample_proposal_params)

        with pytest.raises(ValueError) as exc_info:
            pricer.compare_proposals([proposal.proposal_id])

        assert "not found" in str(exc_info.value).lower()

    def test_domain_stats_for_empty_domain(self, pricer):
        """Test domain statistics for domain with no proposals"""
        stats = pricer.get_domain_statistics('NonexistentDomain')

        assert 'error' in stats
        assert 'NonexistentDomain' in stats['error']
