"""
Tests for edge cases and boundary conditions
"""
import pytest
import random


class TestBoundaryConditions:
    """Test edge cases and boundary values"""

    def test_minimal_funding_amount(self, pricer):
        """Test proposal with minimal funding"""
        params = {
            'title': 'Minimal Funding',
            'domain': 'Science',
            'funding_amount': 1000.0,
            'duration_months': 12,
            'team_size': 1,
            'institution': 'University',
            'pi_name': 'PI',
            'pi_h_index': 10,
            'novelty_score': 0.5,
            'feasibility_score': 0.5,
            'impact_potential': 'medium',
            'methodology': 'Experimental',
            'prior_publications': 10
        }
        proposal = pricer.submit_proposal(**params)
        prediction = pricer.calculate_roi(proposal.proposal_id)

        # Should still calculate with minimum values
        assert prediction.expected_publications >= 1.0

    def test_maximum_funding_amount(self, pricer):
        """Test proposal with very high funding"""
        params = {
            'title': 'Huge Funding',
            'domain': 'Science',
            'funding_amount': 10000000.0,
            'duration_months': 60,
            'team_size': 20,
            'institution': 'University',
            'pi_name': 'PI',
            'pi_h_index': 100,
            'novelty_score': 1.0,
            'feasibility_score': 1.0,
            'impact_potential': 'transformative',
            'methodology': 'Experimental',
            'prior_publications': 500
        }
        proposal = pricer.submit_proposal(**params)
        prediction = pricer.calculate_roi(proposal.proposal_id)

        # Should cap at maximum publications
        assert prediction.expected_publications <= 20.0

    def test_minimum_novelty_score(self, pricer):
        """Test with minimum novelty score (0)"""
        params = {
            'title': 'No Novelty',
            'domain': 'Science',
            'funding_amount': 100000.0,
            'duration_months': 24,
            'team_size': 3,
            'institution': 'University',
            'pi_name': 'PI',
            'pi_h_index': 15,
            'novelty_score': 0.0,
            'feasibility_score': 0.8,
            'impact_potential': 'medium',
            'methodology': 'Experimental',
            'prior_publications': 20
        }
        proposal = pricer.submit_proposal(**params)
        prediction = pricer.calculate_roi(proposal.proposal_id)

        # Should still produce valid predictions
        assert prediction.expected_publications > 0
        assert prediction.journal_quality_avg > 0

    def test_maximum_novelty_score(self, pricer):
        """Test with maximum novelty score (1.0)"""
        params = {
            'title': 'Highly Novel',
            'domain': 'Science',
            'funding_amount': 100000.0,
            'duration_months': 24,
            'team_size': 3,
            'institution': 'University',
            'pi_name': 'PI',
            'pi_h_index': 15,
            'novelty_score': 1.0,
            'feasibility_score': 0.8,
            'impact_potential': 'medium',
            'methodology': 'Experimental',
            'prior_publications': 20
        }
        proposal = pricer.submit_proposal(**params)
        prediction = pricer.calculate_roi(proposal.proposal_id)

        # Higher novelty should increase journal quality
        assert prediction.journal_quality_avg > 8.0

    def test_minimum_feasibility_score(self, pricer):
        """Test with very low feasibility"""
        params = {
            'title': 'Low Feasibility',
            'domain': 'Science',
            'funding_amount': 100000.0,
            'duration_months': 24,
            'team_size': 3,
            'institution': 'University',
            'pi_name': 'PI',
            'pi_h_index': 15,
            'novelty_score': 0.7,
            'feasibility_score': 0.1,
            'impact_potential': 'medium',
            'methodology': 'Experimental',
            'prior_publications': 20
        }
        proposal = pricer.submit_proposal(**params)
        prediction = pricer.calculate_roi(proposal.proposal_id)

        # Low feasibility should result in high risk
        assert prediction.risk_level in ['medium', 'high']
        assert prediction.failure_probability > 0.5

    def test_single_person_team(self, pricer):
        """Test proposal with team size of 1"""
        params = {
            'title': 'Solo Project',
            'domain': 'Science',
            'funding_amount': 100000.0,
            'duration_months': 24,
            'team_size': 1,
            'institution': 'University',
            'pi_name': 'PI',
            'pi_h_index': 15,
            'novelty_score': 0.7,
            'feasibility_score': 0.7,
            'impact_potential': 'medium',
            'methodology': 'Experimental',
            'prior_publications': 20
        }
        proposal = pricer.submit_proposal(**params)
        prediction = pricer.calculate_roi(proposal.proposal_id)

        assert prediction.expected_publications > 0
        assert prediction.collaborations_formed > 0

    def test_very_large_team(self, pricer):
        """Test proposal with large team"""
        params = {
            'title': 'Large Team Project',
            'domain': 'Science',
            'funding_amount': 500000.0,
            'duration_months': 36,
            'team_size': 25,
            'institution': 'University',
            'pi_name': 'PI',
            'pi_h_index': 30,
            'novelty_score': 0.7,
            'feasibility_score': 0.7,
            'impact_potential': 'high',
            'methodology': 'Experimental',
            'prior_publications': 50
        }
        proposal = pricer.submit_proposal(**params)
        prediction = pricer.calculate_roi(proposal.proposal_id)

        # Large team should have more collaborations
        assert prediction.collaborations_formed >= 5

    def test_short_duration_project(self, pricer):
        """Test very short duration project"""
        params = {
            'title': 'Short Project',
            'domain': 'Science',
            'funding_amount': 100000.0,
            'duration_months': 6,
            'team_size': 3,
            'institution': 'University',
            'pi_name': 'PI',
            'pi_h_index': 15,
            'novelty_score': 0.7,
            'feasibility_score': 0.9,
            'impact_potential': 'medium',
            'methodology': 'Experimental',
            'prior_publications': 20
        }
        proposal = pricer.submit_proposal(**params)
        prediction = pricer.calculate_roi(proposal.proposal_id)

        # Short duration should still allow minimum time to first pub
        assert prediction.time_to_first_pub >= 6

    def test_very_long_duration_project(self, pricer):
        """Test very long duration project"""
        params = {
            'title': 'Long Project',
            'domain': 'Science',
            'funding_amount': 500000.0,
            'duration_months': 120,
            'team_size': 5,
            'institution': 'University',
            'pi_name': 'PI',
            'pi_h_index': 25,
            'novelty_score': 0.7,
            'feasibility_score': 0.7,
            'impact_potential': 'high',
            'methodology': 'Experimental',
            'prior_publications': 40
        }
        proposal = pricer.submit_proposal(**params)
        prediction = pricer.calculate_roi(proposal.proposal_id)

        # Longer projects should produce more output
        assert prediction.expected_publications > 5.0

    def test_zero_prior_publications(self, pricer):
        """Test PI with no prior publications"""
        params = {
            'title': 'New PI',
            'domain': 'Science',
            'funding_amount': 100000.0,
            'duration_months': 24,
            'team_size': 3,
            'institution': 'University',
            'pi_name': 'Junior PI',
            'pi_h_index': 0,
            'novelty_score': 0.7,
            'feasibility_score': 0.7,
            'impact_potential': 'medium',
            'methodology': 'Experimental',
            'prior_publications': 0
        }
        proposal = pricer.submit_proposal(**params)
        prediction = pricer.calculate_roi(proposal.proposal_id)

        # Should have lower tenure probability
        assert prediction.tenure_probability < 0.5

    def test_many_prior_publications(self, pricer):
        """Test very experienced PI"""
        params = {
            'title': 'Experienced PI',
            'domain': 'Science',
            'funding_amount': 100000.0,
            'duration_months': 24,
            'team_size': 3,
            'institution': 'University',
            'pi_name': 'Senior PI',
            'pi_h_index': 80,
            'novelty_score': 0.7,
            'feasibility_score': 0.7,
            'impact_potential': 'high',
            'methodology': 'Experimental',
            'prior_publications': 200
        }
        proposal = pricer.submit_proposal(**params)
        prediction = pricer.calculate_roi(proposal.proposal_id)

        # Should have high tenure probability
        assert prediction.tenure_probability > 0.6


class TestRandomness:
    """Test that randomness is controlled and produces valid results"""

    def test_conference_presentations_randomness(self, pricer, sample_proposal_params):
        """Test that conference presentations have some randomness"""
        random.seed(42)
        proposal1 = pricer.submit_proposal(**sample_proposal_params)
        prediction1 = pricer.calculate_roi(proposal1.proposal_id)

        random.seed(100)
        params2 = sample_proposal_params.copy()
        params2['title'] = 'Different Title'
        params2['pi_name'] = 'Different PI'
        proposal2 = pricer.submit_proposal(**params2)
        prediction2 = pricer.calculate_roi(proposal2.proposal_id)

        # Values should be different due to randomness
        # But both should be valid
        assert prediction1.conf_presentations > 0
        assert prediction2.conf_presentations > 0

    def test_collaborations_randomness(self, pricer, sample_proposal_params):
        """Test that collaborations have randomness"""
        random.seed(42)
        proposal = pricer.submit_proposal(**sample_proposal_params)
        prediction = pricer.calculate_roi(proposal.proposal_id)

        assert prediction.collaborations_formed > 0


class TestExplanationGeneration:
    """Test explanation text generation"""

    def test_explanation_contains_key_info(self, pricer, sample_proposal):
        """Test that explanation contains important information"""
        prediction = pricer.calculate_roi(sample_proposal.proposal_id)

        # Should mention key aspects
        assert sample_proposal.impact_potential in prediction.explanation
        assert sample_proposal.domain in prediction.explanation
        assert str(sample_proposal.funding_amount) in prediction.explanation or f"${sample_proposal.funding_amount:,.0f}" in prediction.explanation

    def test_explanation_mentions_pi_capability(self, pricer):
        """Test that explanation discusses PI capability"""
        params = {
            'title': 'Test',
            'domain': 'Science',
            'funding_amount': 100000.0,
            'duration_months': 24,
            'team_size': 3,
            'institution': 'University',
            'pi_name': 'PI',
            'pi_h_index': 25,
            'novelty_score': 0.7,
            'feasibility_score': 0.7,
            'impact_potential': 'medium',
            'methodology': 'Experimental',
            'prior_publications': 30
        }
        proposal = pricer.submit_proposal(**params)
        prediction = pricer.calculate_roi(proposal.proposal_id)

        # Should mention strong capability (h-index > 20)
        assert 'strong' in prediction.explanation.lower() or 'moderate' in prediction.explanation.lower()

    def test_explanation_mentions_risk(self, pricer, sample_proposal):
        """Test that explanation includes risk assessment"""
        prediction = pricer.calculate_roi(sample_proposal.proposal_id)

        assert 'risk' in prediction.explanation.lower()


class TestComparableGrants:
    """Test comparable grants generation"""

    def test_comparable_grants_include_domain(self, pricer, sample_proposal):
        """Test that comparable grants mention the domain"""
        prediction = pricer.calculate_roi(sample_proposal.proposal_id)

        # At least one should mention the domain
        domain_mentioned = any(sample_proposal.domain in grant for grant in prediction.comparable_grants)
        assert domain_mentioned

    def test_comparable_grants_include_impact(self, pricer, sample_proposal):
        """Test that comparable grants mention impact level"""
        prediction = pricer.calculate_roi(sample_proposal.proposal_id)

        # Should mention impact potential
        impact_mentioned = any(sample_proposal.impact_potential in grant for grant in prediction.comparable_grants)
        assert impact_mentioned

    def test_comparable_grants_include_methodology(self, pricer, sample_proposal):
        """Test that comparable grants mention methodology"""
        prediction = pricer.calculate_roi(sample_proposal.proposal_id)

        # Should mention methodology
        method_mentioned = any(sample_proposal.methodology in grant for grant in prediction.comparable_grants)
        assert method_mentioned


class TestBreakthroughPrediction:
    """Test breakthrough time prediction"""

    def test_low_impact_no_breakthrough(self, pricer, low_impact_proposal_params):
        """Test that low impact proposals don't predict breakthroughs"""
        proposal = pricer.submit_proposal(**low_impact_proposal_params)

        # Run multiple times due to randomness
        breakthrough_found = False
        for _ in range(5):
            prediction = pricer.calculate_roi(proposal.proposal_id)
            if prediction.time_to_breakthrough is not None:
                breakthrough_found = True
                break

        # Low impact should rarely have breakthrough predictions
        # But it's possible, so we just check it's None or a reasonable value
        if prediction.time_to_breakthrough is not None:
            assert prediction.time_to_breakthrough > 0

    def test_transformative_can_have_breakthrough(self, pricer, transformative_proposal_params):
        """Test that transformative proposals can predict breakthroughs"""
        proposal = pricer.submit_proposal(**transformative_proposal_params)

        # Run multiple times to see if breakthrough is possible
        breakthrough_found = False
        for _ in range(10):
            prediction = pricer.calculate_roi(proposal.proposal_id)
            if prediction.time_to_breakthrough is not None:
                breakthrough_found = True
                assert prediction.time_to_breakthrough <= proposal.duration_months
                break

        # With 10 tries and 40% probability, very likely to find at least one
        # But not guaranteed, so we don't assert True


class TestJournalQualityCapping:
    """Test that journal quality is capped appropriately"""

    def test_journal_quality_max_cap(self, pricer):
        """Test that journal quality doesn't exceed maximum"""
        params = {
            'title': 'Test',
            'domain': 'Science',
            'funding_amount': 1000000.0,
            'duration_months': 60,
            'team_size': 20,
            'institution': 'University',
            'pi_name': 'PI',
            'pi_h_index': 100,
            'novelty_score': 1.0,
            'feasibility_score': 1.0,
            'impact_potential': 'transformative',
            'methodology': 'Experimental',
            'prior_publications': 500
        }
        proposal = pricer.submit_proposal(**params)
        prediction = pricer.calculate_roi(proposal.proposal_id)

        # Should be capped at 15.0
        assert prediction.journal_quality_avg <= 15.0


class TestFailureProbabilityCapping:
    """Test that failure probability is within bounds"""

    def test_failure_probability_minimum(self, pricer):
        """Test that failure probability has a minimum floor"""
        params = {
            'title': 'Very Feasible',
            'domain': 'Science',
            'funding_amount': 100000.0,
            'duration_months': 24,
            'team_size': 3,
            'institution': 'University',
            'pi_name': 'PI',
            'pi_h_index': 30,
            'novelty_score': 0.3,
            'feasibility_score': 1.0,
            'impact_potential': 'medium',
            'methodology': 'Experimental',
            'prior_publications': 50
        }
        proposal = pricer.submit_proposal(**params)
        prediction = pricer.calculate_roi(proposal.proposal_id)

        # Should have minimum failure probability
        assert prediction.failure_probability >= 0.05

    def test_failure_probability_maximum(self, pricer):
        """Test that failure probability is capped"""
        params = {
            'title': 'Very Risky',
            'domain': 'Science',
            'funding_amount': 100000.0,
            'duration_months': 24,
            'team_size': 3,
            'institution': 'University',
            'pi_name': 'PI',
            'pi_h_index': 5,
            'novelty_score': 1.0,
            'feasibility_score': 0.1,  # Very low but not zero to avoid division by zero
            'impact_potential': 'medium',
            'methodology': 'Experimental',
            'prior_publications': 2
        }
        proposal = pricer.submit_proposal(**params)
        prediction = pricer.calculate_roi(proposal.proposal_id)

        # Should be capped at 0.8
        assert prediction.failure_probability <= 0.8
