"""
Tests for proposal submission functionality
"""
import pytest
from datetime import datetime
from research_pricer.main import ResearchProposal, ResearchPricer


class TestProposalSubmission:
    """Test proposal submission and creation"""

    def test_submit_proposal_creates_proposal(self, pricer, sample_proposal_params):
        """Test that submitting a proposal creates a valid proposal object"""
        proposal = pricer.submit_proposal(**sample_proposal_params)

        assert isinstance(proposal, ResearchProposal)
        assert proposal.proposal_id == 1
        assert proposal.title == sample_proposal_params['title']
        assert proposal.funding_amount == sample_proposal_params['funding_amount']

    def test_proposal_id_increments(self, pricer, sample_proposal_params):
        """Test that proposal IDs increment correctly"""
        proposal1 = pricer.submit_proposal(**sample_proposal_params)
        proposal2 = pricer.submit_proposal(**sample_proposal_params)

        assert proposal1.proposal_id == 1
        assert proposal2.proposal_id == 2

    def test_proposal_stored_in_proposals_dict(self, pricer, sample_proposal_params):
        """Test that proposals are stored in the proposals dictionary"""
        proposal = pricer.submit_proposal(**sample_proposal_params)

        assert proposal.proposal_id in pricer.proposals
        assert pricer.proposals[proposal.proposal_id] == proposal

    def test_proposal_has_timestamp(self, pricer, sample_proposal_params):
        """Test that proposals include creation timestamp"""
        proposal = pricer.submit_proposal(**sample_proposal_params)

        assert proposal.created_at is not None
        # Verify it's a valid ISO format timestamp
        datetime.fromisoformat(proposal.created_at)

    def test_proposal_all_fields_set(self, pricer, sample_proposal_params):
        """Test that all proposal fields are correctly set"""
        proposal = pricer.submit_proposal(**sample_proposal_params)

        assert proposal.title == sample_proposal_params['title']
        assert proposal.domain == sample_proposal_params['domain']
        assert proposal.funding_amount == sample_proposal_params['funding_amount']
        assert proposal.duration_months == sample_proposal_params['duration_months']
        assert proposal.team_size == sample_proposal_params['team_size']
        assert proposal.institution == sample_proposal_params['institution']
        assert proposal.pi_name == sample_proposal_params['pi_name']
        assert proposal.pi_h_index == sample_proposal_params['pi_h_index']
        assert proposal.novelty_score == sample_proposal_params['novelty_score']
        assert proposal.feasibility_score == sample_proposal_params['feasibility_score']
        assert proposal.impact_potential == sample_proposal_params['impact_potential']
        assert proposal.methodology == sample_proposal_params['methodology']
        assert proposal.prior_publications == sample_proposal_params['prior_publications']

    def test_multiple_proposals_independent(self, pricer, sample_proposal_params, low_impact_proposal_params):
        """Test that multiple proposals are stored independently"""
        proposal1 = pricer.submit_proposal(**sample_proposal_params)
        proposal2 = pricer.submit_proposal(**low_impact_proposal_params)

        assert len(pricer.proposals) == 2
        assert pricer.proposals[1].title == sample_proposal_params['title']
        assert pricer.proposals[2].title == low_impact_proposal_params['title']

    def test_proposal_persistence(self, temp_data_file, sample_proposal_params):
        """Test that proposals are saved to file"""
        pricer1 = ResearchPricer(data_file=temp_data_file)
        proposal = pricer1.submit_proposal(**sample_proposal_params)

        # Create new instance with same data file
        pricer2 = ResearchPricer(data_file=temp_data_file)

        assert proposal.proposal_id in pricer2.proposals
        assert pricer2.proposals[proposal.proposal_id].title == sample_proposal_params['title']
