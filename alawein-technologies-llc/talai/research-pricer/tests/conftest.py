"""
Pytest configuration and fixtures for ResearchPricer tests
"""
import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime
from research_pricer.main import ResearchPricer, ResearchProposal, ROIPrediction


@pytest.fixture
def temp_data_file():
    """Create a temporary data file for testing"""
    import tempfile
    import os
    # Create a temporary file path but don't create the file yet
    fd, temp_path = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    os.unlink(temp_path)  # Remove the file so ResearchPricer starts fresh
    yield temp_path
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def pricer(temp_data_file):
    """Create a ResearchPricer instance with temporary data file"""
    return ResearchPricer(data_file=temp_data_file)


@pytest.fixture
def sample_proposal_params():
    """Standard proposal parameters for testing"""
    return {
        'title': 'AI for Climate Modeling',
        'domain': 'Computer Science',
        'funding_amount': 250000.0,
        'duration_months': 24,
        'team_size': 5,
        'institution': 'MIT',
        'pi_name': 'Dr. Jane Smith',
        'pi_h_index': 25,
        'novelty_score': 0.8,
        'feasibility_score': 0.7,
        'impact_potential': 'high',
        'methodology': 'Machine Learning',
        'prior_publications': 30
    }


@pytest.fixture
def low_impact_proposal_params():
    """Low impact proposal parameters"""
    return {
        'title': 'Minor Optimization Study',
        'domain': 'Engineering',
        'funding_amount': 50000.0,
        'duration_months': 12,
        'team_size': 2,
        'institution': 'State University',
        'pi_name': 'Dr. John Doe',
        'pi_h_index': 5,
        'novelty_score': 0.3,
        'feasibility_score': 0.9,
        'impact_potential': 'low',
        'methodology': 'Experimental',
        'prior_publications': 3
    }


@pytest.fixture
def transformative_proposal_params():
    """Transformative impact proposal parameters"""
    return {
        'title': 'Quantum Computing Breakthrough',
        'domain': 'Physics',
        'funding_amount': 1000000.0,
        'duration_months': 36,
        'team_size': 10,
        'institution': 'Stanford',
        'pi_name': 'Dr. Albert Einstein',
        'pi_h_index': 50,
        'novelty_score': 0.95,
        'feasibility_score': 0.6,
        'impact_potential': 'transformative',
        'methodology': 'Quantum Theory',
        'prior_publications': 100
    }


@pytest.fixture
def high_risk_proposal_params():
    """High risk proposal parameters"""
    return {
        'title': 'Risky Novel Approach',
        'domain': 'Biology',
        'funding_amount': 200000.0,
        'duration_months': 18,
        'team_size': 3,
        'institution': 'Research Institute',
        'pi_name': 'Dr. Risk Taker',
        'pi_h_index': 10,
        'novelty_score': 0.9,
        'feasibility_score': 0.3,
        'impact_potential': 'medium',
        'methodology': 'Experimental',
        'prior_publications': 8
    }


@pytest.fixture
def populated_pricer(pricer, sample_proposal_params, low_impact_proposal_params):
    """Pricer with multiple proposals already submitted"""
    pricer.submit_proposal(**sample_proposal_params)
    pricer.submit_proposal(**low_impact_proposal_params)
    pricer.submit_proposal(
        title='Third Proposal',
        domain='Computer Science',
        funding_amount=150000.0,
        duration_months=18,
        team_size=4,
        institution='Berkeley',
        pi_name='Dr. Third Person',
        pi_h_index=15,
        novelty_score=0.6,
        feasibility_score=0.8,
        impact_potential='medium',
        methodology='Data Science',
        prior_publications=20
    )
    return pricer


@pytest.fixture
def sample_proposal(pricer, sample_proposal_params):
    """Create and return a sample proposal"""
    return pricer.submit_proposal(**sample_proposal_params)


@pytest.fixture
def sample_prediction(pricer, sample_proposal):
    """Create and return a sample ROI prediction"""
    return pricer.calculate_roi(sample_proposal.proposal_id)
