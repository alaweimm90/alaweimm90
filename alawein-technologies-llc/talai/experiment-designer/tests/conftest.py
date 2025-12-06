"""Pytest configuration and shared fixtures."""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime

from experiment_designer.main import (
    ExperimentDesigner,
    Hypothesis,
    PowerAnalysis,
    ControlVariable,
    Equipment,
    Step,
    Timeline,
    ExperimentProtocol,
)


@pytest.fixture
def temp_data_file():
    """Provide a temporary data file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        # Write empty JSON structure
        json.dump({'hypotheses': {}, 'protocols': {}}, f)
        temp_path = f.name
    yield temp_path
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def designer(temp_data_file):
    """Provide a fresh ExperimentDesigner instance."""
    return ExperimentDesigner(data_file=temp_data_file)


@pytest.fixture
def sample_hypothesis_data():
    """Provide sample hypothesis data."""
    return {
        'statement': 'Caffeine improves cognitive performance in sleep-deprived individuals',
        'domain': 'psychology',
        'independent_var': 'Caffeine dose',
        'dependent_var': 'Reaction time',
        'control_vars': ['Age', 'Sleep duration', 'Time of day'],
        'expected_effect_size': 'medium',
        'mechanism': 'Adenosine receptor antagonism',
        'prior_evidence': 'Strong evidence from meta-analyses'
    }


@pytest.fixture
def medicine_hypothesis_data():
    """Provide medicine domain hypothesis data."""
    return {
        'statement': 'New drug reduces blood pressure in hypertensive patients',
        'domain': 'medicine',
        'independent_var': 'Drug dosage',
        'dependent_var': 'Systolic blood pressure',
        'control_vars': ['Age', 'Gender', 'BMI'],
        'expected_effect_size': 'large',
        'mechanism': 'ACE inhibition',
        'prior_evidence': 'Promising preclinical studies'
    }


@pytest.fixture
def biology_hypothesis_data():
    """Provide biology domain hypothesis data."""
    return {
        'statement': 'Temperature affects cell growth rate in bacteria',
        'domain': 'biology',
        'independent_var': 'Temperature',
        'dependent_var': 'Colony forming units',
        'control_vars': ['Media type', 'pH level'],
        'expected_effect_size': 'small',
        'mechanism': 'Enzyme kinetics',
        'prior_evidence': 'Limited prior studies'
    }


@pytest.fixture
def physics_hypothesis_data():
    """Provide physics domain hypothesis data."""
    return {
        'statement': 'Magnetic field strength affects electron beam trajectory',
        'domain': 'physics',
        'independent_var': 'Magnetic field strength',
        'dependent_var': 'Beam deflection angle',
        'control_vars': ['Beam energy', 'Ambient temperature'],
        'expected_effect_size': 'large',
        'mechanism': 'Lorentz force',
        'prior_evidence': 'Robust theoretical foundation'
    }


@pytest.fixture
def social_science_hypothesis_data():
    """Provide social science domain hypothesis data."""
    return {
        'statement': 'Social media usage correlates with anxiety levels',
        'domain': 'social_science',
        'independent_var': 'Daily social media hours',
        'dependent_var': 'Anxiety score',
        'control_vars': ['Age', 'Gender', 'SES'],
        'expected_effect_size': 'medium',
        'mechanism': 'Social comparison',
        'prior_evidence': 'Mixed evidence'
    }


@pytest.fixture
def sample_hypothesis(designer, sample_hypothesis_data):
    """Provide a submitted hypothesis."""
    return designer.submit_hypothesis(**sample_hypothesis_data)


@pytest.fixture
def sample_protocol(designer, sample_hypothesis):
    """Provide a complete protocol."""
    return designer.design_protocol(sample_hypothesis.hypothesis_id)


@pytest.fixture
def sample_power_analysis():
    """Provide a sample PowerAnalysis object."""
    return PowerAnalysis(
        effect_size=0.5,
        alpha=0.05,
        power=0.80,
        required_sample_size=128,
        groups=2,
        observations_per_group=64,
        statistical_test='t-test',
        assumptions=['Normal distribution', 'Homogeneity of variance'],
        sensitivity_analysis={'small': 256, 'medium': 64, 'large': 26}
    )


@pytest.fixture
def sample_control_variable():
    """Provide a sample ControlVariable."""
    return ControlVariable(
        name='Age',
        type='block',
        justification='Blocking on Age reduces within-group variance',
        method='Block Age'
    )


@pytest.fixture
def sample_equipment():
    """Provide a sample Equipment object."""
    return Equipment(
        name='Computer workstation',
        quantity=10,
        unit='units',
        estimated_cost=15000,
        vendor='Dell',
        specifications='High-performance computing'
    )


@pytest.fixture
def sample_step():
    """Provide a sample Step object."""
    return Step(
        step_number=1,
        action='Recruit participants',
        duration='2-4 weeks',
        equipment_needed=['Recruitment materials'],
        safety_notes=['Ensure informed consent'],
        quality_checks=['Document decisions'],
        expected_output='Enrolled cohort'
    )


@pytest.fixture
def sample_timeline():
    """Provide a sample Timeline object."""
    return Timeline(
        phase='Planning & Setup',
        description='Protocol development',
        duration_days=60,
        dependencies=[],
        milestones=['IRB approval', 'Equipment delivered']
    )


@pytest.fixture
def designer_with_data(temp_data_file):
    """Provide a designer with pre-populated data."""
    designer = ExperimentDesigner(data_file=temp_data_file)

    # Add hypothesis
    designer.submit_hypothesis(
        statement='Test hypothesis',
        domain='psychology',
        independent_var='Variable A',
        dependent_var='Variable B',
        control_vars=['Control 1', 'Control 2'],
        expected_effect_size='medium',
        mechanism='Test mechanism',
        prior_evidence='Test evidence'
    )

    return designer


@pytest.fixture
def mock_datetime(monkeypatch):
    """Mock datetime.now() for consistent timestamps."""
    class MockDatetime:
        @staticmethod
        def now():
            return datetime(2025, 1, 1, 12, 0, 0)

        @staticmethod
        def isoformat():
            return '2025-01-01T12:00:00'

    monkeypatch.setattr('experiment_designer.main.datetime', MockDatetime)
    return MockDatetime
