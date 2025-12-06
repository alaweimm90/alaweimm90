"""Tests for hypothesis submission and management."""

import pytest
import json
from pathlib import Path
from datetime import datetime

from experiment_designer.main import Hypothesis, ExperimentDesigner


class TestHypothesisSubmission:
    """Test hypothesis submission functionality."""

    def test_submit_hypothesis_creates_object(self, designer, sample_hypothesis_data):
        """Test that submitting hypothesis creates Hypothesis object."""
        hypothesis = designer.submit_hypothesis(**sample_hypothesis_data)

        assert isinstance(hypothesis, Hypothesis)
        assert hypothesis.hypothesis_id == 1
        assert hypothesis.statement == sample_hypothesis_data['statement']
        assert hypothesis.domain == sample_hypothesis_data['domain']

    def test_submit_hypothesis_increments_id(self, designer, sample_hypothesis_data):
        """Test that hypothesis IDs increment correctly."""
        h1 = designer.submit_hypothesis(**sample_hypothesis_data)
        h2 = designer.submit_hypothesis(**sample_hypothesis_data)
        h3 = designer.submit_hypothesis(**sample_hypothesis_data)

        assert h1.hypothesis_id == 1
        assert h2.hypothesis_id == 2
        assert h3.hypothesis_id == 3

    def test_submit_hypothesis_stores_variables(self, designer, sample_hypothesis_data):
        """Test that hypothesis variables are stored correctly."""
        hypothesis = designer.submit_hypothesis(**sample_hypothesis_data)

        assert hypothesis.variables['independent'] == 'Caffeine dose'
        assert hypothesis.variables['dependent'] == 'Reaction time'
        assert hypothesis.variables['control'] == 'Age, Sleep duration, Time of day'

    def test_submit_hypothesis_creates_timestamp(self, designer, sample_hypothesis_data):
        """Test that hypothesis has created_at timestamp."""
        hypothesis = designer.submit_hypothesis(**sample_hypothesis_data)

        assert hypothesis.created_at is not None
        # Verify it's a valid ISO format timestamp
        datetime.fromisoformat(hypothesis.created_at)

    def test_submit_hypothesis_persists_to_file(self, designer, sample_hypothesis_data, temp_data_file):
        """Test that hypothesis is saved to JSON file."""
        hypothesis = designer.submit_hypothesis(**sample_hypothesis_data)

        with open(temp_data_file, 'r') as f:
            data = json.load(f)

        assert str(hypothesis.hypothesis_id) in data['hypotheses']
        assert data['hypotheses'][str(hypothesis.hypothesis_id)]['statement'] == hypothesis.statement

    def test_submit_hypothesis_validates_effect_size(self, designer):
        """Test that hypothesis accepts valid effect sizes."""
        for effect_size in ['small', 'medium', 'large']:
            h = designer.submit_hypothesis(
                statement='Test',
                domain='psychology',
                independent_var='A',
                dependent_var='B',
                control_vars=['C'],
                expected_effect_size=effect_size,
                mechanism='Test',
                prior_evidence='Test'
            )
            assert h.expected_effect_size == effect_size


class TestHypothesisRetrieval:
    """Test hypothesis retrieval and storage."""

    def test_hypothesis_stored_in_designer(self, designer, sample_hypothesis_data):
        """Test that hypothesis is stored in designer's dictionary."""
        hypothesis = designer.submit_hypothesis(**sample_hypothesis_data)

        assert hypothesis.hypothesis_id in designer.hypotheses
        assert designer.hypotheses[hypothesis.hypothesis_id] == hypothesis

    def test_load_hypotheses_from_file(self, temp_data_file, sample_hypothesis_data):
        """Test that hypotheses are loaded from file on initialization."""
        # Create and save hypothesis
        designer1 = ExperimentDesigner(data_file=temp_data_file)
        h = designer1.submit_hypothesis(**sample_hypothesis_data)

        # Create new designer instance - should load from file
        designer2 = ExperimentDesigner(data_file=temp_data_file)

        assert h.hypothesis_id in designer2.hypotheses
        loaded_h = designer2.hypotheses[h.hypothesis_id]
        assert loaded_h.statement == h.statement
        assert loaded_h.domain == h.domain

    def test_empty_file_initializes_empty_dicts(self, designer):
        """Test that designer initializes with empty dicts when file doesn't exist."""
        assert designer.hypotheses == {}
        assert designer.protocols == {}

    def test_multiple_hypotheses_different_domains(self, designer):
        """Test storing multiple hypotheses from different domains."""
        h1 = designer.submit_hypothesis(
            statement='Hypothesis 1',
            domain='medicine',
            independent_var='A',
            dependent_var='B',
            control_vars=['C'],
            expected_effect_size='small',
            mechanism='M1',
            prior_evidence='E1'
        )

        h2 = designer.submit_hypothesis(
            statement='Hypothesis 2',
            domain='biology',
            independent_var='X',
            dependent_var='Y',
            control_vars=['Z'],
            expected_effect_size='large',
            mechanism='M2',
            prior_evidence='E2'
        )

        assert len(designer.hypotheses) == 2
        assert designer.hypotheses[1].domain == 'medicine'
        assert designer.hypotheses[2].domain == 'biology'
