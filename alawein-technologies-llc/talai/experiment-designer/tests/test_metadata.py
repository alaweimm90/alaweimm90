"""Tests for protocol metadata (confidence, limitations, alternatives)."""

import pytest


class TestConfidenceScore:
    """Test confidence score estimation."""

    def test_has_confidence_score(self, designer, sample_hypothesis):
        """Test that protocol has confidence score."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert protocol.confidence_score is not None
        assert 0 <= protocol.confidence_score <= 1

    def test_confidence_never_exceeds_95_percent(self, designer):
        """Test that confidence is capped at 0.95."""
        h = designer.submit_hypothesis(
            statement='Test',
            domain='psychology',
            independent_var='A',
            dependent_var='B',
            control_vars=['C'],
            expected_effect_size='large',
            mechanism='Test',
            prior_evidence='Strong and robust evidence'
        )
        protocol = designer.design_protocol(h.hypothesis_id)

        assert protocol.confidence_score <= 0.95

    def test_confidence_never_below_50_percent(self, designer):
        """Test that confidence has minimum of 0.50."""
        h = designer.submit_hypothesis(
            statement='Test',
            domain='psychology',
            independent_var='A',
            dependent_var='B',
            control_vars=['C'],
            expected_effect_size='small',
            mechanism='Test',
            prior_evidence='No prior evidence'
        )
        protocol = designer.design_protocol(h.hypothesis_id)

        assert protocol.confidence_score >= 0.50

    def test_large_effect_increases_confidence(self, designer):
        """Test that large effect size increases confidence."""
        h1 = designer.submit_hypothesis(
            statement='Test',
            domain='psychology',
            independent_var='A',
            dependent_var='B',
            control_vars=['C'],
            expected_effect_size='small',
            mechanism='Test',
            prior_evidence='Test'
        )
        protocol1 = designer.design_protocol(h1.hypothesis_id)

        h2 = designer.submit_hypothesis(
            statement='Test',
            domain='psychology',
            independent_var='A',
            dependent_var='B',
            control_vars=['C'],
            expected_effect_size='large',
            mechanism='Test',
            prior_evidence='Test'
        )
        protocol2 = designer.design_protocol(h2.hypothesis_id)

        # Large effect should have higher confidence
        assert protocol2.confidence_score > protocol1.confidence_score

    def test_strong_prior_evidence_increases_confidence(self, designer):
        """Test that strong prior evidence increases confidence."""
        h1 = designer.submit_hypothesis(
            statement='Test',
            domain='psychology',
            independent_var='A',
            dependent_var='B',
            control_vars=['C'],
            expected_effect_size='medium',
            mechanism='Test',
            prior_evidence='No prior studies'
        )
        protocol1 = designer.design_protocol(h1.hypothesis_id)

        h2 = designer.submit_hypothesis(
            statement='Test',
            domain='psychology',
            independent_var='A',
            dependent_var='B',
            control_vars=['C'],
            expected_effect_size='medium',
            mechanism='Test',
            prior_evidence='Strong evidence from meta-analyses'
        )
        protocol2 = designer.design_protocol(h2.hypothesis_id)

        # Strong evidence should have higher confidence
        assert protocol2.confidence_score > protocol1.confidence_score


class TestLimitations:
    """Test limitation identification."""

    def test_has_limitations(self, designer, sample_hypothesis):
        """Test that protocol has limitations."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert len(protocol.limitations) > 0

    def test_limitations_mention_effect_size_assumption(self, designer, sample_hypothesis):
        """Test that limitations mention effect size assumptions."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        limitations_text = ' '.join(protocol.limitations)
        assert 'effect size' in limitations_text.lower()

    def test_limitations_mention_generalizability(self, designer, sample_hypothesis):
        """Test that limitations mention generalizability."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        limitations_text = ' '.join(protocol.limitations)
        assert 'generalizability' in limitations_text.lower() or 'generalis' in limitations_text.lower()

    def test_observational_mentions_causality_limitation(self, designer, social_science_hypothesis_data):
        """Test that observational designs mention causality limitation."""
        h = designer.submit_hypothesis(**social_science_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        limitations_text = ' '.join(protocol.limitations)
        assert 'causality' in limitations_text.lower() or 'causal' in limitations_text.lower()

    def test_medicine_mentions_term_limitation(self, designer, medicine_hypothesis_data):
        """Test that medicine studies mention short-term limitation."""
        h = designer.submit_hypothesis(**medicine_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        limitations_text = ' '.join(protocol.limitations)
        assert 'short-term' in limitations_text.lower() or 'long-term' in limitations_text.lower()

    def test_nonobservational_may_not_mention_causality(self, designer, sample_hypothesis_data):
        """Test that RCT designs may not mention causality limitation."""
        h = designer.submit_hypothesis(**sample_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        # Factorial design (not observational) may not have causality limitation
        # This is acceptable


class TestAlternatives:
    """Test alternative design consideration."""

    def test_has_alternatives_considered(self, designer, sample_hypothesis):
        """Test that protocol has alternatives considered."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert len(protocol.alternatives_considered) > 0

    def test_alternatives_are_substantive(self, designer, sample_hypothesis):
        """Test that alternatives are substantive descriptions."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        for alt in protocol.alternatives_considered:
            # Each alternative should be a reasonable description
            assert len(alt) > 20

    def test_alternatives_mention_tradeoffs(self, designer, sample_hypothesis):
        """Test that alternatives mention tradeoffs."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        alternatives_text = ' '.join(protocol.alternatives_considered)
        # Should mention pros/cons (faster, cheaper, better, etc.)
        assert any(word in alternatives_text.lower()
                   for word in ['faster', 'efficient', 'cost', 'bias', 'better', 'more'])

    def test_medicine_includes_multisite_alternative(self, designer, medicine_hypothesis_data):
        """Test that medicine domain considers multi-site design."""
        h = designer.submit_hypothesis(**medicine_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        alternatives_text = ' '.join(protocol.alternatives_considered)
        assert 'multi-site' in alternatives_text.lower() or 'multisite' in alternatives_text.lower()


class TestProtocolMetadata:
    """Test general protocol metadata."""

    def test_protocol_has_created_timestamp(self, designer, sample_hypothesis):
        """Test that protocol has created_at timestamp."""
        from datetime import datetime

        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert protocol.created_at is not None
        # Verify it's a valid ISO format timestamp
        datetime.fromisoformat(protocol.created_at)

    def test_protocol_links_to_hypothesis(self, designer, sample_hypothesis):
        """Test that protocol is linked to hypothesis."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert protocol.hypothesis_id == sample_hypothesis.hypothesis_id
