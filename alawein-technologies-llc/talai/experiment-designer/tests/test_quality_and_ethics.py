"""Tests for quality assurance and ethics considerations."""

import pytest


class TestQualityAssurance:
    """Test quality assurance measures."""

    def test_has_quality_assurance_measures(self, designer, sample_hypothesis):
        """Test that protocol has QA measures."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert len(protocol.quality_assurance) > 0

    def test_qa_includes_sops(self, designer, sample_hypothesis):
        """Test that QA includes standard operating procedures."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        qa_text = ' '.join(protocol.quality_assurance)
        assert 'SOP' in qa_text or 'standard operating procedures' in qa_text.lower()

    def test_qa_includes_training(self, designer, sample_hypothesis):
        """Test that QA includes staff training."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        qa_text = ' '.join(protocol.quality_assurance)
        assert 'training' in qa_text.lower() or 'certification' in qa_text.lower()

    def test_qa_includes_equipment_calibration(self, designer, sample_hypothesis):
        """Test that QA includes equipment calibration."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        qa_text = ' '.join(protocol.quality_assurance)
        assert 'calibration' in qa_text.lower()

    def test_qa_includes_data_quality_checks(self, designer, sample_hypothesis):
        """Test that QA includes data quality checks."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        qa_text = ' '.join(protocol.quality_assurance)
        assert 'data' in qa_text.lower() and ('quality' in qa_text.lower() or 'entry' in qa_text.lower())

    def test_qa_includes_preregistration(self, designer, sample_hypothesis):
        """Test that QA includes pre-registration."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        qa_text = ' '.join(protocol.quality_assurance)
        assert 'pre-registration' in qa_text.lower() or 'preregistration' in qa_text.lower()


class TestRiskIdentification:
    """Test risk identification and mitigation."""

    def test_has_risks_identified(self, designer, sample_hypothesis):
        """Test that risks are identified."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert len(protocol.risks) > 0

    def test_risks_have_risk_and_mitigation(self, designer, sample_hypothesis):
        """Test that each risk has both risk and mitigation."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        for risk_item in protocol.risks:
            assert 'risk' in risk_item
            assert 'mitigation' in risk_item
            assert len(risk_item['risk']) > 0
            assert len(risk_item['mitigation']) > 0

    def test_risks_include_recruitment_failure(self, designer, sample_hypothesis):
        """Test that risks include recruitment concerns."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        risks_text = ' '.join([r['risk'] for r in protocol.risks])
        assert 'recruitment' in risks_text.lower() or 'enrollment' in risks_text.lower()

    def test_risks_include_dropout(self, designer, sample_hypothesis):
        """Test that risks include dropout/attrition."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        risks_text = ' '.join([r['risk'] for r in protocol.risks])
        assert 'dropout' in risks_text.lower() or 'loss to follow-up' in risks_text.lower()

    def test_risks_include_equipment_failure(self, designer, sample_hypothesis):
        """Test that risks include equipment failure."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        risks_text = ' '.join([r['risk'] for r in protocol.risks])
        assert 'equipment' in risks_text.lower()

    def test_risks_include_null_result(self, designer, sample_hypothesis):
        """Test that risks include null result concern."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        risks_text = ' '.join([r['risk'] for r in protocol.risks])
        assert 'null' in risks_text.lower() or 'power' in risks_text.lower()

    def test_all_risks_have_mitigations(self, designer, sample_hypothesis):
        """Test that all risks have mitigation strategies."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        for risk_item in protocol.risks:
            assert len(risk_item['mitigation']) > 10  # Substantive mitigation


class TestEthicsConsiderations:
    """Test ethics considerations."""

    def test_has_ethics_considerations(self, designer, sample_hypothesis):
        """Test that protocol has ethics considerations."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        assert len(protocol.ethics_considerations) > 0

    def test_ethics_includes_irb_approval(self, designer, sample_hypothesis):
        """Test that ethics includes IRB approval."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        ethics_text = ' '.join(protocol.ethics_considerations)
        assert 'IRB' in ethics_text or 'ethics committee' in ethics_text.lower()

    def test_ethics_includes_informed_consent(self, designer, sample_hypothesis):
        """Test that ethics includes informed consent."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        ethics_text = ' '.join(protocol.ethics_considerations)
        assert 'informed consent' in ethics_text.lower() or 'consent' in ethics_text.lower()

    def test_ethics_includes_privacy(self, designer, sample_hypothesis):
        """Test that ethics includes data privacy."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        ethics_text = ' '.join(protocol.ethics_considerations)
        assert 'privacy' in ethics_text.lower() or 'confidentiality' in ethics_text.lower()

    def test_ethics_includes_adverse_events(self, designer, sample_hypothesis):
        """Test that ethics includes adverse event monitoring."""
        protocol = designer.design_protocol(sample_hypothesis.hypothesis_id)

        ethics_text = ' '.join(protocol.ethics_considerations)
        assert 'adverse' in ethics_text.lower()

    def test_medicine_has_additional_ethics(self, designer, medicine_hypothesis_data):
        """Test that medicine domain has additional ethics considerations."""
        h = designer.submit_hypothesis(**medicine_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        ethics_text = ' '.join(protocol.ethics_considerations)
        # Should have clinical trial registration
        assert 'ClinicalTrials.gov' in ethics_text or 'trial registration' in ethics_text.lower()

    def test_medicine_has_dsmb(self, designer, medicine_hypothesis_data):
        """Test that medicine domain includes DSMB."""
        h = designer.submit_hypothesis(**medicine_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        ethics_text = ' '.join(protocol.ethics_considerations)
        assert 'DSMB' in ethics_text or 'safety monitoring board' in ethics_text.lower()

    def test_medicine_has_stopping_rules(self, designer, medicine_hypothesis_data):
        """Test that medicine domain includes stopping rules."""
        h = designer.submit_hypothesis(**medicine_hypothesis_data)
        protocol = designer.design_protocol(h.hypothesis_id)

        ethics_text = ' '.join(protocol.ethics_considerations)
        assert 'stopping rules' in ethics_text.lower() or 'futility' in ethics_text.lower()
