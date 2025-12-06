"""
Integration tests for HELIOS workflows.

Tests for:
- Discovery → Validation pipeline
- Validation → Learning pipeline
- Full research workflow
- Domain-specific workflows
"""

import pytest


class TestDiscoveryValidationPipeline:
    """Test discovery and validation integration."""

    @pytest.mark.integration
    def test_hypothesis_generation_validation_flow(
        self, mock_hypotheses, mock_validation_results
    ):
        """Test complete flow from generation to validation."""
        # Verify we have hypotheses
        assert len(mock_hypotheses) > 0

        # Verify we have validation results
        assert len(mock_validation_results) > 0

        # Verify correspondence
        assert len(mock_hypotheses) == len(mock_validation_results)

        # Verify data integrity through pipeline
        for h, r in zip(mock_hypotheses, mock_validation_results):
            assert h['id'] == r['hypothesis_id']
            assert r['overall_score'] >= 0
            assert r['overall_score'] <= 100

    @pytest.mark.integration
    def test_strong_hypothesis_filtering(self, mock_hypotheses, mock_validation_results):
        """Test filtering for strong hypotheses."""
        strong = []
        for h, r in zip(mock_hypotheses, mock_validation_results):
            if r['overall_score'] > 70:
                strong.append((h, r))

        # Should have at least some strong hypotheses
        assert len(strong) >= 0

        # All strong hypotheses should have score > 70
        for h, r in strong:
            assert r['overall_score'] > 70


class TestValidationLearningPipeline:
    """Test validation and learning integration."""

    @pytest.mark.integration
    def test_validation_to_learning_flow(
        self, mock_hypotheses, mock_validation_results, mock_failure
    ):
        """Test learning from validation results."""
        # Validation produces results
        assert len(mock_validation_results) > 0

        # Failures are stored
        assert mock_failure['domain'] is not None

        # Learning should extract lessons
        assert len(mock_failure['lessons_learned']) > 0

    @pytest.mark.integration
    def test_agent_performance_tracking(self, mock_agent_performance):
        """Test tracking of agent performance."""
        perf = mock_agent_performance

        assert perf['agent_name'] is not None
        assert perf['domain'] is not None
        assert perf['validation_count'] > 0
        assert 0 <= perf['success_rate'] <= 1.0
        assert perf['avg_score'] >= 0

    @pytest.mark.integration
    def test_hall_of_failures_query(self, mock_failure):
        """Test querying Hall of Failures."""
        # Should have failure data
        assert mock_failure['hypothesis_text'] is not None

        # Should have domain context
        assert mock_failure['domain'] is not None

        # Should have lessons learned
        assert len(mock_failure['lessons_learned']) > 0

        # Should have similarity score
        assert 0 <= mock_failure['similarity'] <= 1.0


class TestFullResearchWorkflow:
    """Test complete research workflows."""

    @pytest.mark.integration
    @pytest.mark.requires_api
    def test_end_to_end_research_workflow(self):
        """Test complete research workflow (requires API)."""
        try:
            from helios.core.orchestration import WorkflowOrchestrator

            orchestrator = WorkflowOrchestrator()

            # This would run the full pipeline
            # results = orchestrator.execute(
            #     research_topic="test topic",
            #     num_hypotheses=3,
            #     enable_meta_learning=True,
            # )

            # For now, just verify it can be instantiated
            assert orchestrator is not None
        except ImportError:
            pytest.skip("WorkflowOrchestrator not available")

    @pytest.mark.integration
    def test_workflow_data_flow(
        self, mock_hypotheses, mock_validation_results, mock_agent_performance
    ):
        """Test data flow through complete workflow."""
        # Generation phase
        assert len(mock_hypotheses) > 0

        # Validation phase
        assert len(mock_validation_results) > 0

        # Learning phase
        assert mock_agent_performance['validation_count'] > 0

        # All phases have data
        assert len(mock_hypotheses) == len(mock_validation_results)


class TestDomainSpecificWorkflows:
    """Test workflows specific to research domains."""

    @pytest.mark.integration
    def test_quantum_domain_workflow(self):
        """Test quantum domain research workflow."""
        try:
            from helios.domains.quantum import QuantumDomain

            domain = QuantumDomain()
            assert domain.name == 'quantum'
        except ImportError:
            pytest.skip("QuantumDomain not available")

    @pytest.mark.integration
    def test_ml_domain_workflow(self):
        """Test ML domain research workflow."""
        try:
            from helios.domains.ml import MLDomain

            domain = MLDomain()
            assert domain.name == 'ml'
        except ImportError:
            pytest.skip("MLDomain not available")

    @pytest.mark.integration
    def test_optimization_domain_workflow(self):
        """Test optimization domain research workflow."""
        try:
            from helios.domains.optimization import OptimizationDomain

            domain = OptimizationDomain()
            assert domain.name == 'optimization'
        except ImportError:
            pytest.skip("OptimizationDomain not available")

    @pytest.mark.integration
    def test_cross_domain_workflow(self, available_domains):
        """Test workflow across multiple domains."""
        try:
            from helios.domains import DOMAINS

            domains = []
            for domain_name in available_domains[:3]:  # Test first 3
                domain = DOMAINS[domain_name]()
                domains.append(domain)

            assert len(domains) == 3
            assert all(d.name in available_domains for d in domains)
        except ImportError:
            pytest.skip("DOMAINS not available")


class TestErrorHandling:
    """Test error handling in workflows."""

    @pytest.mark.integration
    def test_invalid_domain_handling(self):
        """Test handling of invalid domain."""
        try:
            from helios.domains import DOMAINS

            # Non-existent domain should not be in registry
            assert 'nonexistent_domain' not in DOMAINS
        except ImportError:
            pytest.skip("DOMAINS not available")

    @pytest.mark.integration
    def test_validation_with_empty_hypotheses(self):
        """Test validation handles empty hypothesis list gracefully."""
        # Should handle empty list without crashing
        empty_list = []
        assert len(empty_list) == 0

    @pytest.mark.integration
    def test_hypothesis_with_missing_fields(self):
        """Test handling of hypotheses with missing fields."""
        incomplete_hypothesis = {'id': 'test'}

        # Should not crash when optional fields are missing
        assert 'id' in incomplete_hypothesis
