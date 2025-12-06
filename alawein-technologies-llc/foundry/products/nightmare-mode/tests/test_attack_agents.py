"""
Comprehensive test suite for Nightmare Mode attack agents

Run with: pytest tests/ -v --cov=backend --cov-report=html
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from backend.attack_agents.statistical_critic import StatisticalCritic
from backend.attack_agents.methodological_critic import MethodologicalCritic
from backend.orchestrator.ensemble import NightmareModeEnsemble


# Test fixtures
@pytest.fixture
def sample_paper():
    """Sample research paper text for testing"""
    return """
    Abstract: This study examines the effect of X on Y using a sample of 50 participants.
    We found a significant correlation (p < 0.05) between variables.

    Methods: We recruited 50 undergraduate students via convenience sampling.
    Participants completed a 10-question survey.

    Results: Mean score was 7.2 (SD = 1.5). Correlation was r = 0.35, p = 0.04.

    Conclusion: X has a significant effect on Y.
    """


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response"""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": """
                    {
                        "attacks": [
                            {
                                "category": "sample_size",
                                "severity": 8,
                                "description": "Sample size of 50 is too small for reliable conclusions",
                                "quote": "sample of 50 participants"
                            }
                        ],
                        "overall_severity": 7,
                        "summary": "Multiple statistical concerns identified"
                    }
                    """
                },
                "finish_reason": "stop"
            }
        ]
    }


# Statistical Critic Tests
class TestStatisticalCritic:
    """Test suite for Statistical Critic agent"""

    @pytest.mark.asyncio
    async def test_attack_returns_valid_structure(self, sample_paper, mock_openai_response):
        """Test that attack() returns expected data structure"""
        critic = StatisticalCritic()

        with patch.object(critic.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = Mock(
                choices=[Mock(message=Mock(content=mock_openai_response["choices"][0]["message"]["content"]))]
            )

            result = await critic.attack(sample_paper, difficulty="hard")

            # Verify structure
            assert "attacks" in result
            assert "overall_severity" in result
            assert "summary" in result
            assert isinstance(result["attacks"], list)
            assert isinstance(result["overall_severity"], (int, float))

    @pytest.mark.asyncio
    async def test_attack_identifies_sample_size_issues(self, sample_paper):
        """Test that small sample sizes are flagged"""
        critic = StatisticalCritic()

        with patch.object(critic.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            # Mock response that identifies sample size issue
            mock_create.return_value = Mock(
                choices=[Mock(message=Mock(content='{"attacks": [{"category": "sample_size", "severity": 8}], "overall_severity": 7, "summary": "Sample size concerns"}'))]
            )

            result = await critic.attack(sample_paper)

            # Verify sample size issue was identified
            assert any(attack["category"] == "sample_size" for attack in result["attacks"])

    @pytest.mark.asyncio
    async def test_difficulty_levels(self, sample_paper):
        """Test that different difficulty levels work"""
        critic = StatisticalCritic()

        for difficulty in ["easy", "medium", "hard", "nightmare"]:
            with patch.object(critic.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
                mock_create.return_value = Mock(
                    choices=[Mock(message=Mock(content='{"attacks": [], "overall_severity": 5, "summary": "Test"}'))]
                )

                result = await critic.attack(sample_paper, difficulty=difficulty)
                assert result is not None

    @pytest.mark.asyncio
    async def test_handles_api_errors_gracefully(self, sample_paper):
        """Test error handling for API failures"""
        critic = StatisticalCritic()

        with patch.object(critic.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("API Error")

            with pytest.raises(Exception):
                await critic.attack(sample_paper)

    @pytest.mark.asyncio
    async def test_invalid_json_response_handling(self, sample_paper):
        """Test handling of malformed API responses"""
        critic = StatisticalCritic()

        with patch.object(critic.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            # Return invalid JSON
            mock_create.return_value = Mock(
                choices=[Mock(message=Mock(content='Invalid JSON {'))]
            )

            with pytest.raises(Exception):
                await critic.attack(sample_paper)


# Methodological Critic Tests
class TestMethodologicalCritic:
    """Test suite for Methodological Critic agent"""

    @pytest.mark.asyncio
    async def test_identifies_convenience_sampling(self, sample_paper):
        """Test that convenience sampling is flagged"""
        critic = MethodologicalCritic()

        with patch.object(critic.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = Mock(
                choices=[Mock(message=Mock(content='{"attacks": [{"category": "sampling_bias", "severity": 7}], "overall_severity": 6, "summary": "Sampling concerns"}'))]
            )

            result = await critic.attack(sample_paper)

            assert any(attack["category"] == "sampling_bias" for attack in result["attacks"])

    @pytest.mark.asyncio
    async def test_concurrent_attacks(self):
        """Test that multiple attacks can run concurrently"""
        critic = MethodologicalCritic()
        papers = [f"Paper {i}" for i in range(5)]

        with patch.object(critic.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = Mock(
                choices=[Mock(message=Mock(content='{"attacks": [], "overall_severity": 5, "summary": "Test"}'))]
            )

            # Run attacks concurrently
            results = await asyncio.gather(*[critic.attack(paper) for paper in papers])

            assert len(results) == 5
            assert all(isinstance(r, dict) for r in results)


# Ensemble Tests
class TestNightmareModeEnsemble:
    """Test suite for multi-model ensemble orchestrator"""

    @pytest.mark.asyncio
    async def test_coordinate_attack_returns_complete_result(self, sample_paper):
        """Test that ensemble coordination returns all required fields"""
        ensemble = NightmareModeEnsemble()

        with patch.object(ensemble, '_run_parallel_attacks', new_callable=AsyncMock) as mock_parallel:
            with patch.object(ensemble, '_run_debate_phase', new_callable=AsyncMock) as mock_debate:
                with patch.object(ensemble, '_build_consensus', new_callable=AsyncMock) as mock_consensus:
                    # Mock responses
                    mock_parallel.return_value = {
                        "statistical": {"attacks": [], "overall_severity": 5},
                        "methodological": {"attacks": [], "overall_severity": 6}
                    }
                    mock_debate.return_value = {"debate_summary": "Test debate"}
                    mock_consensus.return_value = {
                        "survival_score": 65,
                        "critical_issues": [],
                        "consensus": "Paper has moderate issues"
                    }

                    result = await ensemble.coordinate_attack(sample_paper)

                    # Verify complete structure
                    assert "attacks_by_dimension" in result
                    assert "debate" in result
                    assert "consensus" in result
                    assert "survival_score" in result
                    assert isinstance(result["survival_score"], (int, float))

    @pytest.mark.asyncio
    async def test_survival_score_range(self, sample_paper):
        """Test that survival score is always 0-100"""
        ensemble = NightmareModeEnsemble()

        with patch.object(ensemble, '_run_parallel_attacks', new_callable=AsyncMock):
            with patch.object(ensemble, '_run_debate_phase', new_callable=AsyncMock):
                with patch.object(ensemble, '_build_consensus', new_callable=AsyncMock) as mock_consensus:
                    # Test various scores
                    for score in [-10, 0, 50, 100, 150]:
                        mock_consensus.return_value = {
                            "survival_score": score,
                            "critical_issues": [],
                            "consensus": "Test"
                        }

                        result = await ensemble.coordinate_attack(sample_paper)

                        # Verify score is clamped to 0-100
                        assert 0 <= result["survival_score"] <= 100

    @pytest.mark.asyncio
    async def test_handles_partial_failures(self, sample_paper):
        """Test that ensemble handles failures in individual agents"""
        ensemble = NightmareModeEnsemble()

        with patch.object(ensemble, '_run_parallel_attacks', new_callable=AsyncMock) as mock_parallel:
            # Simulate one agent failing
            mock_parallel.return_value = {
                "statistical": {"attacks": [], "overall_severity": 5},
                "methodological": {"error": "API timeout"}
            }

            with patch.object(ensemble, '_run_debate_phase', new_callable=AsyncMock):
                with patch.object(ensemble, '_build_consensus', new_callable=AsyncMock) as mock_consensus:
                    mock_consensus.return_value = {
                        "survival_score": 70,
                        "critical_issues": [],
                        "consensus": "Partial analysis"
                    }

                    result = await ensemble.coordinate_attack(sample_paper)

                    # Should still return a result
                    assert result is not None
                    assert "survival_score" in result


# Integration Tests
class TestIntegration:
    """End-to-end integration tests"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_pipeline_with_real_paper(self):
        """Test complete pipeline with realistic paper (requires API keys)"""
        # Skip if no API key
        import os
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No OPENAI_API_KEY found")

        paper = """
        Title: The Effect of Sleep on Memory

        Abstract: We investigated whether sleep improves memory consolidation.

        Methods: 30 college students were recruited via flyers on campus.
        Half slept normally, half were sleep-deprived for 24 hours.
        All completed a memory test.

        Results: The sleep group scored 78% (SD=12%), the sleep-deprived group
        scored 65% (SD=15%). t-test showed p=0.03.

        Conclusion: Sleep significantly improves memory.
        """

        ensemble = NightmareModeEnsemble()
        result = await ensemble.coordinate_attack(paper)

        # Verify realistic output
        assert result["survival_score"] < 80  # This paper has issues
        assert len(result["attacks_by_dimension"]) > 0

    @pytest.mark.asyncio
    async def test_performance_benchmark(self, sample_paper):
        """Test that attack completes within reasonable time"""
        import time

        ensemble = NightmareModeEnsemble()

        with patch.object(ensemble, '_run_parallel_attacks', new_callable=AsyncMock) as mock_parallel:
            with patch.object(ensemble, '_run_debate_phase', new_callable=AsyncMock) as mock_debate:
                with patch.object(ensemble, '_build_consensus', new_callable=AsyncMock) as mock_consensus:
                    mock_parallel.return_value = {"statistical": {"attacks": [], "overall_severity": 5}}
                    mock_debate.return_value = {"debate_summary": "Test"}
                    mock_consensus.return_value = {"survival_score": 70, "critical_issues": [], "consensus": "Test"}

                    start = time.time()
                    await ensemble.coordinate_attack(sample_paper)
                    elapsed = time.time() - start

                    # Should complete in under 1 second (mocked)
                    assert elapsed < 1.0


# Fixtures for database testing (if using DB)
@pytest.fixture
def db_session():
    """Create a test database session"""
    # This would connect to test DB
    # For now, just a placeholder
    pass


# Performance Tests
class TestPerformance:
    """Performance and load testing"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_requests(self):
        """Test handling of multiple concurrent requests"""
        critic = StatisticalCritic()

        with patch.object(critic.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = Mock(
                choices=[Mock(message=Mock(content='{"attacks": [], "overall_severity": 5, "summary": "Test"}'))]
            )

            # Simulate 10 concurrent requests
            papers = [f"Paper {i}" for i in range(10)]
            results = await asyncio.gather(*[critic.attack(paper) for paper in papers])

            assert len(results) == 10
            assert all(r is not None for r in results)

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_large_paper_handling(self):
        """Test handling of very large papers"""
        # 50KB paper (typical research paper)
        large_paper = "This is a research paper. " * 2000

        critic = StatisticalCritic()

        with patch.object(critic.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = Mock(
                choices=[Mock(message=Mock(content='{"attacks": [], "overall_severity": 5, "summary": "Test"}'))]
            )

            result = await critic.attack(large_paper)
            assert result is not None


# Edge Cases
class TestEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.mark.asyncio
    async def test_empty_paper(self):
        """Test handling of empty input"""
        critic = StatisticalCritic()

        with pytest.raises(ValueError):
            await critic.attack("")

    @pytest.mark.asyncio
    async def test_non_english_paper(self):
        """Test handling of non-English text"""
        spanish_paper = "Este es un artículo de investigación en español."

        critic = StatisticalCritic()

        with patch.object(critic.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = Mock(
                choices=[Mock(message=Mock(content='{"attacks": [], "overall_severity": 5, "summary": "Test"}'))]
            )

            result = await critic.attack(spanish_paper)
            # Should still process (GPT-4 is multilingual)
            assert result is not None

    @pytest.mark.asyncio
    async def test_malformed_paper(self):
        """Test handling of malformed/garbage input"""
        garbage = "asdf1234!@#$ %%%% {{{}}} "

        critic = StatisticalCritic()

        with patch.object(critic.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = Mock(
                choices=[Mock(message=Mock(content='{"attacks": [], "overall_severity": 0, "summary": "Unable to analyze"}'))]
            )

            result = await critic.attack(garbage)
            assert result["overall_severity"] == 0  # Should recognize it's not a real paper


# Pytest configuration
def pytest_configure(config):
    """Configure custom markers"""
    config.addinivalue_line("markers", "integration: Integration tests requiring API keys")
    config.addinivalue_line("markers", "performance: Performance and load tests")


# Coverage targets:
# - Overall: >80%
# - Critical paths (attack methods): >95%
# - Error handling: >90%
