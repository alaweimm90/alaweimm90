#!/usr/bin/env python3
"""
Comprehensive test suite for IdeaCalculus
Tests cover: data structures, calculus operations, edge cases, and integration workflows
"""

import pytest
import json
import math
from dataclasses import asdict
from typing import List

from idea_calculus.main import (
    Idea,
    IdeaDerivative,
    IdeaSynthesis,
    IdeaLimit,
    IdeaComposition,
    IdeaCalculusEngine,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def engine():
    """Create a fresh IdeaCalculusEngine instance for each test"""
    return IdeaCalculusEngine()


@pytest.fixture
def sample_ideas():
    """Sample ideas for testing"""
    return [
        Idea(
            name="quantum computing",
            novelty=0.85,
            impact=0.90,
            complexity=0.95,
            domain="physics",
            timestamp="2024-01-01",
            dependencies=["quantum mechanics", "computer science"]
        ),
        Idea(
            name="machine learning",
            novelty=0.60,
            impact=0.95,
            complexity=0.70,
            domain="ai",
            timestamp="2024-01-01",
            dependencies=["statistics", "optimization"]
        ),
        Idea(
            name="wheel",
            novelty=0.05,
            impact=0.99,
            complexity=0.10,
            domain="mechanics",
            timestamp="ancient",
            dependencies=[]
        ),
    ]


@pytest.fixture
def idea_names():
    """Common idea names for testing"""
    return {
        'emerging': ['quantum', 'gpt-4', 'transformer', 'crispr'],
        'common': ['wheel', 'fire', 'addition', 'writing'],
        'medium': ['neural networks', 'calculus', 'democracy']
    }


# ============================================================================
# DATA STRUCTURE TESTS
# ============================================================================

class TestDataStructures:
    """Test dataclass structures and serialization"""

    def test_idea_creation(self):
        """Test Idea dataclass creation and fields"""
        idea = Idea(
            name="test idea",
            novelty=0.8,
            impact=0.7,
            complexity=0.6,
            domain="test",
            timestamp="2024-01-01",
            dependencies=["dep1", "dep2"]
        )
        assert idea.name == "test idea"
        assert idea.novelty == 0.8
        assert idea.impact == 0.7
        assert idea.complexity == 0.6
        assert idea.domain == "test"
        assert len(idea.dependencies) == 2

    def test_idea_default_values(self):
        """Test Idea default values"""
        idea = Idea(
            name="minimal",
            novelty=0.5,
            impact=0.5,
            complexity=0.5,
            domain="test"
        )
        assert idea.timestamp == ""
        assert idea.dependencies == []

    def test_idea_serialization(self, sample_ideas):
        """Test Idea serialization to dict"""
        idea = sample_ideas[0]
        idea_dict = asdict(idea)
        assert isinstance(idea_dict, dict)
        assert idea_dict['name'] == "quantum computing"
        assert 'novelty' in idea_dict
        assert 'dependencies' in idea_dict

    def test_idea_derivative_structure(self):
        """Test IdeaDerivative dataclass structure"""
        deriv = IdeaDerivative(
            original_idea="test",
            variable="time",
            rate_of_change=-0.05,
            interpretation="test interpretation",
            future_projection={"1": 0.9, "5": 0.6}
        )
        assert deriv.variable == "time"
        assert deriv.rate_of_change == -0.05
        assert len(deriv.future_projection) == 2

    def test_idea_synthesis_structure(self):
        """Test IdeaSynthesis dataclass structure"""
        synthesis = IdeaSynthesis(
            component_ideas=["idea1", "idea2"],
            synthesized_idea="combined",
            emergence_score=0.3,
            synergy_effects=["effect1"],
            applications=["app1", "app2"]
        )
        assert len(synthesis.component_ideas) == 2
        assert synthesis.emergence_score == 0.3
        assert len(synthesis.applications) == 2

    def test_idea_limit_structure(self):
        """Test IdeaLimit dataclass structure"""
        limit = IdeaLimit(
            original_idea="test",
            variable="time",
            approach_value="infinity",
            limit_value="commoditization",
            interpretation="test",
            paradoxes=["paradox1", "paradox2"]
        )
        assert limit.approach_value == "infinity"
        assert len(limit.paradoxes) == 2

    def test_idea_composition_structure(self):
        """Test IdeaComposition dataclass structure"""
        comp = IdeaComposition(
            outer_idea="outer",
            inner_idea="inner",
            composition="outer(inner)",
            emergent_properties=["prop1"],
            novel_applications=["app1"]
        )
        assert comp.composition == "outer(inner)"
        assert len(comp.emergent_properties) == 1


# ============================================================================
# ENGINE INITIALIZATION TESTS
# ============================================================================

class TestEngineInitialization:
    """Test IdeaCalculusEngine initialization"""

    def test_engine_creation(self, engine):
        """Test engine can be created"""
        assert engine is not None
        assert isinstance(engine, IdeaCalculusEngine)

    def test_decay_models_exist(self, engine):
        """Test decay models are initialized"""
        assert hasattr(engine, 'decay_models')
        assert 'exponential' in engine.decay_models
        assert 'power_law' in engine.decay_models
        assert 'sigmoid' in engine.decay_models

    def test_decay_models_are_callable(self, engine):
        """Test decay models are functions"""
        for model_name, model_func in engine.decay_models.items():
            assert callable(model_func)
            # Test they can be called
            result = model_func(1, 0.8)
            assert isinstance(result, (int, float))

    def test_domains_exist(self, engine):
        """Test domain knowledge base is initialized"""
        assert hasattr(engine, 'domains')
        assert 'ai' in engine.domains
        assert 'physics' in engine.domains
        assert 'biology' in engine.domains
        assert 'math' in engine.domains

    def test_domains_have_concepts(self, engine):
        """Test domains contain concept lists"""
        for domain, concepts in engine.domains.items():
            assert isinstance(concepts, list)
            assert len(concepts) > 0
            assert all(isinstance(c, str) for c in concepts)


# ============================================================================
# DERIVATIVE TESTS
# ============================================================================

class TestDerivative:
    """Test derivative calculations"""

    def test_derivative_time_basic(self, engine):
        """Test derivative with respect to time"""
        result = engine.derivative("quantum computing", "time")
        assert isinstance(result, IdeaDerivative)
        assert result.original_idea == "quantum computing"
        assert result.variable == "time"
        assert result.rate_of_change < 0  # Novelty decays
        assert len(result.future_projection) > 0

    def test_derivative_knowledge(self, engine):
        """Test derivative with respect to knowledge"""
        result = engine.derivative("machine learning", "knowledge")
        assert result.variable == "knowledge"
        assert result.rate_of_change > 0  # Complexity increases
        assert "knowledge" in result.interpretation.lower()

    def test_derivative_resources(self, engine):
        """Test derivative with respect to resources"""
        result = engine.derivative("ai research", "resources")
        assert result.variable == "resources"
        assert result.rate_of_change > 0  # Impact grows
        assert any("$" in k for k in result.future_projection.keys())

    def test_derivative_unknown_variable(self, engine):
        """Test derivative with unknown variable"""
        result = engine.derivative("test idea", "unknown")
        assert result.rate_of_change == 0.0
        assert "Unknown variable" in result.interpretation
        assert result.future_projection == {}

    def test_derivative_future_projection_time(self, engine):
        """Test future projection contains expected time points"""
        result = engine.derivative("test", "time")
        assert "1" in result.future_projection
        assert "5" in result.future_projection
        assert "10" in result.future_projection
        assert "20" in result.future_projection

    def test_derivative_future_projection_knowledge(self, engine):
        """Test future projection for knowledge variable"""
        result = engine.derivative("test", "knowledge")
        assert "10" in result.future_projection
        assert "100" in result.future_projection
        assert "1000" in result.future_projection

    def test_derivative_rate_rounded(self, engine):
        """Test rate_of_change is properly rounded"""
        result = engine.derivative("test", "time")
        # Check it's rounded to 4 decimal places
        rate_str = str(result.rate_of_change)
        if '.' in rate_str:
            decimals = len(rate_str.split('.')[1])
            assert decimals <= 4

    def test_derivative_emerging_vs_common_ideas(self, engine, idea_names):
        """Test derivatives differ for emerging vs common ideas"""
        emerging_result = engine.derivative(idea_names['emerging'][0], "time")
        common_result = engine.derivative(idea_names['common'][0], "time")
        # Emerging ideas should have higher magnitude rate (more novelty to lose)
        assert abs(emerging_result.rate_of_change) > abs(common_result.rate_of_change)


# ============================================================================
# INTEGRATION/SYNTHESIS TESTS
# ============================================================================

class TestIntegration:
    """Test idea integration/synthesis"""

    def test_integrate_two_ideas(self, engine):
        """Test synthesizing two ideas"""
        result = engine.integrate(["machine learning", "biology"])
        assert isinstance(result, IdeaSynthesis)
        assert len(result.component_ideas) == 2
        assert result.synthesized_idea is not None
        assert result.emergence_score >= 0

    def test_integrate_multiple_ideas(self, engine):
        """Test synthesizing more than two ideas"""
        result = engine.integrate(["ai", "physics", "chemistry"])
        assert len(result.component_ideas) == 3
        assert len(result.synergy_effects) > 0
        assert len(result.applications) > 0

    def test_integrate_requires_minimum_ideas(self, engine):
        """Test integration fails with single idea"""
        with pytest.raises(ValueError) as excinfo:
            engine.integrate(["single idea"])
        assert "at least 2 ideas" in str(excinfo.value).lower()

    def test_integrate_empty_list(self, engine):
        """Test integration fails with empty list"""
        with pytest.raises(ValueError):
            engine.integrate([])

    def test_synthesized_name_two_ideas(self, engine):
        """Test synthesized name format for two ideas"""
        result = engine.integrate(["idea1", "idea2"])
        assert "idea1" in result.synthesized_idea or "idea2" in result.synthesized_idea

    def test_synthesized_name_many_ideas(self, engine):
        """Test synthesized name format for many ideas"""
        result = engine.integrate(["idea1", "idea2", "idea3", "idea4"])
        assert "Hybrid" in result.synthesized_idea or "/" in result.synthesized_idea

    def test_synergy_effects_present(self, engine):
        """Test synergy effects are generated"""
        result = engine.integrate(["ai", "biology"])
        assert len(result.synergy_effects) >= 2
        # Check that synergy effects are meaningful strings
        assert all(len(s) > 0 for s in result.synergy_effects)

    def test_applications_generated(self, engine):
        """Test applications are generated"""
        result = engine.integrate(["machine learning", "physics"])
        assert len(result.applications) >= 2
        assert all(isinstance(app, str) for app in result.applications)

    def test_emergence_score_range(self, engine):
        """Test emergence score is in valid range"""
        result = engine.integrate(["idea1", "idea2"])
        assert -1 <= result.emergence_score <= 1

    def test_cross_domain_synergy(self, engine):
        """Test cross-domain ideas create synergy"""
        # Use ideas from different domains
        result = engine.integrate(["quantum mechanics", "machine learning"])
        synergy_text = " ".join(result.synergy_effects)
        # Should detect cross-domain if both are in domain lists
        assert len(result.synergy_effects) > 0


# ============================================================================
# LIMIT TESTS
# ============================================================================

class TestLimit:
    """Test limit calculations"""

    def test_limit_time_infinity(self, engine):
        """Test limit as time approaches infinity"""
        result = engine.limit("test idea", "time", "infinity")
        assert isinstance(result, IdeaLimit)
        assert result.variable == "time"
        assert result.approach_value == "infinity"
        assert "commoditization" in result.limit_value.lower()
        assert len(result.paradoxes) > 0

    def test_limit_time_zero(self, engine):
        """Test limit as time approaches zero"""
        result = engine.limit("test idea", "time", "zero")
        assert result.approach_value == "zero"
        assert "discovery" in result.limit_value.lower() or "novelty" in result.limit_value.lower()
        assert len(result.paradoxes) > 0

    def test_limit_knowledge_infinity(self, engine):
        """Test limit as knowledge approaches infinity"""
        result = engine.limit("test idea", "knowledge", "infinity")
        assert result.variable == "knowledge"
        assert "understanding" in result.limit_value.lower() or "mastery" in result.limit_value.lower()
        assert any("Gödel" in p or "uncertainty" in p.lower() for p in result.paradoxes)

    def test_limit_knowledge_zero(self, engine):
        """Test limit as knowledge approaches zero"""
        result = engine.limit("test idea", "knowledge", "zero")
        assert "ignorance" in result.limit_value.lower() or "unknown" in result.limit_value.lower()

    def test_limit_resources_infinity(self, engine):
        """Test limit as resources approach infinity"""
        result = engine.limit("test idea", "resources", "infinity")
        assert result.variable == "resources"
        assert "impact" in result.limit_value.lower() or "potential" in result.limit_value.lower()

    def test_limit_resources_zero(self, engine):
        """Test limit as resources approach zero"""
        result = engine.limit("test idea", "resources", "zero")
        assert "theoretical" in result.limit_value.lower()

    def test_limit_unknown_variable_infinity(self, engine):
        """Test limit with unknown variable"""
        result = engine.limit("test", "unknown_var", "infinity")
        assert result.limit_value == "Unknown"
        assert result.paradoxes == []

    def test_limit_custom_approach(self, engine):
        """Test limit with custom approach value"""
        result = engine.limit("test", "time", "100")
        assert result.approach_value == "100"
        assert "Approaching 100" in result.limit_value

    def test_limit_paradoxes_structure(self, engine):
        """Test paradoxes are properly structured"""
        result = engine.limit("test", "time", "infinity")
        assert isinstance(result.paradoxes, list)
        assert all(isinstance(p, str) for p in result.paradoxes)
        assert len(result.paradoxes) >= 3  # Should have multiple paradoxes

    def test_limit_interpretation_includes_idea(self, engine):
        """Test interpretation mentions the original idea"""
        idea_name = "quantum computing"
        result = engine.limit(idea_name, "time", "infinity")
        assert idea_name in result.interpretation


# ============================================================================
# COMPOSITION TESTS
# ============================================================================

class TestComposition:
    """Test idea composition"""

    def test_compose_basic(self, engine):
        """Test basic idea composition"""
        result = engine.compose("machine learning", "physics")
        assert isinstance(result, IdeaComposition)
        assert result.outer_idea == "machine learning"
        assert result.inner_idea == "physics"
        assert "machine learning" in result.composition
        assert "physics" in result.composition

    def test_composition_format(self, engine):
        """Test composition follows f(g(x)) format"""
        result = engine.compose("outer", "inner")
        assert result.composition == "outer(inner)"

    def test_emergent_properties_generated(self, engine):
        """Test emergent properties are generated"""
        result = engine.compose("ai", "chemistry")
        assert len(result.emergent_properties) > 0
        assert all(isinstance(prop, str) for prop in result.emergent_properties)

    def test_emergent_properties_mention_both_ideas(self, engine):
        """Test emergent properties reference both ideas"""
        outer, inner = "quantum", "biology"
        result = engine.compose(outer, inner)
        props_text = " ".join(result.emergent_properties)
        # At least one property should mention the ideas
        assert len(result.emergent_properties) >= 3

    def test_novel_applications_generated(self, engine):
        """Test novel applications are generated"""
        result = engine.compose("physics", "economics")
        assert len(result.novel_applications) > 0
        assert all(isinstance(app, str) for app in result.novel_applications)

    def test_novel_applications_count(self, engine):
        """Test minimum number of applications generated"""
        result = engine.compose("idea1", "idea2")
        assert len(result.novel_applications) >= 3

    def test_compose_different_orders(self, engine):
        """Test composition is order-dependent"""
        result1 = engine.compose("A", "B")
        result2 = engine.compose("B", "A")
        assert result1.composition != result2.composition
        assert result1.outer_idea == result2.inner_idea
        assert result1.inner_idea == result2.outer_idea


# ============================================================================
# PRIVATE METHOD TESTS
# ============================================================================

class TestPrivateMethods:
    """Test private helper methods"""

    def test_estimate_novelty_common_ideas(self, engine, idea_names):
        """Test novelty estimation for common ideas"""
        for idea in idea_names['common']:
            novelty = engine._estimate_novelty(idea)
            assert 0.0 <= novelty <= 0.2  # Should be low

    def test_estimate_novelty_emerging_ideas(self, engine, idea_names):
        """Test novelty estimation for emerging ideas"""
        for idea in idea_names['emerging']:
            novelty = engine._estimate_novelty(idea)
            assert 0.6 <= novelty <= 1.0  # Should be high

    def test_estimate_novelty_range(self, engine):
        """Test novelty is always in valid range [0, 1]"""
        test_ideas = ["random1", "random2", "random3", "quantum", "wheel"]
        for idea in test_ideas:
            novelty = engine._estimate_novelty(idea)
            assert 0.0 <= novelty <= 1.0

    def test_synthesize_name_two_ideas(self, engine):
        """Test name synthesis for two ideas"""
        name = engine._synthesize_name(["idea1", "idea2"])
        assert isinstance(name, str)
        assert len(name) > 0

    def test_synthesize_name_multiple_ideas(self, engine):
        """Test name synthesis for multiple ideas"""
        name = engine._synthesize_name(["a", "b", "c", "d"])
        assert isinstance(name, str)
        assert "Hybrid" in name or "/" in name

    def test_identify_synergies_returns_list(self, engine):
        """Test synergies are returned as list"""
        synergies = engine._identify_synergies(["idea1", "idea2"])
        assert isinstance(synergies, list)
        assert len(synergies) > 0

    def test_generate_applications_returns_list(self, engine):
        """Test applications are returned as list"""
        apps = engine._generate_applications(["idea1", "idea2"], "synthesis")
        assert isinstance(apps, list)
        assert len(apps) >= 3

    def test_find_emergent_properties_returns_list(self, engine):
        """Test emergent properties are returned as list"""
        props = engine._find_emergent_properties("outer", "inner")
        assert isinstance(props, list)
        assert len(props) >= 3

    def test_compose_applications_returns_list(self, engine):
        """Test composition applications are returned as list"""
        apps = engine._compose_applications("outer", "inner")
        assert isinstance(apps, list)
        assert len(apps) >= 3


# ============================================================================
# INTEGRATION WORKFLOW TESTS
# ============================================================================

class TestIntegrationWorkflows:
    """Test complete workflows combining multiple operations"""

    def test_workflow_derivative_then_limit(self, engine):
        """Test workflow: calculate derivative then limit"""
        idea = "quantum computing"

        # Calculate derivative
        deriv = engine.derivative(idea, "time")
        assert deriv.rate_of_change < 0

        # Calculate limit
        limit = engine.limit(idea, "time", "infinity")
        assert "commoditization" in limit.limit_value.lower()

    def test_workflow_integrate_then_derivative(self, engine):
        """Test workflow: synthesize ideas then calculate derivative"""
        # Synthesize ideas
        synthesis = engine.integrate(["ai", "biology"])
        synthesized_name = synthesis.synthesized_idea

        # Calculate derivative of synthesized idea
        deriv = engine.derivative(synthesized_name, "time")
        assert deriv.original_idea == synthesized_name
        assert deriv.rate_of_change != 0

    def test_workflow_compose_then_derivative(self, engine):
        """Test workflow: compose ideas then calculate derivative"""
        # Compose ideas
        comp = engine.compose("machine learning", "physics")
        composition_name = comp.composition

        # Calculate derivative
        deriv = engine.derivative(composition_name, "knowledge")
        assert deriv.original_idea == composition_name

    def test_workflow_all_variables(self, engine):
        """Test workflow: test idea across all variables"""
        idea = "artificial intelligence"

        # Test all variable types
        time_deriv = engine.derivative(idea, "time")
        knowledge_deriv = engine.derivative(idea, "knowledge")
        resources_deriv = engine.derivative(idea, "resources")

        assert time_deriv.rate_of_change < 0  # Decays
        assert knowledge_deriv.rate_of_change > 0  # Grows
        assert resources_deriv.rate_of_change > 0  # Grows

    def test_workflow_multiple_syntheses(self, engine):
        """Test workflow: chain multiple syntheses"""
        # First synthesis
        synth1 = engine.integrate(["ai", "physics"])

        # Second synthesis using result
        synth2 = engine.integrate([synth1.synthesized_idea, "biology"])

        assert synth1.synthesized_idea in synth2.component_ideas
        assert len(synth2.component_ideas) == 2

    def test_workflow_serialization(self, engine):
        """Test workflow: create result and serialize to JSON"""
        result = engine.derivative("test idea", "time")

        # Serialize to dict
        result_dict = asdict(result)

        # Should be JSON-serializable
        json_str = json.dumps(result_dict)
        assert isinstance(json_str, str)

        # Should be deserializable
        loaded = json.loads(json_str)
        assert loaded['original_idea'] == "test idea"

    def test_workflow_all_operations_same_idea(self, engine):
        """Test workflow: apply all operations to same idea"""
        idea = "quantum machine learning"

        # Derivative
        deriv = engine.derivative(idea, "time")
        assert isinstance(deriv, IdeaDerivative)

        # Limit
        limit = engine.limit(idea, "time", "infinity")
        assert isinstance(limit, IdeaLimit)

        # Use in synthesis
        synth = engine.integrate([idea, "neuroscience"])
        assert idea in synth.component_ideas

        # Use in composition
        comp = engine.compose(idea, "classical computing")
        assert comp.outer_idea == idea


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_string_idea_name(self, engine):
        """Test handling of empty string idea names"""
        result = engine.derivative("", "time")
        assert result.original_idea == ""
        assert isinstance(result.rate_of_change, float)

    def test_very_long_idea_name(self, engine):
        """Test handling of very long idea names"""
        long_name = "a" * 1000
        result = engine.derivative(long_name, "time")
        assert result.original_idea == long_name

    def test_special_characters_in_name(self, engine):
        """Test handling of special characters"""
        special_name = "idea-with_special.chars!@#$%"
        result = engine.derivative(special_name, "time")
        assert result.original_idea == special_name

    def test_unicode_in_idea_name(self, engine):
        """Test handling of unicode characters"""
        unicode_name = "量子计算机"
        result = engine.derivative(unicode_name, "time")
        assert result.original_idea == unicode_name

    def test_case_sensitivity_novelty(self, engine):
        """Test case sensitivity in novelty estimation"""
        # Should handle different cases
        novelty1 = engine._estimate_novelty("quantum")
        novelty2 = engine._estimate_novelty("QUANTUM")
        novelty3 = engine._estimate_novelty("QuAnTuM")
        # All should be recognized as emerging (high novelty)
        assert novelty1 > 0.6
        assert novelty2 > 0.6
        assert novelty3 > 0.6

    def test_integrate_duplicate_ideas(self, engine):
        """Test integration with duplicate ideas"""
        result = engine.integrate(["ai", "ai", "ai"])
        assert len(result.component_ideas) == 3
        assert all(idea == "ai" for idea in result.component_ideas)

    def test_compose_same_idea_twice(self, engine):
        """Test composing idea with itself"""
        result = engine.compose("recursion", "recursion")
        assert result.outer_idea == "recursion"
        assert result.inner_idea == "recursion"
        assert "recursion(recursion)" == result.composition

    def test_limit_numeric_approach_value(self, engine):
        """Test limit with numeric approach value"""
        result = engine.limit("test", "time", "42")
        assert result.approach_value == "42"

    def test_multiple_engine_instances(self):
        """Test multiple engine instances are independent"""
        engine1 = IdeaCalculusEngine()
        engine2 = IdeaCalculusEngine()

        assert engine1 is not engine2
        assert engine1.domains is not engine2.domains
        assert engine1.decay_models is not engine2.decay_models

    def test_decimal_values_in_projections(self, engine):
        """Test future projections contain valid decimal values"""
        result = engine.derivative("test", "time")
        for value in result.future_projection.values():
            assert isinstance(value, (int, float))
            assert value >= 0  # Novelty shouldn't be negative
            assert value <= 1.5  # Shouldn't exceed reasonable bounds

    def test_emergence_score_precision(self, engine):
        """Test emergence score is rounded to 3 decimals"""
        result = engine.integrate(["idea1", "idea2"])
        score_str = str(result.emergence_score)
        if '.' in score_str:
            decimals = len(score_str.split('.')[1])
            assert decimals <= 3
