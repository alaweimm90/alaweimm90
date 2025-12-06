"""
Comprehensive test suite for Causal Reasoning Engine.

Tests causal graph construction, chain discovery, counterfactual analysis,
and intervention identification.
"""

import pytest
from typing import Dict, Any
from atlas_core.causal_engine import (
    CausalReasoningEngine,
    CausalNode,
    CausalEdge,
    CausalChain,
    CausalRelationType,
    create_causal_engine,
)


class TestCausalNode:
    """Test CausalNode functionality."""

    def test_node_creation(self):
        """Test creating a causal node."""
        node = CausalNode(
            id="test_node",
            name="Test Node",
            type="problem_property",
        )
        assert node.id == "test_node"
        assert node.name == "Test Node"
        assert node.type == "problem_property"

    def test_node_with_properties(self):
        """Test node with properties."""
        node = CausalNode(
            id="size",
            name="Problem Size",
            type="problem_property",
            properties={"min": 10, "max": 100},
        )
        assert node.properties["min"] == 10
        assert node.properties["max"] == 100


class TestCausalEdge:
    """Test CausalEdge functionality."""

    def test_edge_creation(self):
        """Test creating a causal edge."""
        edge = CausalEdge(
            source="node_a",
            target="node_b",
            relation_type=CausalRelationType.CAUSES,
            strength=0.85,
        )
        assert edge.source == "node_a"
        assert edge.target == "node_b"
        assert edge.strength == 0.85
        assert edge.relation_type == CausalRelationType.CAUSES

    def test_edge_with_evidence(self):
        """Test edge with evidence."""
        edge = CausalEdge(
            source="a",
            target="b",
            relation_type=CausalRelationType.CAUSES,
            strength=0.9,
            evidence=["empirical_study_1", "theoretical_proof"],
        )
        assert len(edge.evidence) == 2
        assert "empirical_study_1" in edge.evidence


class TestCausalChain:
    """Test CausalChain functionality."""

    def test_chain_creation(self):
        """Test creating a causal chain."""
        chain = CausalChain(
            chain_id="chain_1",
            nodes=["a", "b", "c"],
            strength=0.85,
            explanation="A causes B causes C",
        )
        assert chain.chain_id == "chain_1"
        assert len(chain.nodes) == 3
        assert chain.strength == 0.85


class TestCausalReasoningEngine:
    """Test CausalReasoningEngine functionality."""

    @pytest.fixture
    def engine(self):
        """Create a causal reasoning engine."""
        return create_causal_engine()

    def test_engine_initialization(self, engine):
        """Test engine initializes with knowledge base."""
        assert len(engine.nodes) > 0
        assert len(engine.edges) > 0
        # Should have key nodes
        assert "problem_size" in engine.nodes
        assert "search_space_size" in engine.nodes
        assert "computational_cost" in engine.nodes

    def test_analyze_qap_problem(self, engine):
        """Test analyzing a QAP problem."""
        problem = {
            "type": "qap",
            "size": 30,
            "objectives": 1,
            "constraints": [
                {"type": "assignment"},
                {"type": "capacity", "max": 100},
            ],
        }
        analysis = engine.analyze_problem(problem)

        assert "causal_chains" in analysis
        assert "explanations" in analysis
        assert "interventions" in analysis
        assert "counterfactuals" in analysis
        assert "key_insights" in analysis  # Changed from 'insights'

        # Should have at least one causal chain
        assert len(analysis["causal_chains"]) > 0

        # Should have explanations
        assert len(analysis["explanations"]) > 0

        # Should have counterfactuals
        assert len(analysis["counterfactuals"]) > 0

    def test_analyze_tsp_problem(self, engine):
        """Test analyzing a TSP problem."""
        problem = {
            "type": "tsp",
            "size": 50,
            "objectives": 1,
            "constraints": [],
        }
        analysis = engine.analyze_problem(problem)

        assert "causal_chains" in analysis
        assert len(analysis["causal_chains"]) > 0

    def test_analyze_knapsack_problem(self, engine):
        """Test analyzing a knapsack problem."""
        problem = {
            "type": "knapsack",
            "size": 100,
            "objectives": 1,
            "constraints": [{"type": "capacity", "max": 500}],
        }
        analysis = engine.analyze_problem(problem)

        assert "causal_chains" in analysis
        assert len(analysis["causal_chains"]) > 0

    def test_generate_causal_report(self, engine):
        """Test generating a complete causal report."""
        problem = {
            "type": "qap",
            "size": 30,
            "objectives": 1,
            "constraints": [],
        }
        report = engine.generate_causal_report(problem)

        # Report should be a non-empty string
        assert isinstance(report, str)
        assert len(report) > 100

        # Should contain key sections
        assert "CAUSAL REASONING REPORT" in report
        assert "PROBLEM ANALYSIS" in report
        assert "CAUSAL CHAINS" in report

    def test_problem_size_categories(self, engine):
        """Test problem size categorization."""
        small_problem = {"type": "qap", "size": 15, "objectives": 1, "constraints": []}
        medium_problem = {"type": "qap", "size": 35, "objectives": 1, "constraints": []}
        large_problem = {"type": "qap", "size": 80, "objectives": 1, "constraints": []}

        small_analysis = engine.analyze_problem(small_problem)
        medium_analysis = engine.analyze_problem(medium_problem)
        large_analysis = engine.analyze_problem(large_problem)

        # All should have valid analyses
        assert len(small_analysis["causal_chains"]) > 0
        assert len(medium_analysis["causal_chains"]) > 0
        assert len(large_analysis["causal_chains"]) > 0

    def test_constraint_analysis(self, engine):
        """Test analysis with different constraint counts."""
        no_constraints = {
            "type": "qap",
            "size": 30,
            "objectives": 1,
            "constraints": [],
        }
        many_constraints = {
            "type": "qap",
            "size": 30,
            "objectives": 1,
            "constraints": [
                {"type": "assignment"},
                {"type": "capacity", "max": 100},
                {"type": "distance", "threshold": 50},
                {"type": "flow", "min": 10},
            ],
        }

        no_const_analysis = engine.analyze_problem(no_constraints)
        many_const_analysis = engine.analyze_problem(many_constraints)

        # Both should have valid analyses
        assert len(no_const_analysis["causal_chains"]) > 0
        assert len(many_const_analysis["causal_chains"]) > 0

    def test_multi_objective_analysis(self, engine):
        """Test analysis of multi-objective problems."""
        problem = {
            "type": "qap",
            "size": 30,
            "objectives": 3,
            "constraints": [],
        }
        analysis = engine.analyze_problem(problem)

        assert len(analysis["causal_chains"]) > 0

    def test_insights_generation(self, engine):
        """Test that insights are generated."""
        problem = {
            "type": "qap",
            "size": 45,
            "objectives": 1,
            "constraints": [],
        }
        analysis = engine.analyze_problem(problem)

        assert "key_insights" in analysis
        assert len(analysis["key_insights"]) > 0
        # Insights should be meaningful strings
        for insight in analysis["key_insights"]:
            assert isinstance(insight, str)
            assert len(insight) > 20

    def test_confidence_scores(self, engine):
        """Test that confidence scores are valid."""
        problem = {
            "type": "qap",
            "size": 30,
            "objectives": 1,
            "constraints": [],
        }
        analysis = engine.analyze_problem(problem)

        # Counterfactuals should have confidence scores
        for cf in analysis["counterfactuals"]:
            assert 0.0 <= cf["confidence"] <= 1.0

    def test_add_node(self, engine):
        """Test adding a node to the graph."""
        initial_count = len(engine.nodes)
        node = CausalNode(
            id="custom_test_node",
            name="Custom Node",
            type="algorithm_property",
        )
        engine._add_node(node)
        assert len(engine.nodes) == initial_count + 1
        assert "custom_test_node" in engine.nodes

    def test_add_edge(self, engine):
        """Test adding an edge to the graph."""
        initial_count = len(engine.edges)
        edge = CausalEdge(
            source="problem_size",
            target="search_space_size",
            relation_type=CausalRelationType.CAUSES,
            strength=0.95,
        )
        engine._add_edge(edge)
        assert len(engine.edges) >= initial_count


class TestIntegration:
    """Integration tests for causal reasoning."""

    def test_end_to_end_analysis(self):
        """Test complete end-to-end causal analysis."""
        engine = create_causal_engine()

        problem = {
            "type": "qap",
            "size": 40,
            "objectives": 1,
            "constraints": [
                {"type": "assignment"},
                {"type": "capacity", "max": 120},
            ],
        }

        # Perform complete analysis
        analysis = engine.analyze_problem(problem)
        report = engine.generate_causal_report(problem)

        # Verify all components present
        assert "causal_chains" in analysis
        assert "explanations" in analysis
        assert "interventions" in analysis
        assert "counterfactuals" in analysis
        assert "key_insights" in analysis

        # Verify report quality
        assert len(report) > 500
        assert "CAUSAL CHAINS" in report
        assert "COUNTERFACTUAL ANALYSIS" in report

    def test_multiple_problem_types(self):
        """Test causal analysis across different problem types."""
        engine = create_causal_engine()

        problems = [
            {"type": "qap", "size": 30, "objectives": 1, "constraints": []},
            {"type": "tsp", "size": 50, "objectives": 1, "constraints": []},
            {"type": "knapsack", "size": 100, "objectives": 1, "constraints": []},
            {"type": "scheduling", "size": 20, "objectives": 2, "constraints": []},
        ]

        for problem in problems:
            analysis = engine.analyze_problem(problem)
            assert len(analysis["causal_chains"]) > 0
            assert len(analysis["explanations"]) > 0

    def test_causal_reasoning_consistency(self):
        """Test that causal reasoning is consistent across calls."""
        engine = create_causal_engine()

        problem = {
            "type": "qap",
            "size": 35,
            "objectives": 1,
            "constraints": [],
        }

        # Run analysis twice
        analysis1 = engine.analyze_problem(problem)
        analysis2 = engine.analyze_problem(problem)

        # Results should be consistent
        assert len(analysis1["causal_chains"]) == len(analysis2["causal_chains"])
        assert len(analysis1["explanations"]) == len(analysis2["explanations"])


@pytest.mark.integration
def test_causal_engine_performance():
    """Test performance of causal engine."""
    import time

    engine = create_causal_engine()
    problem = {
        "type": "qap",
        "size": 50,
        "objectives": 1,
        "constraints": [{"type": "assignment"}],
    }

    start = time.time()
    analysis = engine.analyze_problem(problem)
    elapsed = time.time() - start

    # Should be fast (< 0.1 seconds)
    assert elapsed < 0.1
    assert len(analysis["causal_chains"]) > 0


def test_create_causal_engine():
    """Test the factory function."""
    engine = create_causal_engine()
    assert isinstance(engine, CausalReasoningEngine)
    assert len(engine.nodes) > 0
    assert len(engine.edges) > 0


@pytest.mark.unit
def test_causal_chain_structure():
    """Test that causal chains have proper structure."""
    engine = create_causal_engine()
    problem = {
        "type": "qap",
        "size": 30,
        "objectives": 1,
        "constraints": [],
    }
    analysis = engine.analyze_problem(problem)

    for chain_dict in analysis["causal_chains"]:
        assert "chain_id" in chain_dict
        assert "path" in chain_dict
        assert "strength" in chain_dict
        assert "explanation" in chain_dict
        # Strength should be between 0 and 1
        assert 0.0 <= chain_dict["strength"] <= 1.0


@pytest.mark.unit
def test_counterfactual_structure():
    """Test that counterfactuals have proper structure."""
    engine = create_causal_engine()
    problem = {
        "type": "qap",
        "size": 30,
        "objectives": 1,
        "constraints": [],
    }
    analysis = engine.analyze_problem(problem)

    for cf in analysis["counterfactuals"]:
        assert "scenario" in cf
        assert "intervention" in cf
        assert "predicted_outcome" in cf
        assert "confidence" in cf
        assert 0.0 <= cf["confidence"] <= 1.0
