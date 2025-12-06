"""
UARO Test Suite - Explainability Engine

Tests for explainability engine and proof document generation.
Validates all export formats (Markdown, HTML, LaTeX, JSON).
"""

import pytest
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uaro import (
    ExplainabilityEngine,
    ProofDocument,
    explain_solution,
    solve_with_uaro,
    Problem,
)


class SimpleProblem(Problem):
    """Simple test problem"""

    def initial_state(self):
        return 0

    def goal_test(self, state):
        return state == 5

    def actions(self, state):
        return [("increment", state + 1)]

    def result(self, state, action):
        return action[1]

    def cost(self, state, action):
        return 1


@pytest.fixture
def sample_solution():
    """Generate sample solution for testing"""
    problem = SimpleProblem()
    return solve_with_uaro(problem, max_iterations=20)


def test_explainability_engine_initialization():
    """Test ExplainabilityEngine initialization"""
    engine = ExplainabilityEngine()

    assert engine is not None
    assert hasattr(engine, "generate_proof_document")


def test_proof_document_generation(sample_solution):
    """Test proof document generation from solution"""
    engine = ExplainabilityEngine()

    doc = engine.generate_proof_document(sample_solution)

    assert isinstance(doc, ProofDocument)
    assert doc.title is not None
    assert doc.problem_description is not None
    assert doc.solution_summary is not None
    assert len(doc.reasoning_trace) > 0
    assert doc.metadata is not None


def test_proof_document_metadata(sample_solution):
    """Test proof document metadata completeness"""
    engine = ExplainabilityEngine()
    doc = engine.generate_proof_document(sample_solution)

    assert "iterations" in doc.metadata
    assert "duration_seconds" in doc.metadata
    assert "primitives_used" in doc.metadata
    assert "final_confidence" in doc.metadata
    assert "success" in doc.metadata


def test_markdown_export(sample_solution):
    """Test Markdown export format"""
    engine = ExplainabilityEngine()
    doc = engine.generate_proof_document(sample_solution)

    markdown = engine.export_markdown(doc)

    assert isinstance(markdown, str)
    assert len(markdown) > 0
    assert "# " in markdown  # Has headers
    assert "**" in markdown  # Has bold text
    assert "```" in markdown or "- " in markdown  # Has code blocks or lists


def test_html_export(sample_solution):
    """Test HTML export format"""
    engine = ExplainabilityEngine()
    doc = engine.generate_proof_document(sample_solution)

    html = engine.export_html(doc)

    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html
    assert "<html>" in html
    assert "</html>" in html
    assert "<body>" in html


def test_json_export(sample_solution):
    """Test JSON export format"""
    engine = ExplainabilityEngine()
    doc = engine.generate_proof_document(sample_solution)

    json_str = engine.export_json(doc)

    assert isinstance(json_str, str)

    parsed = json.loads(json_str)
    assert "title" in parsed
    assert "reasoning_trace" in parsed
    assert "metadata" in parsed


def test_latex_export(sample_solution):
    """Test LaTeX export format"""
    engine = ExplainabilityEngine()
    doc = engine.generate_proof_document(sample_solution)

    latex = engine.export_latex(doc)

    assert isinstance(latex, str)
    assert r"\documentclass" in latex
    assert r"\begin{document}" in latex
    assert r"\end{document}" in latex
    assert r"\section" in latex


def test_reasoning_trace_documentation(sample_solution):
    """Test reasoning trace is properly documented"""
    engine = ExplainabilityEngine()
    doc = engine.generate_proof_document(sample_solution)

    assert len(doc.reasoning_trace) > 0

    for step in doc.reasoning_trace:
        assert "step_number" in step
        assert "primitive_name" in step
        assert "success" in step
        assert "reasoning" in step
        assert "confidence" in step


def test_confidence_analysis(sample_solution):
    """Test confidence analysis in proof document"""
    engine = ExplainabilityEngine()
    doc = engine.generate_proof_document(sample_solution)

    assert "overall" in doc.confidence_analysis
    assert 0 <= doc.confidence_analysis["overall"] <= 1


def test_validation_results(sample_solution):
    """Test validation results documentation"""
    engine = ExplainabilityEngine()
    doc = engine.generate_proof_document(sample_solution)

    assert len(doc.validation_results) > 0

    for validation in doc.validation_results:
        assert "name" in validation
        assert "result" in validation
        assert "passed" in validation


def test_limitations_identification(sample_solution):
    """Test limitations are identified"""
    engine = ExplainabilityEngine()
    doc = engine.generate_proof_document(sample_solution)

    assert isinstance(doc.limitations, list)


def test_convenience_function_markdown(sample_solution):
    """Test explain_solution convenience function with markdown"""
    output = explain_solution(sample_solution, format="markdown")

    assert isinstance(output, str)
    assert len(output) > 0


def test_convenience_function_json(sample_solution):
    """Test explain_solution convenience function with JSON"""
    output = explain_solution(sample_solution, format="json")

    assert isinstance(output, str)
    parsed = json.loads(output)
    assert "title" in parsed


def test_custom_title(sample_solution):
    """Test custom title in proof document"""
    engine = ExplainabilityEngine()

    custom_title = "Custom Test Title"
    doc = engine.generate_proof_document(sample_solution, title=custom_title)

    assert doc.title == custom_title


def test_include_alternatives_flag(sample_solution):
    """Test include_alternatives flag"""
    engine = ExplainabilityEngine()

    doc_with_alts = engine.generate_proof_document(
        sample_solution, include_alternatives=True
    )
    doc_without_alts = engine.generate_proof_document(
        sample_solution, include_alternatives=False
    )

    assert len(doc_with_alts.reasoning_trace) > 0
    assert len(doc_without_alts.reasoning_trace) > 0


def test_proof_document_timestamp():
    """Test proof document has timestamp"""
    from uaro import SolutionResult

    mock_result = SolutionResult(
        problem=SimpleProblem(),
        solution=5,
        success=True,
        iterations=10,
        duration_seconds=1.0,
        confidence=0.9,
    )

    engine = ExplainabilityEngine()
    doc = engine.generate_proof_document(mock_result)

    assert doc.created_at is not None
    assert len(doc.created_at) > 0


def test_export_to_file(sample_solution, tmp_path):
    """Test exporting proof document to file"""
    output_file = tmp_path / "proof.md"

    result = explain_solution(
        sample_solution, format="markdown", output_file=str(output_file)
    )

    assert output_file.exists()
    content = output_file.read_text()
    assert len(content) > 0
    assert content == result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
