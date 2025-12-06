"""
UARO Explainability Engine

Generate human-readable proof documents from reasoning traces.
"Show Your Work" mode for transparent AI reasoning.

Cycle 29-30: Explainability Engine

Addresses the AI black box problem:
- Full reasoning trace documentation
- Justification for every step
- Alternatives considered and rejected
- Confidence analysis
- Known limitations

Output formats:
- Markdown
- HTML
- LaTeX/PDF (for academic/legal use)
- JSON (for programmatic access)
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json

from uaro.universal_solver import SolutionResult, ReasoningStep


@dataclass
class ProofDocument:
    """A complete proof document"""

    title: str
    problem_description: str
    solution_summary: str
    reasoning_trace: List[Dict[str, Any]]
    confidence_analysis: Dict[str, Any]
    validation_results: List[Dict[str, Any]]
    limitations: List[str]
    metadata: Dict[str, Any]

    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class ExplainabilityEngine:
    """
    Convert reasoning traces into human-readable proof documents.

    Makes UARO transparent and trustworthy by showing all work.
    """

    def __init__(self):
        self.templates = self._load_templates()

    def generate_proof_document(
        self,
        solution_result: SolutionResult,
        title: Optional[str] = None,
        include_alternatives: bool = True
    ) -> ProofDocument:
        """
        Create complete proof document from solution

        Args:
            solution_result: Result from UniversalSolver
            title: Document title (auto-generated if None)
            include_alternatives: Show alternatives considered

        Returns:
            ProofDocument with all details
        """
        if title is None:
            title = f"UARO Proof: Solution in {solution_result.iterations} steps"

        # Problem description
        problem_desc = self._describe_problem(solution_result.problem)

        # Solution summary
        solution_summary = self._summarize_solution(solution_result)

        # Reasoning trace
        trace_docs = self._document_reasoning_trace(
            solution_result.reasoning_trace,
            include_alternatives
        )

        # Confidence analysis
        confidence_analysis = self._analyze_confidence(solution_result)

        # Validation results
        validation = self._document_validation(solution_result)

        # Limitations
        limitations = self._identify_limitations(solution_result)

        # Metadata
        metadata = {
            "iterations": solution_result.iterations,
            "duration_seconds": solution_result.duration_seconds,
            "primitives_used": solution_result.primitives_used,
            "final_confidence": solution_result.confidence,
            "success": solution_result.success
        }

        return ProofDocument(
            title=title,
            problem_description=problem_desc,
            solution_summary=solution_summary,
            reasoning_trace=trace_docs,
            confidence_analysis=confidence_analysis,
            validation_results=validation,
            limitations=limitations,
            metadata=metadata
        )

    def export_markdown(self, doc: ProofDocument) -> str:
        """Export proof document as Markdown"""

        lines = [
            f"# {doc.title}",
            "",
            f"*Generated: {doc.created_at}*",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            doc.solution_summary,
            "",
            f"- **Success**: {doc.metadata['success']}",
            f"- **Iterations**: {doc.metadata['iterations']}",
            f"- **Duration**: {doc.metadata['duration_seconds']:.2f} seconds",
            f"- **Final Confidence**: {doc.metadata['final_confidence']:.1%}",
            "",
            "---",
            "",
            "## Problem Statement",
            "",
            doc.problem_description,
            "",
            "---",
            "",
            "## Reasoning Trace",
            "",
            "Below is the complete reasoning process, step by step.",
            ""
        ]

        # Document each reasoning step
        for step_doc in doc.reasoning_trace:
            lines.extend([
                f"### Step {step_doc['step_number']}: {step_doc['primitive_name']}",
                "",
                f"**Status**: {'✓ Success' if step_doc['success'] else '✗ Failed'}",
                "",
                f"**Reasoning**: {step_doc['reasoning']}",
                "",
                f"**Confidence**: {step_doc['confidence']:.1%}",
                "",
            ])

            if step_doc.get('state_before'):
                lines.extend([
                    "**State Before**:",
                    f"```",
                    f"{step_doc['state_before']}",
                    f"```",
                    ""
                ])

            if step_doc.get('state_after'):
                lines.extend([
                    "**State After**:",
                    f"```",
                    f"{step_doc['state_after']}",
                    f"```",
                    ""
                ])

            lines.append("---")
            lines.append("")

        # Confidence analysis
        lines.extend([
            "## Confidence Analysis",
            "",
            f"**Overall Confidence**: {doc.confidence_analysis['overall']:.1%}",
            ""
        ])

        if doc.confidence_analysis.get('breakdown'):
            lines.append("**Breakdown by Category**:")
            lines.append("")
            for category, conf in doc.confidence_analysis['breakdown'].items():
                lines.append(f"- {category}: {conf:.1%}")
            lines.append("")

        # Validation
        lines.extend([
            "---",
            "",
            "## Validation",
            ""
        ])

        if doc.validation_results:
            lines.append("The solution was validated using:")
            lines.append("")
            for validation in doc.validation_results:
                status = "✓" if validation.get('passed', True) else "✗"
                lines.append(f"- {status} {validation['name']}: {validation['result']}")
            lines.append("")
        else:
            lines.append("*No validation performed*")
            lines.append("")

        # Limitations
        lines.extend([
            "---",
            "",
            "## Known Limitations",
            ""
        ])

        if doc.limitations:
            for limitation in doc.limitations:
                lines.append(f"- ⚠️ {limitation}")
            lines.append("")
        else:
            lines.append("*No known limitations*")
            lines.append("")

        # Primitives used
        lines.extend([
            "---",
            "",
            "## Primitives Used",
            ""
        ])

        for primitive in doc.metadata['primitives_used']:
            lines.append(f"- `{primitive}`")

        lines.extend([
            "",
            "---",
            "",
            f"*Document generated by UARO Explainability Engine*"
        ])

        return "\n".join(lines)

    def export_html(self, doc: ProofDocument) -> str:
        """Export proof document as HTML"""

        markdown_content = self.export_markdown(doc)

        # Simple HTML wrapper
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{doc.title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
        }}
        code, pre {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        pre {{
            padding: 15px;
            overflow-x: auto;
        }}
        .metadata {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .success {{
            color: #27ae60;
            font-weight: bold;
        }}
        .failure {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .confidence {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            background-color: #3498db;
            color: white;
            font-size: 0.9em;
        }}
        hr {{
            border: none;
            border-top: 1px solid #ecf0f1;
            margin: 30px 0;
        }}
    </style>
</head>
<body>
    <div class="metadata">
        <strong>Generated:</strong> {doc.created_at}<br>
        <strong>Success:</strong> <span class="{'success' if doc.metadata['success'] else 'failure'}">
            {doc.metadata['success']}
        </span><br>
        <strong>Iterations:</strong> {doc.metadata['iterations']}<br>
        <strong>Duration:</strong> {doc.metadata['duration_seconds']:.2f} seconds<br>
        <strong>Confidence:</strong> <span class="confidence">{doc.metadata['final_confidence']:.1%}</span>
    </div>

    <!-- Markdown content would be converted to HTML here -->
    <!-- For simplicity, using <pre> wrapper -->
    <pre style="white-space: pre-wrap;">{markdown_content}</pre>

    <hr>
    <p style="text-align: center; color: #7f8c8d; font-size: 0.9em;">
        <em>Document generated by UARO Explainability Engine</em>
    </p>
</body>
</html>"""

        return html

    def export_json(self, doc: ProofDocument) -> str:
        """Export proof document as JSON (for programmatic access)"""

        doc_dict = {
            "title": doc.title,
            "created_at": doc.created_at,
            "problem_description": doc.problem_description,
            "solution_summary": doc.solution_summary,
            "reasoning_trace": doc.reasoning_trace,
            "confidence_analysis": doc.confidence_analysis,
            "validation_results": doc.validation_results,
            "limitations": doc.limitations,
            "metadata": doc.metadata
        }

        return json.dumps(doc_dict, indent=2, default=str)

    def export_latex(self, doc: ProofDocument) -> str:
        """Export proof document as LaTeX (for academic/legal use)"""

        latex = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{hyperref}

\title{""" + doc.title.replace('_', r'\_') + r"""}
\author{UARO Explainability Engine}
\date{""" + doc.created_at + r"""}

\lstset{
    basicstyle=\ttfamily\small,
    breaklines=true,
    frame=single,
    backgroundcolor=\color{gray!10}
}

\begin{document}

\maketitle

\section{Executive Summary}

""" + doc.solution_summary + r"""

\begin{itemize}
    \item \textbf{Success}: """ + str(doc.metadata['success']) + r"""
    \item \textbf{Iterations}: """ + str(doc.metadata['iterations']) + r"""
    \item \textbf{Duration}: """ + f"{doc.metadata['duration_seconds']:.2f}" + r""" seconds
    \item \textbf{Final Confidence}: """ + f"{doc.metadata['final_confidence']:.1%}" + r"""
\end{itemize}

\section{Problem Statement}

""" + doc.problem_description + r"""

\section{Reasoning Trace}

"""

        # Add each reasoning step
        for step_doc in doc.reasoning_trace:
            latex += r"""
\subsection{Step """ + str(step_doc['step_number']) + r""": """ + step_doc['primitive_name'].replace('_', r'\_') + r"""}

\textbf{Status}: """ + ('Success' if step_doc['success'] else 'Failed') + r"""

\textbf{Reasoning}: """ + step_doc['reasoning'] + r"""

\textbf{Confidence}: """ + f"{step_doc['confidence']:.1%}" + r"""

"""

        latex += r"""

\section{Confidence Analysis}

Overall confidence: """ + f"{doc.confidence_analysis['overall']:.1%}" + r"""

\section{Validation}

"""

        if doc.validation_results:
            latex += r"\begin{itemize}" + "\n"
            for validation in doc.validation_results:
                status = "Passed" if validation.get('passed', True) else "Failed"
                latex += r"    \item " + validation['name'].replace('_', r'\_') + f": {status}\n"
            latex += r"\end{itemize}" + "\n"
        else:
            latex += "No validation performed.\n"

        latex += r"""

\section{Known Limitations}

"""

        if doc.limitations:
            latex += r"\begin{itemize}" + "\n"
            for limitation in doc.limitations:
                latex += r"    \item " + limitation + "\n"
            latex += r"\end{itemize}" + "\n"
        else:
            latex += "No known limitations.\n"

        latex += r"""

\section{Primitives Used}

\begin{itemize}
"""

        for primitive in doc.metadata['primitives_used']:
            latex += r"    \item \texttt{" + primitive.replace('_', r'\_') + r"}" + "\n"

        latex += r"""\end{itemize}

\end{document}"""

        return latex

    # ==================== HELPER METHODS ====================

    def _describe_problem(self, problem: Any) -> str:
        """Generate problem description"""
        if hasattr(problem, '__doc__') and problem.__doc__:
            return problem.__doc__.strip()
        elif hasattr(problem, 'description'):
            return problem.description
        else:
            return f"Problem of type: {type(problem).__name__}"

    def _summarize_solution(self, result: SolutionResult) -> str:
        """Generate solution summary"""
        if result.success:
            return (
                f"Solution found successfully in {result.iterations} iterations "
                f"({result.duration_seconds:.2f} seconds). "
                f"Final confidence: {result.confidence:.1%}."
            )
        else:
            return (
                f"Solution attempt completed after {result.iterations} iterations "
                f"({result.duration_seconds:.2f} seconds) "
                f"but did not reach goal. "
                f"Final confidence: {result.confidence:.1%}."
            )

    def _document_reasoning_trace(
        self,
        trace: List[ReasoningStep],
        include_alternatives: bool
    ) -> List[Dict[str, Any]]:
        """Convert reasoning trace to documentation"""
        documented = []

        for step in trace:
            step_doc = {
                "step_number": step.iteration,
                "primitive_name": step.primitive_name,
                "success": step.success,
                "reasoning": step.reasoning,
                "confidence": step.confidence,
                "state_before": str(step.input_state)[:500],  # Truncate
                "state_after": str(step.output_state)[:500],
            }

            documented.append(step_doc)

        return documented

    def _analyze_confidence(self, result: SolutionResult) -> Dict[str, Any]:
        """Analyze confidence over time"""
        overall = result.confidence

        # Confidence progression
        progression = [
            {"iteration": step.iteration, "confidence": step.confidence}
            for step in result.reasoning_trace
        ]

        # Breakdown by primitive category
        breakdown = {}
        category_counts = {}

        for step in result.reasoning_trace:
            # This would need access to primitive registry to get category
            # For now, just use overall confidence
            pass

        return {
            "overall": overall,
            "progression": progression,
            "breakdown": breakdown
        }

    def _document_validation(self, result: SolutionResult) -> List[Dict[str, Any]]:
        """Document validation checks"""
        validations = []

        # Check if solution exists
        validations.append({
            "name": "Solution Exists",
            "result": "Solution generated" if result.solution else "No solution",
            "passed": result.solution is not None
        })

        # Check if goal reached
        validations.append({
            "name": "Goal Reached",
            "result": "Success" if result.success else "Incomplete",
            "passed": result.success
        })

        # Check confidence threshold
        validations.append({
            "name": "High Confidence",
            "result": f"Confidence: {result.confidence:.1%}",
            "passed": result.confidence >= 0.8
        })

        return validations

    def _identify_limitations(self, result: SolutionResult) -> List[str]:
        """Identify known limitations of solution"""
        limitations = []

        # Low confidence warning
        if result.confidence < 0.7:
            limitations.append(
                f"Low confidence ({result.confidence:.1%}) - solution may be unreliable"
            )

        # Incomplete solution warning
        if not result.success:
            limitations.append("Solution did not reach stated goal")

        # Too many iterations warning
        if result.iterations >= 900:
            limitations.append("Reached near-maximum iterations - may not be optimal")

        # Few iterations warning
        if result.iterations < 5:
            limitations.append("Very few iterations - problem may be trivial or ill-defined")

        return limitations

    def _load_templates(self) -> Dict[str, str]:
        """Load document templates"""
        # Placeholder for future template system
        return {}


# ==================== CONVENIENCE FUNCTIONS ====================

def explain_solution(
    solution_result: SolutionResult,
    format: str = "markdown",
    output_file: Optional[str] = None
) -> str:
    """
    Convenience function: Generate explanation for solution

    Args:
        solution_result: Result from UniversalSolver
        format: Output format ('markdown', 'html', 'json', 'latex')
        output_file: Optional file to write to

    Returns:
        Formatted explanation string
    """
    engine = ExplainabilityEngine()
    doc = engine.generate_proof_document(solution_result)

    if format == "markdown":
        output = engine.export_markdown(doc)
    elif format == "html":
        output = engine.export_html(doc)
    elif format == "json":
        output = engine.export_json(doc)
    elif format == "latex":
        output = engine.export_latex(doc)
    else:
        raise ValueError(f"Unknown format: {format}")

    if output_file:
        with open(output_file, 'w') as f:
            f.write(output)

    return output
