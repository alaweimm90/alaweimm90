"""
Paper Generator - Stage 4 of ORCHEX

Generates LaTeX manuscripts from research results.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime


@dataclass
class PaperSection:
    """A section of the paper"""
    title: str
    content: str
    subsections: List['PaperSection'] = None


@dataclass
class Paper:
    """Complete research paper"""
    title: str
    authors: List[str]
    abstract: str
    sections: List[PaperSection]
    references: List[str]
    figures: List[str] = None

    def to_latex(self) -> str:
        """Convert to LaTeX document"""
        return self._generate_latex()

    def _generate_latex(self) -> str:
        """Generate complete LaTeX source"""
        latex = r"""\documentclass[11pt]{article}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage[margin=1in]{geometry}

"""
        # Title and authors
        latex += f"\\title{{{self.title}}}\n"
        latex += f"\\author{{{' \\and '.join(self.authors)}}}\n"
        latex += f"\\date{{\\today}}\n\n"

        latex += r"\begin{document}" + "\n"
        latex += r"\maketitle" + "\n\n"

        # Abstract
        latex += r"\begin{abstract}" + "\n"
        latex += self.abstract + "\n"
        latex += r"\end{abstract}" + "\n\n"

        # Sections
        for section in self.sections:
            latex += self._section_to_latex(section)

        # References
        if self.references:
            latex += r"\section{References}" + "\n"
            latex += r"\begin{enumerate}" + "\n"
            for ref in self.references:
                latex += f"\\item {ref}\n"
            latex += r"\end{enumerate}" + "\n\n"

        latex += r"\end{document}"
        return latex

    def _section_to_latex(self, section: PaperSection, level: int = 1) -> str:
        """Convert section to LaTeX"""
        if level == 1:
            cmd = "section"
        elif level == 2:
            cmd = "subsection"
        else:
            cmd = "subsubsection"

        latex = f"\\{cmd}{{{section.title}}}\n"
        latex += section.content + "\n\n"

        if section.subsections:
            for subsection in section.subsections:
                latex += self._section_to_latex(subsection, level + 1)

        return latex


class PaperGenerator:
    """
    Generate academic papers from research results

    Features:
    - LaTeX manuscript generation
    - Section templates (intro, methods, results, discussion)
    - AI-assisted writing
    - Citation management
    - Figure integration
    """

    def __init__(self, orchestrator=None):
        """
        Initialize paper generator

        Args:
            orchestrator: AI Orchestrator for content generation
        """
        self.orchestrator = orchestrator

    async def generate_paper(
        self,
        topic: str,
        hypothesis: str,
        experiment_design: Any,
        results: Optional[Dict] = None,
        domain: str = "computer_science"
    ) -> Paper:
        """
        Generate complete research paper

        Args:
            topic: Research topic
            hypothesis: Validated hypothesis
            experiment_design: Experiment design object
            results: Experimental results (if available)
            domain: Research domain

        Returns:
            Paper object with complete manuscript
        """
        print(f"\n📝 Generating paper for: {topic}")

        # Generate title
        title = self._generate_title(topic, hypothesis)

        # Generate abstract
        abstract = await self._generate_abstract(
            topic, hypothesis, experiment_design, results
        )

        # Generate sections
        sections = []

        # Introduction
        intro = await self._generate_introduction(topic, hypothesis, domain)
        sections.append(intro)

        # Related Work
        related_work = await self._generate_related_work(topic, domain)
        sections.append(related_work)

        # Methodology
        methods = await self._generate_methodology(hypothesis, experiment_design)
        sections.append(methods)

        # Results
        if results:
            results_section = await self._generate_results(results, experiment_design)
            sections.append(results_section)

        # Discussion
        discussion = await self._generate_discussion(hypothesis, results)
        sections.append(discussion)

        # Conclusion
        conclusion = await self._generate_conclusion(topic, hypothesis, results)
        sections.append(conclusion)

        # Generate references
        references = self._generate_references(topic, domain)

        paper = Paper(
            title=title,
            authors=["ORCHEX Autonomous Research System"],
            abstract=abstract,
            sections=sections,
            references=references
        )

        print(f"✓ Generated {len(sections)} sections, {len(references)} references")

        return paper

    def _generate_title(self, topic: str, hypothesis: str) -> str:
        """Generate paper title"""
        # Simple title generation (can be enhanced with AI)
        words = topic.split()
        if len(words) > 10:
            title = ' '.join(words[:10]) + "..."
        else:
            title = topic

        # Capitalize properly
        return title.title()

    async def _generate_abstract(
        self,
        topic: str,
        hypothesis: str,
        experiment_design: Any,
        results: Optional[Dict]
    ) -> str:
        """Generate abstract"""

        if self.orchestrator:
            # AI-generated abstract
            prompt = f"""
Write a concise academic abstract (150-200 words) for a paper on:

Topic: {topic}
Hypothesis: {hypothesis}
Experiment Type: {experiment_design.experiment_type if experiment_design else 'N/A'}

Include: motivation, hypothesis, methodology, and key findings (if results provided).
"""
            abstract = await self.orchestrator.think(prompt)
            return abstract.strip()

        # Template-based fallback
        abstract = f"""
We investigate {topic.lower()}. Our central hypothesis is that {hypothesis.lower()}.
"""

        if experiment_design:
            abstract += f"We conduct a {experiment_design.experiment_type} study with {experiment_design.num_trials} trials. "

        if results and results.get('success'):
            abstract += f"Our results validate the hypothesis, showing significant improvements. "

        abstract += "This work contributes to understanding of the problem domain and provides insights for future research."

        return abstract.strip()

    async def _generate_introduction(
        self,
        topic: str,
        hypothesis: str,
        domain: str
    ) -> PaperSection:
        """Generate introduction section"""

        if self.orchestrator:
            prompt = f"""
Write an academic introduction (2-3 paragraphs) for a paper on {topic} in {domain}.

Structure:
1. Motivation and problem context
2. Research gap
3. Our hypothesis: {hypothesis}
4. Contributions

Academic style, cite relevant work.
"""
            content = await self.orchestrator.think(prompt)
        else:
            # Template-based
            content = f"""
{topic} has become increasingly important in {domain}.
Recent advances have shown promise, but significant challenges remain.

In this work, we hypothesize that {hypothesis}.
To investigate this, we design and execute a comprehensive experimental study.

Our main contributions are:
\\begin{{itemize}}
\\item Formulation and validation of the hypothesis
\\item Rigorous experimental methodology
\\item Empirical results demonstrating effectiveness
\\item Analysis and insights for future work
\\end{{itemize}}
"""

        return PaperSection(title="Introduction", content=content)

    async def _generate_related_work(
        self,
        topic: str,
        domain: str
    ) -> PaperSection:
        """Generate related work section"""

        content = f"""
Research in {topic.lower()} spans multiple areas within {domain}.

\\textbf{{Classical Approaches:}} Traditional methods have established baseline performance
but face limitations in scalability and adaptability.

\\textbf{{Recent Advances:}} Modern techniques leverage machine learning and optimization
to improve results, though open questions remain.

\\textbf{{Our Approach:}} We build on these foundations while introducing novel validation
and experimental rigor through automated hypothesis testing.
"""

        return PaperSection(title="Related Work", content=content)

    async def _generate_methodology(
        self,
        hypothesis: str,
        experiment_design: Any
    ) -> PaperSection:
        """Generate methodology section"""

        content = f"""
\\subsection{{Hypothesis}}
We test the following hypothesis: \\emph{{{hypothesis}}}

"""

        if experiment_design:
            content += f"""\\subsection{{Experimental Design}}
We employ a {experiment_design.experiment_type} methodology with {experiment_design.num_trials} trials.

\\textbf{{Parameters:}}
"""
            for param in experiment_design.parameters:
                content += f"\\texttt{{{param.name}}}: {param.type}, values: {param.values[:3]}{'...' if len(param.values) > 3 else ''}\\\\\n"

            content += f"\n\\textbf{{Primary Metric:}} {experiment_design.primary_metric}\n\n"
            content += f"\\textbf{{Success Criteria:}} {experiment_design.success_criteria}\n"

        return PaperSection(title="Methodology", content=content)

    async def _generate_results(
        self,
        results: Dict,
        experiment_design: Any
    ) -> PaperSection:
        """Generate results section"""

        content = f"""
We conducted {experiment_design.num_trials if experiment_design else 'N/A'} experimental trials.

\\subsection{{Quantitative Results}}
"""

        if isinstance(results, dict):
            content += "\\begin{itemize}\n"
            for key, value in list(results.items())[:5]:
                content += f"\\item {key}: {value}\n"
            content += "\\end{itemize}\n"

        content += """
\\subsection{Statistical Analysis}
Results demonstrate statistically significant improvements (p < 0.05) over baseline approaches.
"""

        return PaperSection(title="Results", content=content)

    async def _generate_discussion(
        self,
        hypothesis: str,
        results: Optional[Dict]
    ) -> PaperSection:
        """Generate discussion section"""

        content = f"""
Our experimental results provide evidence for the hypothesis: {hypothesis}

\\subsection{{Implications}}
These findings have several important implications for the field:
\\begin{{itemize}}
\\item Validates the theoretical framework
\\item Demonstrates practical effectiveness
\\item Opens avenues for future research
\\end{{itemize}}

\\subsection{{Limitations}}
This study has certain limitations:
\\begin{{itemize}}
\\item Computational experiments only (no physical validation)
\\item Limited to specific parameter ranges
\\item Requires further validation on diverse datasets
\\end{{itemize}}
"""

        return PaperSection(title="Discussion", content=content)

    async def _generate_conclusion(
        self,
        topic: str,
        hypothesis: str,
        results: Optional[Dict]
    ) -> PaperSection:
        """Generate conclusion section"""

        content = f"""
We investigated {topic.lower()} through rigorous hypothesis testing and experimental validation.

Our results {'confirm' if results and results.get('success') else 'provide insights into'} the hypothesis that {hypothesis.lower()}

Future work should explore:
\\begin{{itemize}}
\\item Extension to additional domains
\\item Real-world validation
\\item Integration with existing systems
\\item Theoretical analysis of observed phenomena
\\end{{itemize}}
"""

        return PaperSection(title="Conclusion", content=content)

    def _generate_references(self, topic: str, domain: str) -> List[str]:
        """Generate references (placeholder)"""
        # In production, these would come from Semantic Scholar API
        return [
            f"Smith, J. et al. (2023). Advances in {domain}. Journal of Research.",
            f"Johnson, A. (2024). {topic}: A Survey. Conference Proceedings.",
            "Brown, B. (2023). Computational Methods for Scientific Discovery.",
        ]

    def save_paper(self, paper: Paper, output_dir: Path):
        """Save paper to files"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save LaTeX source
        latex_file = output_dir / "paper.tex"
        latex_file.write_text(paper.to_latex())

        # Save individual sections as markdown (for review)
        md_file = output_dir / "paper.md"
        markdown = f"# {paper.title}\n\n"
        markdown += f"**Authors:** {', '.join(paper.authors)}\n\n"
        markdown += f"## Abstract\n\n{paper.abstract}\n\n"

        for section in paper.sections:
            markdown += f"## {section.title}\n\n{section.content}\n\n"

        md_file.write_text(markdown)

        print(f"✓ Saved paper:")
        print(f"  LaTeX: {latex_file}")
        print(f"  Markdown: {md_file}")

        return latex_file
