#!/usr/bin/env python3
"""
Automated Report Generation for TalAI Turing Challenge

Generates comprehensive reports in multiple formats with
visualizations, summaries, and technical documentation.

Features:
- Generate PDF/HTML reports of validation results
- Include visualizations (charts, graphs)
- Executive summaries for non-technical stakeholders
- Detailed technical appendices
- Export to LaTeX for publication
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import base64
from io import BytesIO
import logging

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd

# Document generation imports
from jinja2 import Environment, FileSystemLoader, Template

# Configure visualization
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Available report formats"""
    HTML = "html"
    PDF = "pdf"
    LATEX = "latex"
    MARKDOWN = "markdown"
    JSON = "json"


class ReportSection(Enum):
    """Standard report sections"""
    EXECUTIVE_SUMMARY = "executive_summary"
    METHODOLOGY = "methodology"
    RESULTS = "results"
    VISUALIZATIONS = "visualizations"
    TECHNICAL_DETAILS = "technical_details"
    APPENDICES = "appendices"
    RECOMMENDATIONS = "recommendations"


@dataclass
class ReportData:
    """Data structure for report generation"""
    title: str
    subtitle: Optional[str] = None
    author: str = "TalAI System"
    date: datetime = field(default_factory=datetime.now)
    validation_results: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    visualizations: List[Dict[str, Any]] = field(default_factory=list)
    executive_summary: Optional[str] = None
    technical_details: Optional[Dict[str, Any]] = None
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualizationConfig:
    """Configuration for visualizations"""
    chart_type: str
    title: str
    data: Any
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    color_scheme: str = "viridis"
    include_legend: bool = True
    save_path: Optional[Path] = None


class ReportGenerator:
    """Main report generation engine"""

    def __init__(self, template_dir: Optional[Path] = None):
        self.template_dir = template_dir or Path(__file__).parent / "templates"
        self.output_dir = Path(__file__).parent / "reports"
        self.output_dir.mkdir(exist_ok=True)

        # Create Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.template_dir) if self.template_dir.exists() else None,
            autoescape=True
        )

        # Chart color schemes
        self.color_schemes = {
            "default": ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"],
            "professional": ["#2c3e50", "#34495e", "#7f8c8d", "#95a5a6", "#bdc3c7"],
            "vibrant": ["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#feca57"]
        }

    async def generate_report(
        self,
        data: ReportData,
        format: ReportFormat = ReportFormat.HTML,
        include_visualizations: bool = True
    ) -> Dict[str, Any]:
        """Generate a comprehensive report"""
        logger.info(f"Generating {format.value} report: {data.title}")

        # Generate visualizations if requested
        if include_visualizations:
            viz_paths = await self._generate_visualizations(data)
            data.visualizations.extend(viz_paths)

        # Generate executive summary if not provided
        if not data.executive_summary:
            data.executive_summary = await self._generate_executive_summary(data)

        # Generate report based on format
        if format == ReportFormat.HTML:
            output = await self._generate_html_report(data)
        elif format == ReportFormat.PDF:
            output = await self._generate_pdf_report(data)
        elif format == ReportFormat.LATEX:
            output = await self._generate_latex_report(data)
        elif format == ReportFormat.MARKDOWN:
            output = await self._generate_markdown_report(data)
        elif format == ReportFormat.JSON:
            output = await self._generate_json_report(data)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{data.title.replace(' ', '_')}_{timestamp}.{format.value}"
        output_path = self.output_dir / filename

        if format == ReportFormat.JSON:
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2, default=str)
        else:
            with open(output_path, 'w') as f:
                f.write(output)

        logger.info(f"Report saved to {output_path}")

        return {
            "status": "success",
            "format": format.value,
            "path": str(output_path),
            "sections": list(ReportSection),
            "visualizations": len(data.visualizations),
            "size": output_path.stat().st_size if output_path.exists() else 0
        }

    async def _generate_visualizations(self, data: ReportData) -> List[Dict[str, Any]]:
        """Generate visualizations for the report"""
        visualizations = []

        # Performance metrics chart
        if data.performance_metrics:
            viz = await self._create_performance_chart(data.performance_metrics)
            visualizations.append(viz)

        # Validation results distribution
        if data.validation_results:
            viz = await self._create_results_distribution(data.validation_results)
            visualizations.append(viz)

        # Timeline chart
        if data.validation_results:
            viz = await self._create_timeline_chart(data.validation_results)
            visualizations.append(viz)

        # Correlation heatmap
        if len(data.validation_results) > 10:
            viz = await self._create_correlation_heatmap(data.validation_results)
            visualizations.append(viz)

        return visualizations

    async def _create_performance_chart(
        self,
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Create performance metrics bar chart"""
        fig, ax = plt.subplots(figsize=(10, 6))

        keys = list(metrics.keys())
        values = list(metrics.values())

        bars = ax.bar(keys, values, color=self.color_schemes["professional"])

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom')

        ax.set_title('Performance Metrics Overview', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0, max(values) * 1.2 if values else 1)
        ax.grid(axis='y', alpha=0.3)

        # Save to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()

        return {
            "type": "performance_chart",
            "title": "Performance Metrics",
            "data": image_base64,
            "format": "base64_png"
        }

    async def _create_results_distribution(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create results distribution pie chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Outcome distribution
        outcomes = [r.get("outcome", "unknown") for r in results]
        outcome_counts = pd.Series(outcomes).value_counts()

        colors = self.color_schemes["vibrant"][:len(outcome_counts)]
        wedges, texts, autotexts = ax1.pie(
            outcome_counts.values,
            labels=outcome_counts.index,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90
        )

        ax1.set_title('Validation Outcomes Distribution', fontsize=12, fontweight='bold')

        # Confidence distribution
        confidences = [r.get("confidence", 0) for r in results if "confidence" in r]
        if confidences:
            ax2.hist(confidences, bins=20, color=self.color_schemes["professional"][0], edgecolor='black', alpha=0.7)
            ax2.set_xlabel('Confidence Score', fontsize=10)
            ax2.set_ylabel('Frequency', fontsize=10)
            ax2.set_title('Confidence Score Distribution', fontsize=12, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No confidence data available',
                    ha='center', va='center', transform=ax2.transAxes)

        plt.tight_layout()

        # Save to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()

        return {
            "type": "distribution_chart",
            "title": "Results Distribution",
            "data": image_base64,
            "format": "base64_png"
        }

    async def _create_timeline_chart(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create timeline chart of validations"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Extract timestamps and scores
        timestamps = []
        scores = []
        for r in results:
            if "timestamp" in r and "score" in r:
                timestamps.append(pd.to_datetime(r["timestamp"]))
                scores.append(r["score"])

        if timestamps and scores:
            ax.plot(timestamps, scores, marker='o', linestyle='-',
                   color=self.color_schemes["professional"][0], markersize=6)

            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel('Validation Score', fontsize=10)
            ax.set_title('Validation Performance Over Time', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Rotate x-axis labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # Add trend line
            if len(timestamps) > 3:
                z = np.polyfit(range(len(timestamps)), scores, 1)
                p = np.poly1d(z)
                ax.plot(timestamps, p(range(len(timestamps))),
                       "r--", alpha=0.5, label='Trend')
                ax.legend()
        else:
            ax.text(0.5, 0.5, 'Insufficient timeline data',
                   ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()

        # Save to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()

        return {
            "type": "timeline_chart",
            "title": "Performance Timeline",
            "data": image_base64,
            "format": "base64_png"
        }

    async def _create_correlation_heatmap(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create correlation heatmap of metrics"""
        # Extract numeric features
        numeric_data = []
        for r in results:
            row = {}
            for key, value in r.items():
                if isinstance(value, (int, float)):
                    row[key] = value
            if row:
                numeric_data.append(row)

        if len(numeric_data) > 2:
            df = pd.DataFrame(numeric_data).fillna(0)

            # Calculate correlation matrix
            corr_matrix = df.corr()

            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                       center=0, square=True, linewidths=1, ax=ax,
                       cbar_kws={"shrink": 0.8})

            ax.set_title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')
            plt.tight_layout()

            # Save to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()

            return {
                "type": "correlation_heatmap",
                "title": "Feature Correlations",
                "data": image_base64,
                "format": "base64_png"
            }

        return {
            "type": "correlation_heatmap",
            "title": "Feature Correlations",
            "data": None,
            "error": "Insufficient data for correlation analysis"
        }

    async def _generate_executive_summary(self, data: ReportData) -> str:
        """Generate executive summary from data"""
        summary_parts = []

        # Overview
        summary_parts.append(f"Report: {data.title}")
        if data.subtitle:
            summary_parts.append(f"Focus: {data.subtitle}")

        # Key metrics
        if data.performance_metrics:
            avg_score = np.mean(list(data.performance_metrics.values()))
            summary_parts.append(f"Average Performance Score: {avg_score:.2f}")

        # Results summary
        if data.validation_results:
            total = len(data.validation_results)
            successful = sum(1 for r in data.validation_results if r.get("outcome") == "success")
            success_rate = (successful / total * 100) if total > 0 else 0
            summary_parts.append(f"Validation Success Rate: {success_rate:.1f}% ({successful}/{total})")

        # Recommendations
        if data.recommendations:
            summary_parts.append(f"Key Recommendations: {len(data.recommendations)} identified")

        return " | ".join(summary_parts)

    async def _generate_html_report(self, data: ReportData) -> str:
        """Generate HTML report"""
        # HTML template
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        .summary { background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .metric { background: #fff; border: 1px solid #ddd; padding: 15px; border-radius: 5px; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; color: #3498db; }
        .visualization { margin: 30px 0; text-align: center; }
        .visualization img { max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 10px; text-align: left; border: 1px solid #ddd; }
        th { background: #3498db; color: white; }
        .recommendations { background: #d4edda; border: 1px solid #c3e6cb; padding: 15px; border-radius: 5px; }
        .footer { margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #7f8c8d; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    {% if subtitle %}<h2 style="margin-top: 0; color: #7f8c8d;">{{ subtitle }}</h2>{% endif %}

    <div class="summary">
        <h2>Executive Summary</h2>
        <p>{{ executive_summary }}</p>
    </div>

    {% if performance_metrics %}
    <h2>Performance Metrics</h2>
    <div class="metrics">
        {% for key, value in performance_metrics.items() %}
        <div class="metric">
            <div>{{ key }}</div>
            <div class="metric-value">{{ "%.2f"|format(value) }}</div>
        </div>
        {% endfor %}
    </div>
    {% endif %}

    {% if visualizations %}
    <h2>Visualizations</h2>
    {% for viz in visualizations %}
    <div class="visualization">
        <h3>{{ viz.title }}</h3>
        {% if viz.data %}
        <img src="data:image/png;base64,{{ viz.data }}" alt="{{ viz.title }}">
        {% endif %}
    </div>
    {% endfor %}
    {% endif %}

    {% if recommendations %}
    <div class="recommendations">
        <h2>Recommendations</h2>
        <ul>
        {% for rec in recommendations %}
            <li>{{ rec }}</li>
        {% endfor %}
        </ul>
    </div>
    {% endif %}

    <div class="footer">
        <p>Generated by {{ author }} on {{ date.strftime('%Y-%m-%d %H:%M:%S') }}</p>
    </div>
</body>
</html>
        """

        template = Template(html_template)
        return template.render(**data.__dict__)

    async def _generate_latex_report(self, data: ReportData) -> str:
        """Generate LaTeX report for academic publication"""
        latex_template = r"""
\documentclass[12pt]{article}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{booktabs}

\title{{{ title }}}
{% if subtitle %}\subtitle{{{ subtitle }}}{% endif %}
\author{{{ author }}}
\date{{{ date.strftime('%B %d, %Y') }}}

\begin{document}

\maketitle

\begin{abstract}
{{ executive_summary }}
\end{abstract}

\section{Introduction}
This report presents the comprehensive analysis and validation results from the TalAI Turing Challenge System.

{% if performance_metrics %}
\section{Performance Metrics}
\begin{table}[h]
\centering
\begin{tabular}{lr}
\toprule
Metric & Value \\
\midrule
{% for key, value in performance_metrics.items() %}
{{ key }} & {{ "%.3f"|format(value) }} \\
{% endfor %}
\bottomrule
\end{tabular}
\caption{System Performance Metrics}
\end{table}
{% endif %}

{% if recommendations %}
\section{Recommendations}
\begin{enumerate}
{% for rec in recommendations %}
\item {{ rec }}
{% endfor %}
\end{enumerate}
{% endif %}

\section{Conclusion}
The analysis demonstrates the effectiveness of the adaptive validation system.

\end{document}
        """

        template = Template(latex_template)
        return template.render(**data.__dict__)

    async def _generate_markdown_report(self, data: ReportData) -> str:
        """Generate Markdown report"""
        md_lines = []

        # Header
        md_lines.append(f"# {data.title}")
        if data.subtitle:
            md_lines.append(f"## {data.subtitle}")
        md_lines.append(f"\n*Generated by {data.author} on {data.date.strftime('%Y-%m-%d')}*\n")

        # Executive Summary
        md_lines.append("## Executive Summary")
        md_lines.append(data.executive_summary or "No summary available")
        md_lines.append("")

        # Performance Metrics
        if data.performance_metrics:
            md_lines.append("## Performance Metrics")
            md_lines.append("| Metric | Value |")
            md_lines.append("|--------|-------|")
            for key, value in data.performance_metrics.items():
                md_lines.append(f"| {key} | {value:.3f} |")
            md_lines.append("")

        # Recommendations
        if data.recommendations:
            md_lines.append("## Recommendations")
            for i, rec in enumerate(data.recommendations, 1):
                md_lines.append(f"{i}. {rec}")
            md_lines.append("")

        return "\n".join(md_lines)

    async def _generate_json_report(self, data: ReportData) -> Dict[str, Any]:
        """Generate JSON report"""
        return {
            "title": data.title,
            "subtitle": data.subtitle,
            "author": data.author,
            "date": data.date.isoformat(),
            "executive_summary": data.executive_summary,
            "performance_metrics": data.performance_metrics,
            "validation_results": data.validation_results,
            "recommendations": data.recommendations,
            "metadata": data.metadata,
            "visualizations": [
                {k: v for k, v in viz.items() if k != "data"}
                for viz in data.visualizations
            ]
        }

    async def _generate_pdf_report(self, data: ReportData) -> str:
        """Generate PDF report (returns LaTeX to be compiled)"""
        # For actual PDF generation, you would compile the LaTeX
        # This returns LaTeX that can be compiled with pdflatex
        return await self._generate_latex_report(data)