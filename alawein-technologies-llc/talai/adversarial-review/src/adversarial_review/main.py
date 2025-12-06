#!/usr/bin/env python3
"""
AdversarialReview - Brutal AI Research Paper Critic

Provides adversarial feedback on research papers across 6 dimensions:
- Statistical validity
- Methodological rigor
- Logical consistency
- Historical context
- Ethical implications
- Economic feasibility

Usage:
    python review.py --title "Paper Title" --abstract "Abstract text" --mode nightmare
    python review.py --input paper.txt --mode brutal
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any


@dataclass
class CriticFeedback:
    """Feedback from a single adversarial critic"""
    critic_name: str
    dimension: str
    severity: str  # CRITICAL | MAJOR | MINOR
    issues: List[str]
    score: float  # 0-10, lower is worse
    recommendations: List[str]


@dataclass
class ReviewResult:
    """Complete adversarial review result"""
    paper_title: str
    overall_score: float
    verdict: str  # REJECT | MAJOR_REVISION | MINOR_REVISION | ACCEPT
    feedbacks: List[CriticFeedback]
    executive_summary: str
    timestamp: str


class AdversarialCritic:
    """Base class for adversarial critics"""

    def __init__(self, name: str, dimension: str):
        """Initialize adversarial critic

        Args:
            name: Name of the critic (e.g., "Statistical Skeptic")
            dimension: Review dimension (e.g., "statistical_validity")
        """
        self.name = name
        self.dimension = dimension

    def review(self, paper_text: str, mode: str) -> CriticFeedback:
        """Generate adversarial feedback"""
        raise NotImplementedError


class StatisticalCritic(AdversarialCritic):
    """Attacks statistical validity"""

    def __init__(self):
        """Initialize Statistical Skeptic critic"""
        super().__init__("Statistical Skeptic", "statistical_validity")

    def review(self, paper_text: str, mode: str) -> CriticFeedback:
        """Review paper for statistical validity issues

        Args:
            paper_text: Full text of the paper to review
            mode: Review mode ('normal' or 'nightmare')

        Returns:
            CriticFeedback with statistical issues and recommendations
        """
        issues = []
        recommendations = []

        # Simulate statistical analysis
        potential_issues = [
            "Sample size appears insufficient for claimed effect size (power analysis needed)",
            "Multiple comparison corrections not mentioned despite testing multiple hypotheses",
            "P-hacking risk: selective reporting of significant results suspected",
            "Assumptions of statistical test (normality, homoscedasticity) not verified",
            "Confidence intervals not reported, only p-values (selective reporting)",
            "Effect size not reported or discussed (statistical vs practical significance)",
            "Missing information about data distribution and outliers",
            "Independence assumption may be violated (temporal/spatial autocorrelation)",
        ]

        potential_recommendations = [
            "Conduct formal power analysis and report required sample size",
            "Apply Bonferroni or FDR correction for multiple comparisons",
            "Pre-register analysis plan to avoid p-hacking accusations",
            "Verify and report assumption checks (QQ plots, Levene's test)",
            "Report effect sizes (Cohen's d, eta-squared) alongside p-values",
            "Include confidence intervals for all estimates",
            "Perform sensitivity analysis excluding outliers",
            "Use mixed-effects models if independence assumption violated",
        ]

        # In nightmare/brutal mode, find more issues
        n_issues = 5 if mode == "nightmare" else 3
        issues = random.sample(potential_issues, min(n_issues, len(potential_issues)))
        recommendations = random.sample(potential_recommendations, min(n_issues, len(potential_recommendations)))

        # Score decreases with more issues
        score = max(1.0, 8.0 - len(issues) * 1.2)
        severity = "CRITICAL" if score < 4 else "MAJOR" if score < 6 else "MINOR"

        return CriticFeedback(
            critic_name=self.name,
            dimension=self.dimension,
            severity=severity,
            issues=issues,
            score=round(score, 1),
            recommendations=recommendations
        )


class MethodologicalCritic(AdversarialCritic):
    """Attacks methodological rigor"""

    def __init__(self):
        """Initialize Methodology Maverick critic"""
        super().__init__("Methodology Maverick", "methodological_rigor")

    def review(self, paper_text: str, mode: str) -> CriticFeedback:
        """Review paper for methodological issues

        Args:
            paper_text: Full text of the paper to review
            mode: Review mode ('normal' or 'nightmare')

        Returns:
            CriticFeedback with methodological issues and recommendations
        """
        potential_issues = [
            "Control group inadequately matched to treatment group",
            "Randomization procedure not described or potentially biased",
            "Blinding not implemented (researcher and/or participant bias risk)",
            "Confounding variables not controlled or measured",
            "Measurement instruments not validated or reliability not reported",
            "Temporal validity concerns: data collection timing may introduce bias",
            "Ecological validity: lab setting may not generalize to real world",
            "Selection bias in participant recruitment not addressed",
            "Attrition/dropout analysis missing (missing data mechanism not random)",
        ]

        potential_recommendations = [
            "Use propensity score matching or regression adjustment for confounders",
            "Describe randomization method (block randomization, stratification)",
            "Implement double-blind or triple-blind procedure where possible",
            "Measure and control for identified confounding variables",
            "Report reliability coefficients (Cronbach's alpha, test-retest)",
            "Collect data at multiple time points to establish temporal precedence",
            "Conduct field study to complement laboratory findings",
            "Use probability sampling or discuss selection bias limitations",
            "Perform sensitivity analysis for missing data (MCAR, MAR, MNAR)",
        ]

        n_issues = 6 if mode == "nightmare" else 4
        issues = random.sample(potential_issues, min(n_issues, len(potential_issues)))
        recommendations = random.sample(potential_recommendations, min(n_issues, len(potential_recommendations)))

        score = max(1.5, 7.5 - len(issues) * 1.0)
        severity = "CRITICAL" if score < 4 else "MAJOR" if score < 6 else "MINOR"

        return CriticFeedback(
            critic_name=self.name,
            dimension=self.dimension,
            severity=severity,
            issues=issues,
            score=round(score, 1),
            recommendations=recommendations
        )


class LogicalCritic(AdversarialCritic):
    """Attacks logical consistency"""

    def __init__(self):
        """Initialize Logic Enforcer critic"""
        super().__init__("Logic Enforcer", "logical_consistency")

    def review(self, paper_text: str, mode: str) -> CriticFeedback:
        """Review paper for logical consistency issues

        Args:
            paper_text: Full text of the paper to review
            mode: Review mode ('normal' or 'nightmare')

        Returns:
            CriticFeedback with logical consistency issues and recommendations
        """
        potential_issues = [
            "Circular reasoning: conclusion assumes what it claims to prove",
            "Non-sequitur: conclusions don't follow from stated premises",
            "False dichotomy: presents limited options when more exist",
            "Correlation-causation confusion: causal claims from correlational data",
            "Overgeneralization: conclusions broader than data support",
            "Inconsistent definitions across paper sections",
            "Self-contradictory statements in introduction vs discussion",
            "Unfalsifiable hypothesis (cannot be disproven by any evidence)",
        ]

        potential_recommendations = [
            "Restructure argument to avoid circular dependencies",
            "Add intermediate logical steps to connect premises to conclusions",
            "Acknowledge alternative explanations or interpretations",
            "Limit causal language unless experimental manipulation performed",
            "Narrow scope of conclusions to match sample characteristics",
            "Use consistent terminology and definitions throughout",
            "Resolve contradictions between sections",
            "Reformulate hypothesis to be empirically testable",
        ]

        n_issues = 4 if mode == "nightmare" else 2
        issues = random.sample(potential_issues, min(n_issues, len(potential_issues)))
        recommendations = random.sample(potential_recommendations, min(n_issues, len(potential_recommendations)))

        score = max(2.0, 8.5 - len(issues) * 1.5)
        severity = "CRITICAL" if score < 4 else "MAJOR" if score < 6 else "MINOR"

        return CriticFeedback(
            critic_name=self.name,
            dimension=self.dimension,
            severity=severity,
            issues=issues,
            score=round(score, 1),
            recommendations=recommendations
        )


class HistoricalCritic(AdversarialCritic):
    """Attacks historical context and novelty"""

    def __init__(self):
        super().__init__("History Hunter", "historical_context")

    def review(self, paper_text: str, mode: str) -> CriticFeedback:
        potential_issues = [
            "Literature review incomplete: missing seminal papers from 1990s-2000s",
            "Cherry-picking citations: only cites supporting evidence",
            "Ignores contradictory findings from previous studies",
            "Claims novelty for previously established phenomenon (reinventing wheel)",
            "Misrepresents or oversimplifies prior work",
            "Fails to cite foundational theoretical frameworks",
            "Neglects relevant work from adjacent disciplines",
            "Recency bias: only cites papers from last 5 years",
        ]

        potential_recommendations = [
            "Conduct systematic literature review back to origin of field",
            "Include citations representing diverse perspectives and findings",
            "Explicitly address contradictory evidence and explain discrepancies",
            "Clearly articulate novel contribution beyond existing work",
            "Accurately represent cited work (verify quotes and summaries)",
            "Ground work in established theoretical traditions",
            "Search databases from related fields (cross-disciplinary review)",
            "Balance recent and classic citations (median age ~10 years)",
        ]

        n_issues = 4 if mode == "nightmare" else 3
        issues = random.sample(potential_issues, min(n_issues, len(potential_issues)))
        recommendations = random.sample(potential_recommendations, min(n_issues, len(potential_recommendations)))

        score = max(2.5, 7.0 - len(issues) * 1.0)
        severity = "CRITICAL" if score < 4 else "MAJOR" if score < 6 else "MINOR"

        return CriticFeedback(
            critic_name=self.name,
            dimension=self.dimension,
            severity=severity,
            issues=issues,
            score=round(score, 1),
            recommendations=recommendations
        )


class EthicalCritic(AdversarialCritic):
    """Attacks ethical implications"""

    def __init__(self):
        super().__init__("Ethics Enforcer", "ethical_implications")

    def review(self, paper_text: str, mode: str) -> CriticFeedback:
        """Review paper for ethical issues

        Args:
            paper_text: Full text of the paper to review
            mode: Review mode ('normal' or 'nightmare')

        Returns:
            CriticFeedback with ethical issues and recommendations
        """
        potential_issues = [
            "Informed consent procedures inadequately described",
            "Vulnerable populations included without additional protections",
            "Privacy risks from data sharing or de-anonymization",
            "Dual-use concerns: findings could enable harmful applications",
            "Environmental impact of methods not considered",
            "Equity concerns: benefits/risks distributed unequally",
            "IRB approval not mentioned or insufficient for study scope",
            "Conflicts of interest not disclosed (funding sources)",
        ]

        potential_recommendations = [
            "Provide detailed informed consent protocol in appendix",
            "Justify inclusion of vulnerable populations and describe protections",
            "Perform privacy risk assessment and apply differential privacy",
            "Discuss potential misuse and propose mitigation strategies",
            "Calculate and report carbon footprint of computational methods",
            "Analyze distributional effects across demographic groups",
            "Obtain IRB approval or explain exemption rationale",
            "Disclose all funding sources and potential conflicts",
        ]

        n_issues = 3 if mode == "nightmare" else 2
        issues = random.sample(potential_issues, min(n_issues, len(potential_issues)))
        recommendations = random.sample(potential_recommendations, min(n_issues, len(potential_recommendations)))

        score = max(3.0, 8.0 - len(issues) * 1.5)
        severity = "CRITICAL" if score < 4 else "MAJOR" if score < 6 else "MINOR"

        return CriticFeedback(
            critic_name=self.name,
            dimension=self.dimension,
            severity=severity,
            issues=issues,
            score=round(score, 1),
            recommendations=recommendations
        )


class EconomicCritic(AdversarialCritic):
    """Attacks economic feasibility and impact"""

    def __init__(self):
        super().__init__("Economic Realist", "economic_feasibility")

    def review(self, paper_text: str, mode: str) -> CriticFeedback:
        potential_issues = [
            "Cost-benefit analysis missing or overly optimistic",
            "Scalability concerns: method too expensive for widespread adoption",
            "Resource requirements (compute, materials) not sustainable",
            "Ignores opportunity cost of proposed approach",
            "Market viability questionable (who would pay for this?)",
            "Assumes unlimited resources or idealized conditions",
            "Neglects implementation and maintenance costs",
            "Economic assumptions not validated with real-world data",
        ]

        potential_recommendations = [
            "Conduct formal cost-effectiveness analysis with realistic assumptions",
            "Analyze scaling curve: how cost changes with problem size",
            "Calculate resource efficiency metrics (cost per unit improvement)",
            "Compare to alternative approaches on economic dimensions",
            "Survey potential users/buyers for willingness to pay",
            "Model resource-constrained scenario with practical limits",
            "Estimate total cost of ownership over lifecycle",
            "Ground economic assumptions in published data or industry reports",
        ]

        n_issues = 3 if mode == "nightmare" else 2
        issues = random.sample(potential_issues, min(n_issues, len(potential_issues)))
        recommendations = random.sample(potential_recommendations, min(n_issues, len(potential_recommendations)))

        score = max(3.5, 7.5 - len(issues) * 1.0)
        severity = "CRITICAL" if score < 4 else "MAJOR" if score < 6 else "MINOR"

        return CriticFeedback(
            critic_name=self.name,
            dimension=self.dimension,
            severity=severity,
            issues=issues,
            score=round(score, 1),
            recommendations=recommendations
        )


class AdversarialReview:
    """Main adversarial review system"""

    def __init__(self):
        self.critics = [
            StatisticalCritic(),
            MethodologicalCritic(),
            LogicalCritic(),
            HistoricalCritic(),
            EthicalCritic(),
            EconomicCritic(),
        ]

    def review_paper(self, title: str, paper_text: str, mode: str = "standard") -> ReviewResult:
        """
        Conduct adversarial review

        Args:
            title: Paper title
            paper_text: Full paper text or abstract
            mode: 'nightmare' (harshest) | 'brutal' | 'standard'

        Returns:
            ReviewResult with all critic feedback
        """
        print(f"\n{'='*70}")
        print(f"ADVERSARIAL REVIEW - Mode: {mode.upper()}")
        print(f"{'='*70}\n")
        print(f"Paper: {title}\n")

        feedbacks = []

        for critic in self.critics:
            print(f"[{critic.name}] Analyzing {critic.dimension}...")
            feedback = critic.review(paper_text, mode)
            feedbacks.append(feedback)

        # Calculate overall score
        overall_score = sum(f.score for f in feedbacks) / len(feedbacks)

        # Determine verdict
        if overall_score < 4.0:
            verdict = "REJECT"
        elif overall_score < 5.5:
            verdict = "MAJOR_REVISION"
        elif overall_score < 7.0:
            verdict = "MINOR_REVISION"
        else:
            verdict = "ACCEPT"

        # Generate executive summary
        critical_issues = [f for f in feedbacks if f.severity == "CRITICAL"]
        major_issues = [f for f in feedbacks if f.severity == "MAJOR"]

        summary_parts = []
        summary_parts.append(f"Overall Score: {overall_score:.1f}/10")
        summary_parts.append(f"Verdict: {verdict}")

        if critical_issues:
            summary_parts.append(f"\nCritical issues found in: {', '.join(f.dimension for f in critical_issues)}")
        if major_issues:
            summary_parts.append(f"Major issues found in: {', '.join(f.dimension for f in major_issues)}")

        executive_summary = "\n".join(summary_parts)

        result = ReviewResult(
            paper_title=title,
            overall_score=round(overall_score, 1),
            verdict=verdict,
            feedbacks=feedbacks,
            executive_summary=executive_summary,
            timestamp=datetime.now().isoformat()
        )

        return result

    def print_result(self, result: ReviewResult):
        """Print formatted review result"""
        print(f"\n{'='*70}")
        print(f"ADVERSARIAL REVIEW RESULT")
        print(f"{'='*70}\n")

        print(f"Paper: {result.paper_title}")
        print(f"Overall Score: {result.overall_score}/10")
        print(f"Verdict: {result.verdict}\n")

        for feedback in result.feedbacks:
            print(f"\n{'-'*70}")
            print(f"[{feedback.critic_name}] - {feedback.dimension.upper()}")
            print(f"Severity: {feedback.severity} | Score: {feedback.score}/10")
            print(f"{'-'*70}")

            print("\nISSUES IDENTIFIED:")
            for i, issue in enumerate(feedback.issues, 1):
                print(f"  {i}. {issue}")

            print("\nRECOMMENDATIONS:")
            for i, rec in enumerate(feedback.recommendations, 1):
                print(f"  {i}. {rec}")

        print(f"\n{'='*70}")
        print(f"EXECUTIVE SUMMARY")
        print(f"{'='*70}")
        print(result.executive_summary)
        print(f"{'='*70}\n")

    def save_result(self, result: ReviewResult, output_file: str):
        """Save result to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        print(f"\nReview saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="AdversarialReview - Brutal AI Research Paper Critic"
    )
    parser.add_argument(
        "--title",
        type=str,
        required=True,
        help="Paper title"
    )
    parser.add_argument(
        "--abstract",
        type=str,
        help="Paper abstract or summary"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to text file containing paper"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "brutal", "nightmare"],
        default="standard",
        help="Review harshness level"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path"
    )

    args = parser.parse_args()

    # Get paper text
    if args.input:
        with open(args.input, 'r') as f:
            paper_text = f.read()
    elif args.abstract:
        paper_text = args.abstract
    else:
        print("Error: Must provide either --abstract or --input")
        sys.exit(1)

    # Conduct review
    reviewer = AdversarialReview()
    result = reviewer.review_paper(args.title, paper_text, args.mode)

    # Print result
    reviewer.print_result(result)

    # Save if requested
    if args.output:
        reviewer.save_result(result, args.output)


if __name__ == "__main__":
    main()
