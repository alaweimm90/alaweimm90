#!/usr/bin/env python3
"""
hallucination_verifier.py - Three-Layer Verification Cascade for AI Outputs

Implements hallucination prevention through multi-layer verification:
- Layer 1: Semantic Grounding - output must reference provided context
- Layer 2: Entity Verification - mentioned files/functions must exist
- Layer 3: Claim Verification - factual claims must be provable

Usage:
    python hallucination_verifier.py verify output.json --context context.json
    python hallucination_verifier.py check-entities --files "src/*.py" --output output.txt
    python hallucination_verifier.py analyze --input response.txt --workspace .
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from enum import Enum

import click
import yaml


class VerificationLayer(Enum):
    """Verification layer types."""
    SEMANTIC_GROUNDING = "semantic_grounding"
    ENTITY_VERIFICATION = "entity_verification"
    CLAIM_VERIFICATION = "claim_verification"


@dataclass
class VerificationResult:
    """Result of a single verification check."""
    layer: VerificationLayer
    passed: bool
    score: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    flagged_items: List[str] = field(default_factory=list)


@dataclass
class HallucinationReport:
    """Complete hallucination verification report."""
    timestamp: str
    overall_passed: bool
    confidence_score: float
    results: List[VerificationResult] = field(default_factory=list)
    flagged_claims: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class HallucinationVerifier:
    """Multi-layer hallucination verification system."""

    # Common hallucination patterns
    HALLUCINATION_INDICATORS = [
        r"(?i)\b(obviously|clearly|everyone knows|as you know)\b",
        r"(?i)\b(will always|never fails|guaranteed to)\b",
        r"(?i)\b(the only way|the best way|the right way)\b",
        r"(?i)\b(simply|just|easily)\b.{0,20}\b(do|add|implement|fix)\b",
    ]

    # File/function reference patterns
    FILE_PATTERNS = [
        r'`([a-zA-Z0-9_\-./]+\.[a-zA-Z]+)`',  # backtick quoted
        r'"([a-zA-Z0-9_\-./]+\.[a-zA-Z]+)"',  # double quoted
        r"'([a-zA-Z0-9_\-./]+\.[a-zA-Z]+)'",  # single quoted
        r'\b([a-zA-Z0-9_\-]+\.(py|js|ts|yaml|yml|json|md|txt))\b',  # bare extension
    ]

    FUNCTION_PATTERNS = [
        r'`([a-zA-Z_][a-zA-Z0-9_]*)\(`',  # function call in backticks
        r'\b(def|function|async function)\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # definitions
        r'(?:method|function|class)\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?',  # references
    ]

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path.cwd()
        self.policy = self._load_policy()

        # Get configuration from policy
        hall_config = self.policy.get("hallucination_prevention", {})
        self.confidence_threshold = hall_config.get("confidence_threshold", 0.8)

        layers = hall_config.get("verification_layers", [])
        self.layer_weights = {}
        for layer in layers:
            name = layer.get("name", "")
            weight = layer.get("weight", 0.33)
            self.layer_weights[name] = weight

    def _load_policy(self) -> Dict[str, Any]:
        """Load orchestration governance policy."""
        policy_path = self.base_path / ".metaHub/policies/orchestration-governance.yaml"
        if policy_path.exists():
            with open(policy_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}

    def verify(
        self,
        output_text: str,
        context: Optional[Dict[str, Any]] = None,
        workspace_path: Optional[Path] = None
    ) -> HallucinationReport:
        """Run full verification cascade on output text."""
        report = HallucinationReport(
            timestamp=datetime.now().isoformat(),
            overall_passed=True,
            confidence_score=1.0
        )

        workspace = workspace_path or self.base_path

        # Layer 1: Semantic Grounding
        semantic_result = self._verify_semantic_grounding(output_text, context)
        report.results.append(semantic_result)

        # Layer 2: Entity Verification
        entity_result = self._verify_entities(output_text, workspace)
        report.results.append(entity_result)

        # Layer 3: Claim Verification
        claim_result = self._verify_claims(output_text, context)
        report.results.append(claim_result)

        # Calculate overall confidence score
        total_weight = 0
        weighted_score = 0

        for result in report.results:
            layer_name = result.layer.value
            weight = self.layer_weights.get(layer_name, 0.33)
            weighted_score += result.score * weight
            total_weight += weight

            # Collect flagged items
            report.flagged_claims.extend(result.flagged_items)

        if total_weight > 0:
            report.confidence_score = weighted_score / total_weight
        else:
            report.confidence_score = min(r.score for r in report.results)

        report.overall_passed = report.confidence_score >= self.confidence_threshold

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        return report

    def _verify_semantic_grounding(
        self,
        output_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> VerificationResult:
        """Layer 1: Verify output is semantically grounded in context."""
        if not context:
            return VerificationResult(
                layer=VerificationLayer.SEMANTIC_GROUNDING,
                passed=True,
                score=0.5,
                message="No context provided - cannot fully verify semantic grounding",
                details={"context_provided": False}
            )

        # Extract key terms from context
        context_text = json.dumps(context).lower()
        context_terms = self._extract_significant_terms(context_text)

        # Extract key terms from output
        output_lower = output_text.lower()
        output_terms = self._extract_significant_terms(output_lower)

        # Calculate overlap
        if not context_terms:
            return VerificationResult(
                layer=VerificationLayer.SEMANTIC_GROUNDING,
                passed=True,
                score=0.7,
                message="Empty context - semantic grounding check skipped",
                details={"context_terms": 0}
            )

        overlap = context_terms & output_terms
        grounding_ratio = len(overlap) / max(len(output_terms), 1)

        # Check for hallucination indicators
        indicator_matches = []
        for pattern in self.HALLUCINATION_INDICATORS:
            matches = re.findall(pattern, output_text)
            indicator_matches.extend(matches)

        # Penalize for hallucination indicators
        indicator_penalty = min(len(indicator_matches) * 0.1, 0.3)

        score = max(0, min(1.0, grounding_ratio - indicator_penalty))
        passed = score >= 0.6

        return VerificationResult(
            layer=VerificationLayer.SEMANTIC_GROUNDING,
            passed=passed,
            score=score,
            message=f"Semantic grounding: {len(overlap)}/{len(output_terms)} terms grounded",
            details={
                "context_terms": len(context_terms),
                "output_terms": len(output_terms),
                "grounded_terms": len(overlap),
                "grounding_ratio": grounding_ratio,
                "indicator_matches": len(indicator_matches)
            },
            flagged_items=indicator_matches[:5]
        )

    def _extract_significant_terms(self, text: str) -> Set[str]:
        """Extract significant terms from text."""
        # Remove common stop words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
            'by', 'from', 'as', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'under', 'again',
            'this', 'that', 'these', 'those', 'it', 'its', 'they',
            'their', 'them', 'and', 'or', 'but', 'if', 'then', 'else',
            'when', 'where', 'why', 'how', 'all', 'each', 'every',
            'both', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 'just', 'also', 'now', 'here', 'there'
        }

        # Extract words (3+ chars, alphanumeric with underscores)
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b', text.lower())

        # Filter stop words and return unique terms
        return {w for w in words if w not in stop_words}

    def _verify_entities(
        self,
        output_text: str,
        workspace: Path
    ) -> VerificationResult:
        """Layer 2: Verify mentioned files and functions exist."""
        # Extract file references
        mentioned_files = set()
        for pattern in self.FILE_PATTERNS:
            matches = re.findall(pattern, output_text)
            for match in matches:
                if isinstance(match, tuple):
                    mentioned_files.add(match[0])
                else:
                    mentioned_files.add(match)

        # Extract function references
        mentioned_functions = set()
        for pattern in self.FUNCTION_PATTERNS:
            matches = re.findall(pattern, output_text)
            for match in matches:
                if isinstance(match, tuple):
                    mentioned_functions.update(m for m in match if m)
                else:
                    mentioned_functions.add(match)

        # Verify files exist
        verified_files = []
        missing_files = []

        for file_ref in mentioned_files:
            # Try multiple path resolutions
            found = False
            for base in [workspace, workspace / "src", workspace / "lib"]:
                candidate = base / file_ref
                if candidate.exists():
                    verified_files.append(str(file_ref))
                    found = True
                    break

            # Also check if file exists with glob pattern
            if not found:
                matches = list(workspace.rglob(file_ref))
                if matches:
                    verified_files.append(str(file_ref))
                    found = True

            if not found:
                missing_files.append(str(file_ref))

        # Verify functions exist (simplified - just check if they appear in codebase)
        verified_functions = []
        missing_functions = []

        if mentioned_functions:
            # Build a set of known functions from codebase
            known_functions = self._scan_codebase_functions(workspace)

            for func in mentioned_functions:
                if func in known_functions or len(func) < 3:
                    verified_functions.append(func)
                else:
                    missing_functions.append(func)

        # Calculate score
        total_entities = len(mentioned_files) + len(mentioned_functions)
        verified_entities = len(verified_files) + len(verified_functions)

        if total_entities == 0:
            score = 1.0
            message = "No file/function references to verify"
        else:
            score = verified_entities / total_entities
            message = f"Entity verification: {verified_entities}/{total_entities} entities found"

        return VerificationResult(
            layer=VerificationLayer.ENTITY_VERIFICATION,
            passed=score >= 0.7,
            score=score,
            message=message,
            details={
                "mentioned_files": list(mentioned_files),
                "verified_files": verified_files,
                "missing_files": missing_files,
                "mentioned_functions": list(mentioned_functions),
                "verified_functions": verified_functions,
                "missing_functions": missing_functions
            },
            flagged_items=missing_files + missing_functions[:3]
        )

    def _scan_codebase_functions(self, workspace: Path) -> Set[str]:
        """Scan codebase for function/method definitions."""
        functions = set()

        # Common built-in functions/methods to whitelist
        builtins = {
            'print', 'len', 'str', 'int', 'float', 'list', 'dict', 'set',
            'open', 'read', 'write', 'close', 'append', 'extend', 'insert',
            'pop', 'remove', 'clear', 'copy', 'update', 'get', 'keys',
            'values', 'items', 'join', 'split', 'strip', 'replace',
            'format', 'lower', 'upper', 'find', 'index', 'count',
            'map', 'filter', 'reduce', 'sorted', 'reversed', 'enumerate',
            'zip', 'range', 'sum', 'min', 'max', 'abs', 'round',
            'isinstance', 'issubclass', 'hasattr', 'getattr', 'setattr',
            'main', 'init', 'new', 'call', 'str', 'repr', 'iter', 'next'
        }
        functions.update(builtins)

        # Scan Python files
        try:
            for py_file in workspace.rglob("*.py"):
                if ".venv" in str(py_file) or "node_modules" in str(py_file):
                    continue
                try:
                    content = py_file.read_text(encoding='utf-8', errors='ignore')
                    # Extract function definitions
                    func_matches = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', content)
                    functions.update(func_matches)
                    # Extract class definitions
                    class_matches = re.findall(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
                    functions.update(class_matches)
                except (OSError, UnicodeDecodeError):
                    continue
        except Exception:
            pass

        return functions

    def _verify_claims(
        self,
        output_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> VerificationResult:
        """Layer 3: Verify factual claims are provable."""
        flagged_claims = []

        # Check for absolute claims
        absolute_patterns = [
            (r'\b(always|never|all|none|every|no one)\b', "absolute_claim"),
            (r'\b(guaranteed|certain|definitely|absolutely)\b', "certainty_claim"),
            (r'\b(the only|the best|the worst|the most)\b', "superlative_claim"),
            (r'\b(\d+%|percent)\b.*\b(of|increase|decrease)\b', "statistical_claim"),
        ]

        for pattern, claim_type in absolute_patterns:
            matches = re.findall(pattern, output_text, re.IGNORECASE)
            if matches:
                # Find the sentence containing the match
                for match in matches[:2]:
                    if isinstance(match, tuple):
                        match_text = match[0]
                    else:
                        match_text = match
                    # Get surrounding context
                    idx = output_text.lower().find(match_text.lower())
                    if idx >= 0:
                        start = max(0, idx - 30)
                        end = min(len(output_text), idx + len(match_text) + 50)
                        context_snippet = output_text[start:end].strip()
                        flagged_claims.append(f"[{claim_type}] ...{context_snippet}...")

        # Check for version/date claims without source
        version_claims = re.findall(
            r'(?:version|v)\s*[\d.]+|(?:since|from)\s+\d{4}|(?:in|released)\s+\d{4}',
            output_text,
            re.IGNORECASE
        )
        for claim in version_claims[:2]:
            flagged_claims.append(f"[version_claim] {claim}")

        # Calculate score based on claim density
        word_count = len(output_text.split())
        claim_density = len(flagged_claims) / max(word_count / 100, 1)

        # Score decreases with more unverified claims
        score = max(0, 1.0 - (claim_density * 0.3))

        # If context is provided, check claim alignment
        if context:
            context_text = json.dumps(context).lower()
            aligned_claims = 0
            for claim in flagged_claims:
                # Check if claim terms appear in context
                claim_terms = self._extract_significant_terms(claim)
                if any(term in context_text for term in claim_terms):
                    aligned_claims += 1

            if flagged_claims:
                alignment_ratio = aligned_claims / len(flagged_claims)
                score = (score + alignment_ratio) / 2

        return VerificationResult(
            layer=VerificationLayer.CLAIM_VERIFICATION,
            passed=score >= 0.6,
            score=score,
            message=f"Claim verification: {len(flagged_claims)} claims flagged for review",
            details={
                "total_claims_flagged": len(flagged_claims),
                "word_count": word_count,
                "claim_density": claim_density
            },
            flagged_items=flagged_claims[:5]
        )

    def _generate_recommendations(self, report: HallucinationReport) -> List[str]:
        """Generate recommendations based on verification results."""
        recommendations = []

        for result in report.results:
            if not result.passed:
                if result.layer == VerificationLayer.SEMANTIC_GROUNDING:
                    recommendations.append(
                        "Review output for claims not grounded in provided context"
                    )
                    if result.details.get("indicator_matches", 0) > 0:
                        recommendations.append(
                            "Remove or qualify absolute language (always, never, clearly, etc.)"
                        )

                elif result.layer == VerificationLayer.ENTITY_VERIFICATION:
                    missing = result.details.get("missing_files", [])
                    if missing:
                        recommendations.append(
                            f"Verify file references exist: {', '.join(missing[:3])}"
                        )
                    missing_funcs = result.details.get("missing_functions", [])
                    if missing_funcs:
                        recommendations.append(
                            f"Verify function references: {', '.join(missing_funcs[:3])}"
                        )

                elif result.layer == VerificationLayer.CLAIM_VERIFICATION:
                    if result.flagged_items:
                        recommendations.append(
                            "Review and qualify or cite sources for flagged claims"
                        )

        if not recommendations and not report.overall_passed:
            recommendations.append(
                f"Overall confidence ({report.confidence_score:.2f}) below threshold "
                f"({self.confidence_threshold:.2f}) - review for accuracy"
            )

        return recommendations


def format_report(report: HallucinationReport, fmt: str = "text") -> str:
    """Format hallucination report for output."""
    if fmt == "json":
        return json.dumps({
            "timestamp": report.timestamp,
            "overall_passed": report.overall_passed,
            "confidence_score": report.confidence_score,
            "results": [
                {
                    "layer": r.layer.value,
                    "passed": r.passed,
                    "score": r.score,
                    "message": r.message,
                    "details": r.details,
                    "flagged_items": r.flagged_items
                }
                for r in report.results
            ],
            "flagged_claims": report.flagged_claims,
            "recommendations": report.recommendations
        }, indent=2)

    # Text format
    lines = [
        "=" * 60,
        "HALLUCINATION VERIFICATION REPORT",
        "=" * 60,
        "",
        f"Timestamp: {report.timestamp}",
        f"Overall: {'PASSED' if report.overall_passed else 'FAILED'}",
        f"Confidence Score: {report.confidence_score:.2%}",
        "",
        "-" * 40,
        "VERIFICATION LAYERS",
        "-" * 40,
    ]

    for result in report.results:
        icon = "[OK]" if result.passed else "[WARN]"
        lines.append(f"\n{icon} {result.layer.value.replace('_', ' ').title()}")
        lines.append(f"    Score: {result.score:.2%}")
        lines.append(f"    {result.message}")

        if result.flagged_items:
            lines.append("    Flagged:")
            for item in result.flagged_items[:3]:
                lines.append(f"      - {item[:60]}...")

    if report.flagged_claims:
        lines.extend([
            "",
            "-" * 40,
            "FLAGGED CLAIMS",
            "-" * 40,
        ])
        for claim in report.flagged_claims[:5]:
            lines.append(f"  - {claim[:70]}...")

    if report.recommendations:
        lines.extend([
            "",
            "-" * 40,
            "RECOMMENDATIONS",
            "-" * 40,
        ])
        for rec in report.recommendations:
            lines.append(f"  - {rec}")

    lines.extend(["", "=" * 60])
    return '\n'.join(lines)


@click.group()
def cli():
    """Hallucination verification for AI outputs."""
    pass


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--context', '-c', type=click.Path(exists=True),
              help='Context JSON file')
@click.option('--workspace', '-w', type=click.Path(exists=True),
              help='Workspace path for entity verification')
@click.option('--json-output', is_flag=True, help='Output as JSON')
def verify(input_file: str, context: Optional[str], workspace: Optional[str],
           json_output: bool):
    """Verify an AI output for hallucinations."""
    try:
        # Load input text
        with open(input_file, 'r', encoding='utf-8') as f:
            if input_file.endswith('.json'):
                data = json.load(f)
                output_text = data.get('output', data.get('text', json.dumps(data)))
            else:
                output_text = f.read()

        # Load context if provided
        context_data = None
        if context:
            with open(context, 'r', encoding='utf-8') as f:
                context_data = json.load(f)

        # Run verification
        workspace_path = Path(workspace) if workspace else None
        verifier = HallucinationVerifier()
        report = verifier.verify(output_text, context_data, workspace_path)

        # Output report
        output = format_report(report, "json" if json_output else "text")
        click.echo(output)

        raise SystemExit(0 if report.overall_passed else 1)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command('check-entities')
@click.option('--text', '-t', required=True, help='Text to check')
@click.option('--workspace', '-w', type=click.Path(exists=True),
              default='.', help='Workspace path')
@click.option('--json-output', is_flag=True, help='Output as JSON')
def check_entities(text: str, workspace: str, json_output: bool):
    """Check entity references in text."""
    try:
        verifier = HallucinationVerifier(Path(workspace))
        result = verifier._verify_entities(text, Path(workspace))

        if json_output:
            click.echo(json.dumps({
                "passed": result.passed,
                "score": result.score,
                "message": result.message,
                "details": result.details
            }, indent=2))
        else:
            status = "PASSED" if result.passed else "FAILED"
            click.echo(f"Entity Check: {status} (score: {result.score:.2%})")
            click.echo(f"\n{result.message}")

            if result.details.get("missing_files"):
                click.echo(f"\nMissing files: {', '.join(result.details['missing_files'])}")
            if result.details.get("missing_functions"):
                click.echo(f"Missing functions: {', '.join(result.details['missing_functions'])}")

        raise SystemExit(0 if result.passed else 1)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option('--text', '-t', required=True, help='Text to analyze')
@click.option('--json-output', is_flag=True, help='Output as JSON')
def analyze_claims(text: str, json_output: bool):
    """Analyze claims in text."""
    try:
        verifier = HallucinationVerifier()
        result = verifier._verify_claims(text, None)

        if json_output:
            click.echo(json.dumps({
                "passed": result.passed,
                "score": result.score,
                "flagged": result.flagged_items,
                "details": result.details
            }, indent=2))
        else:
            status = "PASSED" if result.passed else "NEEDS REVIEW"
            click.echo(f"Claim Analysis: {status} (score: {result.score:.2%})")

            if result.flagged_items:
                click.echo("\nFlagged claims:")
                for claim in result.flagged_items:
                    click.echo(f"  - {claim}")

        raise SystemExit(0 if result.passed else 1)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


if __name__ == '__main__':
    cli()
