"""
Failure classifier - Automatically classifies failure types and severity
"""

from typing import Tuple
from hall_of_failures.models import Failure, FailureType, Severity


class FailureClassifier:
    """
    Classify failures by type and severity

    Uses keyword matching and pattern recognition.
    Can be enhanced with ML models.
    """

    # Keywords for each failure type
    TYPE_KEYWORDS = {
        FailureType.HYPOTHESIS: [
            "unfalsifiable", "contradiction", "logical flaw", "vague",
            "not testable", "ambiguous", "poorly defined", "claim",
        ],
        FailureType.EXPERIMENTAL: [
            "sample size", "control", "confounding", "bias",
            "underpowered", "experimental design", "methodology",
            "measurement", "protocol", "data collection",
        ],
        FailureType.COMPUTATIONAL: [
            "bug", "error", "crash", "numerical", "implementation",
            "algorithm", "code", "performance", "timeout",
            "memory", "overflow", "precision",
        ],
        FailureType.INTEGRATION: [
            "integration", "API", "interface", "compatibility",
            "format", "protocol", "communication", "data format",
            "mismatch", "connection",
        ],
        FailureType.THEORETICAL: [
            "violates", "impossible", "theory", "physics",
            "mathematics", "principle", "law", "theorem",
            "contradiction", "mechanism",
        ],
    }

    # Keywords for severity
    SEVERITY_KEYWORDS = {
        Severity.CRITICAL: [
            "fundamental", "fatal", "impossible", "violates",
            "completely wrong", "invalidates", "catastrophic",
        ],
        Severity.MAJOR: [
            "significant", "major", "serious", "substantial",
            "large", "important", "critical issue",
        ],
        Severity.MINOR: [
            "minor", "small", "trivial", "slight", "fixable",
            "easy fix", "cosmetic",
        ],
    }

    def classify(self, failure: Failure) -> Tuple[FailureType, Severity]:
        """
        Classify failure type and severity

        Args:
            failure: Failure to classify

        Returns:
            (failure_type, severity) tuple
        """
        # If already classified, return as-is
        if failure.failure_type and failure.severity:
            return failure.failure_type, failure.severity

        # Classify type
        failure_type = self._classify_type(failure)

        # Classify severity
        severity = self._classify_severity(failure)

        return failure_type, severity

    def _classify_type(self, failure: Failure) -> FailureType:
        """Classify failure type"""
        text = f"{failure.hypothesis} {failure.description}".lower()

        # Count keyword matches for each type
        scores = {}
        for ftype, keywords in self.TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            scores[ftype] = score

        # Return type with highest score
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)

        # Default
        return FailureType.HYPOTHESIS

    def _classify_severity(self, failure: Failure) -> Severity:
        """Classify failure severity"""
        text = f"{failure.hypothesis} {failure.description}".lower()

        # Check for critical keywords first
        for kw in self.SEVERITY_KEYWORDS[Severity.CRITICAL]:
            if kw in text:
                return Severity.CRITICAL

        # Check for major keywords
        for kw in self.SEVERITY_KEYWORDS[Severity.MAJOR]:
            if kw in text:
                return Severity.MAJOR

        # Check for minor keywords
        for kw in self.SEVERITY_KEYWORDS[Severity.MINOR]:
            if kw in text:
                return Severity.MINOR

        # Default to MAJOR
        return Severity.MAJOR

    def identify_root_causes(self, failure: Failure) -> list[str]:
        """
        Identify potential root causes

        Returns:
            List of identified root causes
        """
        causes = []
        text = failure.description.lower()

        # Common root causes by pattern
        cause_patterns = {
            "Insufficient sample size": ["sample size", "underpowered", "too small"],
            "Lack of proper controls": ["no control", "lacking control", "insufficient control"],
            "Confounding variables": ["confounding", "confound", "hidden variable"],
            "Selection bias": ["selection bias", "biased sample"],
            "Measurement error": ["measurement error", "measurement noise"],
            "Logical contradiction": ["contradiction", "contradicts", "inconsistent"],
            "Unfalsifiable claim": ["unfalsifiable", "not testable", "cannot test"],
            "Implementation bug": ["bug", "error in code", "implementation error"],
            "Numerical instability": ["numerical", "overflow", "underflow"],
            "Violation of assumptions": ["violates", "assumption violated"],
        }

        for cause, patterns in cause_patterns.items():
            if any(pattern in text for pattern in patterns):
                causes.append(cause)

        return causes if causes else ["Unspecified root cause"]
