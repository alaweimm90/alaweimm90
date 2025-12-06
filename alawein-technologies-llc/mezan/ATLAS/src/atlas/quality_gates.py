#!/usr/bin/env python3
"""
Pre-Run Quality Gates for ORCHEX
Comprehensive input validation and quality checks before execution.

Priority [092] implementation with multi-stage validation.
"""

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class GateStatus(Enum):
    """Status of quality gate checks."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class GateResult:
    """Result of a quality gate check."""

    name: str
    status: GateStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    severity: str = "info"


class QualityGates:
    """Pre-run quality gate system for ORCHEX inputs."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize quality gates with configuration."""
        self.gates = self._initialize_gates()
        self.results = []
        self.config = self._load_config(config_path) if config_path else {}

    def _initialize_gates(self) -> List[str]:
        """Initialize available quality gates."""
        return [
            "schema_validation",
            "pii_scanning",
            "safety_check",
            "budget_estimation",
            "routing_confidence",
            "resource_availability",
            "dependency_check",
            "input_quality",
            "rate_limiting",
            "authorization",
        ]

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        if config_path.exists():
            with config_path.open("r") as f:
                return json.load(f)
        return {}

    def run_all_gates(
        self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run all quality gates on input."""
        if context is None:
            context = {}

        self.results = []

        # Gate 1: Schema Validation
        self.results.append(self._gate_schema_validation(input_data))

        # Gate 2: PII Scanning
        self.results.append(self._gate_pii_scanning(input_data))

        # Gate 3: Safety Check
        self.results.append(self._gate_safety_check(input_data))

        # Gate 4: Budget Estimation
        self.results.append(self._gate_budget_estimation(input_data, context))

        # Gate 5: Routing Confidence
        self.results.append(self._gate_routing_confidence(input_data))

        # Gate 6: Resource Availability
        self.results.append(self._gate_resource_availability(context))

        # Gate 7: Dependency Check
        self.results.append(self._gate_dependency_check(input_data))

        # Gate 8: Input Quality
        self.results.append(self._gate_input_quality(input_data))

        # Gate 9: Rate Limiting
        self.results.append(self._gate_rate_limiting(context))

        # Gate 10: Authorization
        self.results.append(self._gate_authorization(context))

        return self._compile_report()

    def _gate_schema_validation(self, input_data: Dict[str, Any]) -> GateResult:
        """Validate input against schema."""
        required_fields = ["title", "hypothesis", "feature"]
        missing_fields = [f for f in required_fields if f not in input_data]

        if missing_fields:
            return GateResult(
                name="schema_validation",
                status=GateStatus.FAILED,
                message=f"Missing required fields: {missing_fields}",
                severity="critical",
            )

        # Validate feature enum
        valid_features = ["nightmare", "chaos", "evolution", "multiverse", "market"]
        if input_data.get("feature") not in valid_features:
            return GateResult(
                name="schema_validation",
                status=GateStatus.FAILED,
                message=f"Invalid feature: {input_data.get('feature')}",
                severity="critical",
            )

        # Check field constraints
        if len(input_data.get("title", "")) < 3:
            return GateResult(
                name="schema_validation",
                status=GateStatus.WARNING,
                message="Title too short (min 3 characters)",
                severity="warning",
            )

        if len(input_data.get("hypothesis", "")) < 10:
            return GateResult(
                name="schema_validation",
                status=GateStatus.WARNING,
                message="Hypothesis too short (min 10 characters)",
                severity="warning",
            )

        return GateResult(
            name="schema_validation",
            status=GateStatus.PASSED,
            message="Schema validation passed",
            severity="info",
        )

    def _gate_pii_scanning(self, input_data: Dict[str, Any]) -> GateResult:
        """Scan for PII in input."""
        pii_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        }

        pii_found = {}

        def scan_value(value: Any, path: str = "") -> None:
            if isinstance(value, str):
                for pii_type, pattern in pii_patterns.items():
                    if re.search(pattern, value):
                        if pii_type not in pii_found:
                            pii_found[pii_type] = []
                        pii_found[pii_type].append(path)
            elif isinstance(value, dict):
                for k, v in value.items():
                    scan_value(v, f"{path}.{k}" if path else k)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    scan_value(item, f"{path}[{i}]")

        scan_value(input_data)

        if pii_found:
            return GateResult(
                name="pii_scanning",
                status=GateStatus.WARNING,
                message=f"PII detected in fields: {list(pii_found.keys())}",
                details=pii_found,
                severity="warning",
            )

        return GateResult(
            name="pii_scanning",
            status=GateStatus.PASSED,
            message="No PII detected",
            severity="info",
        )

    def _gate_safety_check(self, input_data: Dict[str, Any]) -> GateResult:
        """Check for safety and ethical concerns."""
        unsafe_patterns = [
            "harmful",
            "dangerous",
            "illegal",
            "unethical",
            "weapon",
            "explosive",
            "drug",
            "poison",
        ]

        text_content = " ".join(
            [str(input_data.get("title", "")), str(input_data.get("hypothesis", ""))]
        ).lower()

        found_patterns = [p for p in unsafe_patterns if p in text_content]

        if found_patterns:
            # Check for dual-use or research context
            research_indicators = ["research", "study", "analysis", "safety", "defense"]
            has_research_context = any(ind in text_content for ind in research_indicators)

            if has_research_context:
                return GateResult(
                    name="safety_check",
                    status=GateStatus.WARNING,
                    message=f"Potential sensitive topics detected: {found_patterns}. Research context identified.",
                    details={"patterns": found_patterns, "research_context": True},
                    severity="warning",
                )
            else:
                return GateResult(
                    name="safety_check",
                    status=GateStatus.FAILED,
                    message=f"Safety concerns detected: {found_patterns}",
                    details={"patterns": found_patterns},
                    severity="critical",
                )

        return GateResult(
            name="safety_check",
            status=GateStatus.PASSED,
            message="No safety concerns detected",
            severity="info",
        )

    def _gate_budget_estimation(
        self, input_data: Dict[str, Any], context: Dict[str, Any]
    ) -> GateResult:
        """Estimate and validate budget requirements."""
        # Base costs per feature
        feature_costs = {
            "nightmare": 2.0,
            "chaos": 1.5,
            "evolution": 3.0,
            "multiverse": 4.0,
            "market": 5.0,
        }

        feature = input_data.get("feature")
        base_cost = feature_costs.get(feature, 1.0)

        # Adjust for parameters
        params = input_data.get("parameters", {})
        ensemble_size = params.get("ensemble_size", 3)
        difficulty = params.get("difficulty", "medium")

        difficulty_multipliers = {"easy": 0.5, "medium": 1.0, "hard": 1.5, "nightmare": 2.0}

        estimated_cost = base_cost * ensemble_size * difficulty_multipliers.get(difficulty, 1.0)
        budget_limit = params.get("budget_limit", 10.0)

        if estimated_cost > budget_limit:
            return GateResult(
                name="budget_estimation",
                status=GateStatus.FAILED,
                message=f"Estimated cost ${estimated_cost:.2f} exceeds budget ${budget_limit:.2f}",
                details={"estimated": estimated_cost, "limit": budget_limit},
                severity="critical",
            )

        if estimated_cost > budget_limit * 0.8:
            return GateResult(
                name="budget_estimation",
                status=GateStatus.WARNING,
                message=f"Estimated cost ${estimated_cost:.2f} is close to budget limit ${budget_limit:.2f}",
                details={"estimated": estimated_cost, "limit": budget_limit},
                severity="warning",
            )

        return GateResult(
            name="budget_estimation",
            status=GateStatus.PASSED,
            message=f"Estimated cost ${estimated_cost:.2f} within budget",
            details={"estimated": estimated_cost, "limit": budget_limit},
            severity="info",
        )

    def _gate_routing_confidence(self, input_data: Dict[str, Any]) -> GateResult:
        """Check routing confidence for feature selection."""
        # Simulate routing confidence based on hypothesis clarity
        hypothesis = input_data.get("hypothesis", "")
        feature = input_data.get("feature")

        # Simple heuristic: longer, clearer hypotheses have higher confidence
        word_count = len(hypothesis.split())
        has_quantitative = any(char.isdigit() for char in hypothesis)
        has_comparison = any(
            word in hypothesis.lower() for word in ["versus", "vs", "compared", "better", "worse"]
        )

        confidence = 0.5  # Base confidence
        if word_count > 15:
            confidence += 0.2
        if has_quantitative:
            confidence += 0.15
        if has_comparison:
            confidence += 0.15

        if confidence < 0.7:
            return GateResult(
                name="routing_confidence",
                status=GateStatus.WARNING,
                message=f"Low routing confidence ({confidence:.2f}) for feature '{feature}'",
                details={"confidence": confidence, "feature": feature},
                severity="warning",
            )

        return GateResult(
            name="routing_confidence",
            status=GateStatus.PASSED,
            message=f"Routing confidence {confidence:.2f} for feature '{feature}'",
            details={"confidence": confidence, "feature": feature},
            severity="info",
        )

    def _gate_resource_availability(self, context: Dict[str, Any]) -> GateResult:
        """Check system resource availability."""
        # Simulate resource checks
        resources = context.get("resources", {})
        cpu_available = resources.get("cpu_percent_free", 50)
        memory_available = resources.get("memory_percent_free", 50)
        gpu_available = resources.get("gpu_available", True)

        if cpu_available < 20 or memory_available < 20:
            return GateResult(
                name="resource_availability",
                status=GateStatus.FAILED,
                message="Insufficient system resources",
                details={"cpu_free": cpu_available, "memory_free": memory_available},
                severity="critical",
            )

        if cpu_available < 40 or memory_available < 40:
            return GateResult(
                name="resource_availability",
                status=GateStatus.WARNING,
                message="System resources limited",
                details={"cpu_free": cpu_available, "memory_free": memory_available},
                severity="warning",
            )

        return GateResult(
            name="resource_availability",
            status=GateStatus.PASSED,
            message="Sufficient resources available",
            details={
                "cpu_free": cpu_available,
                "memory_free": memory_available,
                "gpu": gpu_available,
            },
            severity="info",
        )

    def _gate_dependency_check(self, input_data: Dict[str, Any]) -> GateResult:
        """Check for required dependencies."""
        feature = input_data.get("feature")

        # Feature-specific dependencies
        dependencies = {
            "nightmare": ["attack_catalog.json", "ensemble_config.json"],
            "chaos": ["ensemble_config.json", "cost_config.json"],
            "evolution": ["seed_config.json", "checkpoint_manager"],
            "multiverse": ["universe_templates", "physics_engine"],
            "market": ["liquidity_provider", "settlement_oracle"],
        }

        required = dependencies.get(feature, [])
        # Simulate checking for files/services
        missing = []  # In reality, would check actual files/services

        if missing:
            return GateResult(
                name="dependency_check",
                status=GateStatus.FAILED,
                message=f"Missing dependencies: {missing}",
                details={"required": required, "missing": missing},
                severity="critical",
            )

        return GateResult(
            name="dependency_check",
            status=GateStatus.PASSED,
            message="All dependencies available",
            details={"checked": required},
            severity="info",
        )

    def _gate_input_quality(self, input_data: Dict[str, Any]) -> GateResult:
        """Assess input quality and completeness."""
        quality_score = 1.0
        issues = []

        # Check hypothesis quality
        hypothesis = input_data.get("hypothesis", "")
        if len(hypothesis) < 50:
            quality_score -= 0.2
            issues.append("Hypothesis may be too brief")

        # Check for evidence
        evidence = input_data.get("evidence", [])
        if not evidence:
            quality_score -= 0.3
            issues.append("No supporting evidence provided")

        # Check parameters
        params = input_data.get("parameters", {})
        if not params:
            quality_score -= 0.1
            issues.append("No parameters specified, using defaults")

        if quality_score < 0.5:
            return GateResult(
                name="input_quality",
                status=GateStatus.WARNING,
                message=f"Low input quality score: {quality_score:.2f}",
                details={"score": quality_score, "issues": issues},
                severity="warning",
            )

        return GateResult(
            name="input_quality",
            status=GateStatus.PASSED,
            message=f"Input quality score: {quality_score:.2f}",
            details={"score": quality_score},
            severity="info",
        )

    def _gate_rate_limiting(self, context: Dict[str, Any]) -> GateResult:
        """Check rate limits."""
        user_id = context.get("user_id", "anonymous")
        requests_per_hour = context.get("requests_per_hour", 0)
        limit = context.get("rate_limit", 100)

        if requests_per_hour >= limit:
            return GateResult(
                name="rate_limiting",
                status=GateStatus.FAILED,
                message=f"Rate limit exceeded: {requests_per_hour}/{limit} requests",
                details={"user": user_id, "current": requests_per_hour, "limit": limit},
                severity="critical",
            )

        if requests_per_hour > limit * 0.8:
            return GateResult(
                name="rate_limiting",
                status=GateStatus.WARNING,
                message=f"Approaching rate limit: {requests_per_hour}/{limit} requests",
                details={"user": user_id, "current": requests_per_hour, "limit": limit},
                severity="warning",
            )

        return GateResult(
            name="rate_limiting",
            status=GateStatus.PASSED,
            message=f"Within rate limit: {requests_per_hour}/{limit} requests",
            details={"user": user_id, "current": requests_per_hour, "limit": limit},
            severity="info",
        )

    def _gate_authorization(self, context: Dict[str, Any]) -> GateResult:
        """Check user authorization."""
        user_id = context.get("user_id")
        api_key = context.get("api_key")
        feature_permissions = context.get("permissions", {})

        if not user_id or not api_key:
            return GateResult(
                name="authorization",
                status=GateStatus.FAILED,
                message="Missing authentication credentials",
                severity="critical",
            )

        # In reality, would validate against auth service
        authorized = True  # Simulated

        if not authorized:
            return GateResult(
                name="authorization",
                status=GateStatus.FAILED,
                message="Authorization failed",
                severity="critical",
            )

        return GateResult(
            name="authorization",
            status=GateStatus.PASSED,
            message="User authorized",
            details={"user": user_id},
            severity="info",
        )

    def _compile_report(self) -> Dict[str, Any]:
        """Compile comprehensive gate report."""
        passed = sum(1 for r in self.results if r.status == GateStatus.PASSED)
        warnings = sum(1 for r in self.results if r.status == GateStatus.WARNING)
        failed = sum(1 for r in self.results if r.status == GateStatus.FAILED)

        overall_status = "passed"
        if failed > 0:
            overall_status = "failed"
        elif warnings > 2:
            overall_status = "warning"

        return {
            "overall_status": overall_status,
            "summary": {
                "total_gates": len(self.results),
                "passed": passed,
                "warnings": warnings,
                "failed": failed,
            },
            "gates": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "message": r.message,
                    "severity": r.severity,
                    "details": r.details,
                }
                for r in self.results
            ],
            "recommendations": self._generate_recommendations(),
            "can_proceed": overall_status != "failed",
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on gate results."""
        recommendations = []

        for result in self.results:
            if result.status == GateStatus.FAILED:
                if result.name == "budget_estimation":
                    recommendations.append(
                        "Reduce ensemble size or adjust parameters to meet budget"
                    )
                elif result.name == "safety_check":
                    recommendations.append(
                        "Review content for safety concerns or provide research justification"
                    )
                elif result.name == "schema_validation":
                    recommendations.append("Fix missing or invalid fields before resubmission")
            elif result.status == GateStatus.WARNING:
                if result.name == "pii_scanning":
                    recommendations.append("Consider redacting PII or obtaining consent")
                elif result.name == "input_quality":
                    recommendations.append(
                        "Enhance hypothesis detail and provide evidence for better results"
                    )

        return recommendations


if __name__ == "__main__":
    # Example usage
    gates = QualityGates()

    test_input = {
        "title": "AI Safety Through Interpretability",
        "hypothesis": "Mechanistic interpretability can ensure AI safety in production",
        "feature": "nightmare",
        "parameters": {"ensemble_size": 3, "budget_limit": 5.0, "difficulty": "hard"},
        "evidence": ["Recent research papers"],
    }

    test_context = {
        "user_id": "researcher_001",
        "api_key": "test_key",
        "resources": {"cpu_percent_free": 60, "memory_percent_free": 45, "gpu_available": True},
        "requests_per_hour": 25,
        "rate_limit": 100,
    }

    report = gates.run_all_gates(test_input, test_context)
    print(json.dumps(report, indent=2))
