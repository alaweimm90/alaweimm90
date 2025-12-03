#!/usr/bin/env python3
"""Technical Debt Gate

Simple policy gate that consumes debt_scan.json or remediation_summary.json
and exits non-zero when technical debt thresholds are exceeded.

Usage:
    python debt_gate.py --scan debt_scan.json
    python debt_gate.py --remediation remediation_summary.json

Intended for CI, MCP agents, and automation to enforce basic quality rules.
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"❌ JSON file not found: {p}")
        sys.exit(2)
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load JSON from {p}: {e}")
        sys.exit(2)


def load_technical_debt_policy(env: str | None = None) -> dict:
    """Load technical debt policy for the given environment from YAML.

    Falls back to in-code defaults when the policy file or PyYAML
    is unavailable.
    """

    base_env = env or os.getenv("TECHNICAL_DEBT_ENV", "default")

    # Default in-code policy mirrors previous hardcoded behavior
    default_policy = {
        "scan": {
            "fail_on_critical": True,
            "high_ratio_limit": 0.9,
            "max_total_items": None,
        },
        "remediation": {
            "fail_on_critical": True,
            "warn_on_zero_success": True,
            "min_success_rate": 0.0,
        },
    }

    policy_path = (
        Path(__file__).parent
        / "governance"
        / "policies"
        / "technical_debt.yaml"
    )

    if yaml is None or not policy_path.exists():
        return default_policy

    try:
        with policy_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return default_policy

    envs = data.get("environments", {})
    env_policy = envs.get(base_env) or envs.get("default")
    if not isinstance(env_policy, dict):
        return default_policy

    # Ensure both scan and remediation keys exist with sane fallbacks
    scan = env_policy.get("scan") or default_policy["scan"]
    remediation = env_policy.get("remediation") or default_policy["remediation"]

    return {"scan": scan, "remediation": remediation}


def evaluate_scan_policy(data: dict, env: str | None = None) -> int:
    """Evaluate basic policies on a debt_scan.json snapshot.

    Returns an exit code: 0 = pass, non-zero = fail.
    """

    assessment = data.get("assessment", {})
    severity = assessment.get("debt_by_severity", {})

    policy = load_technical_debt_policy(env)["scan"]

    critical = int(severity.get("critical", 0))
    high = int(severity.get("high", 0))
    total = int(assessment.get("total_debt_items", 0))

    failed = False

    fail_on_critical = bool(policy.get("fail_on_critical", True))
    high_ratio_limit = policy.get("high_ratio_limit")
    max_total_items = policy.get("max_total_items")

    if fail_on_critical and critical > 0:
        print(f"❌ Policy failed: {critical} critical technical debt item(s) present.")
        failed = True

    if total > 0 and high_ratio_limit is not None:
        high_ratio = high / total
        if high_ratio > float(high_ratio_limit):
            print(
                f"❌ Policy failed: high-severity ratio {high_ratio:.1%} "
                f"exceeds {float(high_ratio_limit):.0%}."
            )
            failed = True

    if max_total_items is not None and total > int(max_total_items):
        print(
            f"❌ Policy failed: total technical debt items {total} "
            f"exceeds limit {int(max_total_items)}."
        )
        failed = True

    if not failed:
        print("✅ Technical debt scan passes gate policies.")
        print(f"   Total items: {total}, critical: {critical}, high: {high}")
        return 0

    print("⚠️ Technical debt scan does NOT meet gate policies.")
    return 1


def evaluate_remediation_policy(data: dict, env: str | None = None) -> int:
    """Evaluate policies on a remediation_summary.json snapshot."""

    assessment = data.get("assessment", {})
    results = data.get("remediation_results", {})

    severity = assessment.get("debt_by_severity", {})
    critical = int(severity.get("critical", 0))

    total_items = int(results.get("total_items", 0))
    success_rate = float(results.get("success_rate", 0.0))

    policy = load_technical_debt_policy(env)["remediation"]

    failed = False

    fail_on_critical = bool(policy.get("fail_on_critical", True))
    warn_on_zero_success = bool(policy.get("warn_on_zero_success", True))
    min_success_rate = float(policy.get("min_success_rate", 0.0))

    if fail_on_critical and critical > 0:
        print(f"❌ Policy failed: {critical} critical technical debt item(s) present.")
        failed = True

    if total_items > 0 and success_rate < min_success_rate:
        print(
            f"❌ Policy failed: remediation success rate {success_rate:.1%} "
            f"is below minimum {min_success_rate:.1%}."
        )
        failed = True

    if total_items > 0 and success_rate <= 0.0 and not failed and warn_on_zero_success:
        print("⚠️ Warning: remediation success rate is 0.0% for attempted items.")

    if not failed:
        print("✅ Remediation summary passes gate policies.")
        print(f"   Attempted: {total_items}, success rate: {success_rate:.1%}")
        return 0

    print("⚠️ Remediation summary does NOT meet gate policies.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Technical Debt Policy Gate")

    parser.add_argument(
        "--scan",
        help="Path to debt_scan.json (assessment snapshot)",
    )
    parser.add_argument(
        "--remediation",
        help="Path to remediation_summary.json (full workflow summary)",
    )

    parser.add_argument(
        "--env",
        help="Policy environment to use (default, ci, prod_release, ...)",
    )

    args = parser.parse_args()

    if not args.scan and not args.remediation:
        parser.print_help()
        return 1

    if args.scan:
        data = load_json(args.scan)
        return evaluate_scan_policy(data, env=args.env)

    if args.remediation:
        data = load_json(args.remediation)
        return evaluate_remediation_policy(data, env=args.env)

    return 1


if __name__ == "__main__":
    sys.exit(main())
