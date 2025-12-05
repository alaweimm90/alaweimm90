# Governance Ownership

This document defines who owns technical-debt, security, and policy configuration in the `automation` repo.

## 1. Policy ownership

- **Technical debt policies**
  - Source of truth: `governance/policies/technical_debt.yaml`.
  - Primary owners: the engineering team responsible for build/CI and code quality.
  - Changes SHOULD:
    - Be reviewed like code (PRs, code review).
    - Consider the impact on CI (`ci` env) and releases (`prod_release` env).

- **Security scanning policies**
  - Source of truth: `governance/policies/security_scanning.yaml`.
  - Primary owners: security or platform engineering.
  - Changes SHOULD:
    - Preserve strong detection for real secrets.
    - Avoid noisy false positives by using `excludes.paths` and `excludes.markers` for demos/examples.

## 2. Responsibilities

- **Developers**
  - Keep code aligned with current policies.
  - Mark demo/example content clearly (for example `# Example only`, `# DEMO ONLY`).
  - Use `debt_cli.py` locally to understand the impact of changes before opening a PR.

- **CI / DevEx maintainers**
  - Own how `debt_cli.py` and `debt_gate.py` are wired into CI.
  - Decide which environments (`default`, `ci`, `prod_release`) are used in which pipelines.
  - Tune thresholds in `technical_debt.yaml` when the signal-to-noise ratio changes (for example after scanner refinements).

- **Security / Risk owners**
  - Review security findings coming from the `security_issues` debt type.
  - Decide when to treat security issues as hard gates vs warnings.

## 3. Change management

- **When to change policies**
  - The scanner behavior or heuristics have changed significantly.
  - The team repeatedly hits CI or release gates for reasons that are not actionable.
  - New risk tolerance or compliance requirements are introduced.

- **How to change policies safely**
  - Propose edits to the relevant YAML file under `governance/policies/`.
  - Run:

    ```bash
    python debt_cli.py scan --path . --json debt_scan.json
    python debt_gate.py --scan debt_scan.json --env default
    python debt_gate.py --scan debt_scan.json --env ci
    ```

  - Confirm that changes make gates more realistic without silently hiding critical issues.

## 4. Long-term evolution

- Treat the YAML policies as versioned configuration.
- Consider tagging policy changes in release notes when they materially affect CI or release behavior.
- Over time, converge physical layout toward the structure documented in `governance/structure.md` so that ownership boundaries stay clear.
