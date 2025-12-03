# Repository Governance & Structure

This document describes how the `automation` repo is organized for AI, technical debt, and policy governance.

## 1. High-level domains

- **AI Integrations**
  - Files that integrate with external AI providers (Claude, OpenAI, etc.).
  - Current examples:
    - `claude_http_integration.py` (HTTP-based, canonical for new work).
    - `claude_integration.py` (SDK-based, legacy).
    - `services/integration_coordinator.py`, `services/claude_reasoning.py` (service orchestration and analysis).

- **Technical Debt System**
  - Files that implement scanning, planning, and gating of technical debt.
  - Current examples:
    - `technical_debt_remediation.py`
    - `remediation_plan_generator.py`
    - `debt_cli.py`
    - `debt_gate.py`

- **Governance & Policies**
  - Files in `governance/` that describe structure, ownership, and machine-readable policies.

## 2. Target directory layout (north star)

Over time, code should converge toward this logical layout (names may map to existing files):

```text
automation/
  governance/
    structure.md
    ownership.md            # optional, for code ownership / responsibilities
    policies/
      technical_debt.yaml   # thresholds for gates
      security_scanning.yaml# secret patterns, excludes

  ai_integrations/
    claude/
      claude_http_integration.py   # canonical Claude HTTP integration
      claude_sdk_legacy.py         # SDK-based legacy integration
    services/
      integration_coordinator.py
      claude_reasoning.py

  technical_debt/
    technical_debt_remediation.py
    remediation_plan_generator.py
    debt_cli.py
    debt_gate.py
```

This document is descriptive, not yet enforced. Physical moves can be staged to avoid breaking imports.

## 3. Canonical vs legacy vs demo

- **Canonical**
  - `claude_http_integration.py` for Claude over HTTP.
  - `technical_debt_remediation.py`, `debt_cli.py`, `debt_gate.py` for technical debt.

- **Legacy**
  - `claude_integration.py` (SDK-based); keep available for backward compatibility but do not use for new integrations.

- **Demo / examples**
  - Any file or snippet explicitly marked with comments like `# Example only` or `# DEMO ONLY`.
  - Security scanners and gates should treat these more leniently (e.g., via `security_scanning.yaml`).

## 4. Policy-as-data principle

- All thresholds, regexes, and environment-specific rules SHOULD live in `governance/policies/*.yaml`.
- Python and TypeScript code SHOULD read from these policies, with safe defaults when the files are missing.

This keeps governance transparent and makes it easier for CI, MCP agents, and humans to reason about system behavior.
