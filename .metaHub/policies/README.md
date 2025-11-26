# Governance Policies

This directory contains OPA/Rego policies that enforce governance across the portfolio.

## Available Policies

All policies run in **warning-only mode** — violations generate warnings but don't block commits. This allows teams to learn before enforcement tightens.

### 1. Repository Structure (`repo-structure.rego`)

Validates repository organization and file placement.

**What it checks:**
- Root files are in the allowed list (.github, .metaHub, .allstar, .gitignore, .gitattributes, README.md, LICENSE)
- .metaHub subdirectories are properly organized (policies/, schemas/, infra/examples/)
- Large files (>10MB) are flagged for Git LFS

**Mode:** Warning-only

### 2. Docker Security (`docker-security.rego`)

Enforces Docker container best practices and security.

**What it checks:**
- Non-root USER directive (required for security)
- HEALTHCHECK directive (required for reliability)
- Base image tags are not `:latest` (prevents version drift)
- No hardcoded secrets in ENV variables

**Mode:** Warning-only

### 3. Kubernetes Governance (`k8s-governance.rego`)

Validates Kubernetes manifests for security and compliance.

**What it checks:**
- Resource requests and limits are defined
- Liveness and readiness probes are configured
- Security context is properly set
- Network policies are in place (when applicable)

**Mode:** Warning-only

### 4. Service SLO (`service-slo.rego`)

Ensures service-level objectives are defined and tracked.

**What it checks:**
- `.meta/repo.yaml` includes SLO definitions
- Availability targets are realistic
- Error budgets are specified
- Monitoring and alerting are configured

**Mode:** Warning-only

### 5. Architecture Decision Records (`adr-policy.rego`)

Ensures significant decisions are documented.

**What it checks:**
- ADRs exist for major architectural decisions
- ADRs follow the standard format
- Decisions are approved and tracked

**Mode:** Warning-only

## How to Use These Policies

### In Consumer Repositories

Consumer repos reference these policies through:

1. **OPA Bundle Reference** (centralizes policy updates)
   ```bash
   opa eval -d https://github.com/alaweimm90/alaweimm90/.metaHub/policies \
     -i <(cat .meta/repo.yaml) 'data.repo.warn'
   ```

2. **GitHub Actions Workflow**
   Consumer repos call the reusable policy workflow:
   ```yaml
   jobs:
     policies:
       uses: alaweimm90/alaweimm90/.github/workflows/reusable-policy.yml@main
   ```

3. **Pre-commit Hook** (local validation before commit)
   ```bash
   ./.metaHub/policies/pre-commit-opa.sh
   ```

### Local Policy Testing

```bash
# Install OPA
curl -L -o opa https://openpolicyagent.org/downloads/latest/opa_linux_x86_64
chmod +x opa
sudo mv opa /usr/local/bin/

# Test a policy against a repo.yaml
opa eval -d .metaHub/policies -i <(cat .meta/repo.yaml) 'data.repo.warn'
```

## Adding New Policies

To add a new governance policy:

1. Create a new `.rego` file in this directory
2. Define policy rules in `warn[]` blocks (warning-only) or `deny[]` blocks (blocking, optional)
3. Document the policy in this README
4. Test the policy against example repos
5. Submit PR for review

**Example policy structure:**
```rego
package my_policy

warn[msg] {
    # Check condition
    input.some_field == "invalid"
    msg := "Description of why this is a warning"
}

pass = true
```

## Policy Philosophy

- **Learn Before Enforcement:** Policies warn before they block
- **Documentation as Code:** Policy rules are the source of truth
- **Centralized Updates:** Changes to this repo propagate to all consumers
- **Non-Breaking:** Warnings don't block work, allowing gradual adoption

---

**For more information:** See [parent README](../../README.md) and [Consumer Guide](.../guides/consumer-guide.md)
