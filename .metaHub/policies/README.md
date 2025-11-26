# Governance Policies

<img src="https://img.shields.io/badge/Engine-OPA/Rego-A855F7?style=flat-square&labelColor=1a1b27" alt="OPA"/>
<img src="https://img.shields.io/badge/Mode-Warning_Only-F59E0B?style=flat-square&labelColor=1a1b27" alt="Mode"/>
<img src="https://img.shields.io/badge/Policies-6-4CC9F0?style=flat-square&labelColor=1a1b27" alt="Count"/>

---

> OPA/Rego policies that enforce governance across the portfolio.

All policies run in **warning-only mode** — violations generate warnings but don't block commits. This allows teams to learn before enforcement tightens.

---

## Available Policies

### 1. Repository Structure

**File:** `repo-structure.rego`

| Check | Description |
|-------|-------------|
| Root files | Only allowed files in root directory |
| `.metaHub/` structure | Proper subdirectory organization |
| Large files | Files >10MB flagged for Git LFS |

### 2. Docker Security

**File:** `docker-security.rego`

| Check | Description |
|-------|-------------|
| `USER` directive | Non-root user required |
| `HEALTHCHECK` | Health check required |
| Base image tags | No `:latest` tags |
| Secrets | No hardcoded secrets in `ENV` |

### 3. Kubernetes Governance

**File:** `k8s-governance.rego`

| Check | Description |
|-------|-------------|
| Resources | Requests and limits defined |
| Probes | Liveness and readiness configured |
| Security context | Properly set |
| Network policies | In place when applicable |

### 4. Service SLO

**File:** `service-slo.rego`

| Check | Description |
|-------|-------------|
| SLO definitions | Present in `.meta/repo.yaml` |
| Availability | Realistic targets |
| Error budgets | Specified |
| Monitoring | Configured |

### 5. Architecture Decision Records

**File:** `adr-policy.rego`

| Check | Description |
|-------|-------------|
| ADR existence | Present for major decisions |
| Format | Standard format followed |
| Tracking | Decisions approved and tracked |

### 6. Dependency Security

**File:** `dependency-security.rego`

| Check | Description |
|-------|-------------|
| Lock files | Present and up-to-date |
| Known vulnerabilities | No critical CVEs |
| License compliance | Approved licenses only |

---

## Usage

### In Consumer Repositories

**Option 1: OPA Bundle Reference**

```bash
opa eval -d https://github.com/alaweimm90/alaweimm90/.metaHub/policies \
  -i <(cat .meta/repo.yaml) 'data.repo.warn'
```

**Option 2: GitHub Actions Workflow**

```yaml
jobs:
  policies:
    uses: alaweimm90/alaweimm90/.github/workflows/reusable-policy.yml@main
```

**Option 3: Pre-commit Hook**

```bash
./.metaHub/policies/pre-commit-opa.sh
```

### Local Testing

```bash
# Install OPA
curl -L -o opa https://openpolicyagent.org/downloads/latest/opa_linux_x86_64
chmod +x opa && sudo mv opa /usr/local/bin/

# Test policy
opa eval -d .metaHub/policies -i <(cat .meta/repo.yaml) 'data.repo.warn'
```

---

## Adding New Policies

1. Create `.rego` file in this directory
2. Define rules in `warn[]` blocks (warning-only) or `deny[]` blocks (blocking)
3. Document in this README
4. Test against example repos
5. Submit PR

**Example structure:**

```rego
package my_policy

warn[msg] {
    input.some_field == "invalid"
    msg := "Description of why this is a warning"
}

pass = true
```

---

## Philosophy

| Principle | Description |
|-----------|-------------|
| **Learn Before Enforcement** | Policies warn before they block |
| **Documentation as Code** | Policy rules are source of truth |
| **Centralized Updates** | Changes propagate to all consumers |
| **Non-Breaking** | Warnings don't block work |

---

**See also:** [Governance README](../README.md) · [Consumer Guide](../guides/consumer-guide.md)
