# Governance Policies

OPA/Rego policies that enforce repository structure and practices across the portfolio.

## Available Policies

### `repo-structure.rego`
Enforces canonical repository structure at root level and within `.metaHub/` directory. This policy documents the contract that consumer repositories should follow.

**Enforced:**
- Root files must be in allowed list (policies, workflows, configs only)
- `.metaHub/` must contain only policies/, schemas/, infra/examples/
- Forbidden patterns blocked (node_modules, .env, *.log, etc.)

**Mode:** Warning-only (non-blocking)

### `docker-security.rego`
Docker security best practices and container image hardening.

**Enforced:**
- Non-root USER directive required
- HEALTHCHECK recommended for services
- Latest tags forbidden (use specific versions)
- Secrets not allowed in ENV variables
- Best practices for apt-get, COPY vs ADD, etc.

### `k8s-governance.rego`
Kubernetes manifest validation for services deployed to clusters.

**Enforced:**
- Required labels (owner, environment)
- No privileged containers
- Resource requests/limits recommended

### `service-slo.rego`
Service-level objective enforcement for critical services.

**Enforced:**
- SLO blocks required in service catalog
- Availability and latency targets defined

### `adr-policy.rego`
Architecture Decision Records (ADRs) presence check.

**Enforced:**
- ADR directory recommended (docs/adr/ or adr/)
- Documents major architectural decisions

## How Consumer Repos Use These Policies

Consumer repositories reference these policies via:

1. **OPA Bundle Reference** (recommended) — Centralizes policy updates
   ```bash
   opa eval -d <this-repo>/policies/ -i repo-snapshot.json 'data.repo.deny'
   ```

2. **GitHub Actions Workflow** — In consumer's `.github/workflows/policy.yml`
   ```yaml
   - uses: open-policy-agent/setup-opa@v2
   - run: |
       opa eval -d <this-repo>/policies/ \
         -i <(./scripts/repo-snapshot.sh) \
         'data.repo.deny'
   ```

3. **Pre-Commit Hook** (local validation)
   ```bash
   conftest test --policy .metaHub/policies/ Dockerfile
   ```

## Adding New Policies

To add a new policy:

1. Create `policy-name.rego` in this directory
2. Document in this README
3. Test with: `opa eval -d . -i test-data.json 'data'`
4. Consumer repos automatically pick up new policies

## References

- **OPA/Rego Docs:** https://www.openpolicyagent.org/docs/latest/
- **Conftest (OPA CLI):** https://www.conftest.dev/
- **Root README:** `../../README.md` for consumption guide
