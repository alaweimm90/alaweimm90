# OPA Policy Enforcement

This directory contains Open Policy Agent (OPA) policies for governance enforcement.

## Policies

### 1. Repository Structure (`repo-structure.rego`)
Enforces allowed root directory structure for the multi-org monorepo.

**Checks**:
- ✅ Files only in allowed root directories
- ✅ No forbidden file patterns (.DS_Store, *.log, etc.)
- ✅ Dockerfiles in appropriate service directories
- ⚠️ Large file warnings (>10MB)

**Allowed Roots**:
- `.github/`, `.metaHub/`, `.config/`
- `apps/`, `packages/`, `alaweimm90/`
- `ops/`, `scripts/`, `templates/`, `docs/`
- Standard files: `README.md`, `package.json`, `docker-compose.yml`, etc.

### 2. Docker Security (`docker-security.rego`)
Enforces security best practices in Dockerfiles.

**Checks**:
- ✅ Must include USER directive (non-root)
- ✅ Must include HEALTHCHECK
- ✅ No `:latest` tags in FROM directives
- ✅ All base images must be tagged
- ⚠️ Prefer COPY over ADD
- ✅ apt-get install must use -y flag
- ⚠️ apt-get should include cleanup
- ⚠️ COPY should use --chown with USER
- ✅ No privileged ports (<1024) in EXPOSE
- ✅ No secrets in ENV variables

## Pre-commit Hook Integration

The `pre-commit-opa.sh` script integrates OPA validation into your git workflow.

### Installation

```bash
# Link to git hooks (if using standard git hooks)
ln -sf ../../.metaHub/policies/pre-commit-opa.sh .git/hooks/pre-commit

# Or use with husky (recommended)
# Add to .husky/pre-commit:
# .metaHub/policies/pre-commit-opa.sh
```

### Usage

The hook automatically runs on `git commit` and will:
1. Check all staged files against repository structure policy
2. Check all staged Dockerfiles against security policy
3. Block commit if any violations found
4. Show warnings for non-critical issues

### Manual Testing

Test policies manually:

```bash
# Test repository structure
opa eval -d .metaHub/policies/repo-structure.rego \
  -i '{"file": {"path": "forbidden/file.js", "size": 1024}}' \
  "data.repo_structure.deny"

# Test Dockerfile security
opa eval -d .metaHub/policies/docker-security.rego \
  -i '{"dockerfile": "FROM node:20-alpine\nRUN apt-get install -y curl\nUSER nodejs\nHEALTHCHECK CMD curl localhost"}' \
  "data.docker_security.deny"
```

### Bypassing Policies

**Emergency bypass** (use sparingly):
```bash
git commit --no-verify -m "emergency: description"
```

**Note**: Bypassed commits will be flagged in CI/CD for review.

## CI/CD Integration

Policies are also enforced in GitHub Actions workflows:

- `.github/workflows/policy-enforcement.yml` (if created)
- Part of CI matrix builds
- Blocks PRs with policy violations

## Policy Development

### Adding New Policies

1. Create new `.rego` file in this directory
2. Follow OPA policy structure:
   ```rego
   package policy_name

   deny[msg] {
       # Denial logic
       msg := "Error message"
   }

   warn[msg] {
       # Warning logic
       msg := "Warning message"
   }
   ```

3. Add to pre-commit hook script
4. Test with sample inputs
5. Document in this README

### Testing Policies

```bash
# Run OPA tests
opa test .metaHub/policies/*.rego -v

# Format policies
opa fmt -w .metaHub/policies/
```

## Resources

- [OPA Documentation](https://www.openpolicyagent.org/docs/latest/)
- [Rego Language Guide](https://www.openpolicyagent.org/docs/latest/policy-language/)
- [OPA Playground](https://play.openpolicyagent.org/)

---


