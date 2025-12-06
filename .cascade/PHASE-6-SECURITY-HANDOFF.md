# 🔒 PHASE 6: SECURITY HARDENING - CASCADE HANDOFF

## Mission
Complete security hardening for the entire repository in a single session. AI-accelerated execution expected: 30-60 minutes.

## Context
- **Repository**: Meta-governance monorepo with 3 LLCs, research projects, automation systems
- **Current State**: Phase 5 (CI/CD) complete, security scanning partially configured
- **Goal**: Enterprise-grade security before Q1 2025 multi-site launch

---

## Tasks (Execute in Order)

### 1. Enable Dependabot Everywhere (5 min)
Create/update `.github/dependabot.yml`:
```yaml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
    groups:
      dev-dependencies:
        patterns: ["*"]
        dependency-type: "development"
  - package-ecosystem: "pip"
    directory: "/automation"
    schedule:
      interval: "weekly"
  - package-ecosystem: "pip"
    directory: "/alawein-technologies-llc/librex"
    schedule:
      interval: "weekly"
  - package-ecosystem: "pip"
    directory: "/research/maglogic"
    schedule:
      interval: "weekly"
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

### 2. Add Snyk Security Scanning to CI (10 min)
Update `.github/workflows/ci.yml` to include:
```yaml
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Snyk to check for vulnerabilities
        uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          args: --severity-threshold=high
```

### 3. Create Security Policy (5 min)
Ensure `SECURITY.md` exists at root with:
- Supported versions
- Vulnerability reporting process
- Security contact email
- Disclosure timeline (90 days)

### 4. Secret Scanning Configuration (5 min)
Create `.github/secret_scanning.yml`:
```yaml
paths-ignore:
  - "docs/**"
  - "*.md"
  - "tests/fixtures/**"
```

### 5. Add Pre-commit Secret Detection (10 min)
Update `.pre-commit-config.yaml` to include:
```yaml
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
```

Create `.secrets.baseline` file (run `detect-secrets scan > .secrets.baseline`)

### 6. Security Headers for Web Apps (10 min)
For each web app (repz, liveiticonic, attributa), ensure security headers in their configs:
- Content-Security-Policy
- X-Frame-Options: DENY
- X-Content-Type-Options: nosniff
- Referrer-Policy: strict-origin-when-cross-origin

### 7. Create Security Audit Script (5 min)
Create `tools/security/audit.ts`:
```typescript
#!/usr/bin/env tsx
// Security audit runner
// npm run security:audit
```
Add npm script: `"security:audit": "tsx tools/security/audit.ts"`

---

## Files to Create/Modify

| File | Action | Priority |
|------|--------|----------|
| `.github/dependabot.yml` | Create/Update | P0 |
| `.github/workflows/ci.yml` | Add security job | P0 |
| `.github/secret_scanning.yml` | Create | P1 |
| `.pre-commit-config.yaml` | Add detect-secrets | P1 |
| `.secrets.baseline` | Generate | P1 |
| `SECURITY.md` | Update if needed | P1 |
| `tools/security/audit.ts` | Create | P2 |
| `package.json` | Add security scripts | P2 |

---

## Success Criteria

- [ ] Dependabot enabled for npm, pip, github-actions
- [ ] Snyk scanning in CI pipeline
- [ ] Secret detection in pre-commit hooks
- [ ] SECURITY.md with proper disclosure process
- [ ] `npm run security:audit` works
- [ ] No high/critical vulnerabilities in current scan

---

## DO NOT

- Modify any LLC business logic
- Change authentication systems
- Rotate actual secrets (just add detection)
- Delete any existing security configs

## After Completion

Commit with: `feat(security): Complete Phase 6 security hardening`

Then report back for Phase 7 handoff.

