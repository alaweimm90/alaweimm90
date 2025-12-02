---
name: 'Gating & Approval Workflows Superprompt'
version: '1.0'
category: 'project'
tags: ['gating', 'approval', 'code-review', 'security', 'compliance']
created: '2024-11-30'
---

# Gating & Approval Workflows Superprompt

## Purpose

Comprehensive framework for implementing quality gates, approval workflows, and security checkpoints to ensure code quality, compliance, and controlled releases.

---

## System Prompt

```
You are a Quality Assurance Architect and Release Manager with expertise in:
- Code review processes and best practices
- Security scanning and vulnerability management
- Compliance frameworks (SOC2, HIPAA, GDPR, PCI-DSS)
- Approval workflow automation
- Risk assessment and mitigation
- Change management processes

Your mission is to design gating mechanisms that:
1. Prevent defective code from reaching production
2. Enforce security and compliance standards
3. Enable efficient review processes
4. Maintain audit trails for all changes
5. Balance velocity with quality
```

---

## Gating Framework

### Gate Types

```yaml
gate_types:
  automated_gates:
    - lint_check
    - type_check
    - unit_tests
    - coverage_threshold
    - security_scan
    - dependency_audit
    - license_compliance
    - performance_benchmarks

  manual_gates:
    - code_review
    - architecture_review
    - security_review
    - compliance_approval
    - release_approval
    - change_advisory_board

  conditional_gates:
    - breaking_change_review
    - database_migration_review
    - api_contract_review
    - infrastructure_change_review
```

---

## Code Review Process

### Pull Request Template

```markdown
<!-- .github/PULL_REQUEST_TEMPLATE.md -->

## Description

<!-- Describe your changes in detail -->

## Type of Change

- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update
- [ ] Refactoring (no functional changes)
- [ ] Performance improvement
- [ ] Security fix

## Related Issues

<!-- Link to related issues: Fixes #123, Relates to #456 -->

## Testing

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] E2E tests added/updated
- [ ] Manual testing performed

## Security Checklist

- [ ] No secrets or credentials in code
- [ ] Input validation implemented
- [ ] Authentication/authorization checked
- [ ] SQL injection prevention verified
- [ ] XSS prevention verified

## Documentation

- [ ] README updated (if needed)
- [ ] API documentation updated (if needed)
- [ ] Changelog entry added

## Deployment Notes

<!-- Any special deployment considerations -->

## Screenshots

<!-- If applicable, add screenshots -->
```

### CODEOWNERS Configuration

```
# .github/CODEOWNERS

# Default owners for everything
* @team-leads

# Frontend
/src/components/ @frontend-team
/src/pages/ @frontend-team
/src/styles/ @frontend-team @design-team

# Backend
/src/api/ @backend-team
/src/services/ @backend-team
/src/database/ @backend-team @dba-team

# Infrastructure
/infrastructure/ @devops-team @security-team
/k8s/ @devops-team
/.github/workflows/ @devops-team

# Security-sensitive files
/src/auth/ @security-team @backend-team
/src/crypto/ @security-team
*.pem @security-team
*.key @security-team

# Configuration
/config/ @team-leads
*.config.* @team-leads

# Documentation
/docs/ @tech-writers @team-leads
README.md @team-leads
```

### Branch Protection Rules

```yaml
# Branch protection configuration
branch_protection:
  main:
    required_reviews: 2
    dismiss_stale_reviews: true
    require_code_owner_reviews: true
    required_status_checks:
      - ci/lint
      - ci/test
      - ci/security-scan
      - ci/build
    require_branches_up_to_date: true
    restrict_pushes: true
    allow_force_pushes: false
    allow_deletions: false
    require_signed_commits: true
    require_linear_history: true

  develop:
    required_reviews: 1
    dismiss_stale_reviews: true
    required_status_checks:
      - ci/lint
      - ci/test
    require_branches_up_to_date: true

  release/*:
    required_reviews: 2
    require_code_owner_reviews: true
    required_status_checks:
      - ci/lint
      - ci/test
      - ci/security-scan
      - ci/e2e
    restrict_pushes: true
```

---

## Security Gates

### Security Scanning Pipeline

```yaml
# .github/workflows/security-gates.yml
name: Security Gates

on:
  pull_request:
    branches: [main, develop]

jobs:
  # ============================================
  # SAST - Static Application Security Testing
  # ============================================
  sast:
    name: SAST Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Initialize CodeQL
        uses: github/codeql-action/init@v2
        with:
          languages: javascript, typescript

      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v2

      - name: Semgrep Scan
        uses: returntocorp/semgrep-action@v1
        with:
          config: >-
            p/security-audit
            p/secrets
            p/owasp-top-ten

  # ============================================
  # Dependency Scanning
  # ============================================
  dependency-scan:
    name: Dependency Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Snyk Security Scan
        uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          args: --severity-threshold=high

      - name: OWASP Dependency Check
        uses: dependency-check/Dependency-Check_Action@main
        with:
          project: '${{ github.repository }}'
          path: '.'
          format: 'HTML'

      - name: License Compliance
        run: |
          npx license-checker --production --failOn "GPL;AGPL"

  # ============================================
  # Container Scanning
  # ============================================
  container-scan:
    name: Container Scan
    runs-on: ubuntu-latest
    if: contains(github.event.pull_request.changed_files, 'Dockerfile')
    steps:
      - uses: actions/checkout@v4

      - name: Build image
        run: docker build -t test-image .

      - name: Trivy Container Scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'test-image'
          severity: 'CRITICAL,HIGH'
          exit-code: '1'

      - name: Dockle Lint
        uses: erzz/dockle-action@v1
        with:
          image: 'test-image'
          failure-threshold: high

  # ============================================
  # Secret Detection
  # ============================================
  secret-scan:
    name: Secret Detection
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Gitleaks Scan
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: TruffleHog Scan
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: ${{ github.event.pull_request.base.sha }}
          head: ${{ github.event.pull_request.head.sha }}

  # ============================================
  # Security Gate Decision
  # ============================================
  security-gate:
    name: Security Gate
    needs: [sast, dependency-scan, secret-scan]
    runs-on: ubuntu-latest
    steps:
      - name: Security Gate Check
        run: |
          echo "All security scans passed"
          echo "✅ SAST: Passed"
          echo "✅ Dependencies: Passed"
          echo "✅ Secrets: Passed"
```

---

## Approval Workflows

### Multi-Stage Approval

```yaml
# .github/workflows/approval-workflow.yml
name: Approval Workflow

on:
  pull_request:
    types: [labeled]

jobs:
  # ============================================
  # Architecture Review Gate
  # ============================================
  architecture-review:
    name: Architecture Review
    if: contains(github.event.pull_request.labels.*.name, 'architecture-change')
    runs-on: ubuntu-latest
    environment:
      name: architecture-review
    steps:
      - name: Request Architecture Review
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.pulls.requestReviewers({
              owner: context.repo.owner,
              repo: context.repo.repo,
              pull_number: context.issue.number,
              team_reviewers: ['architecture-team']
            });

      - name: Wait for Approval
        run: echo "Waiting for architecture team approval..."

  # ============================================
  # Security Review Gate
  # ============================================
  security-review:
    name: Security Review
    if: contains(github.event.pull_request.labels.*.name, 'security-sensitive')
    runs-on: ubuntu-latest
    environment:
      name: security-review
    steps:
      - name: Request Security Review
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.pulls.requestReviewers({
              owner: context.repo.owner,
              repo: context.repo.repo,
              pull_number: context.issue.number,
              team_reviewers: ['security-team']
            });

  # ============================================
  # Database Migration Gate
  # ============================================
  database-review:
    name: Database Migration Review
    if: contains(github.event.pull_request.labels.*.name, 'database-migration')
    runs-on: ubuntu-latest
    environment:
      name: database-review
    steps:
      - uses: actions/checkout@v4

      - name: Analyze Migration
        run: |
          # Check for destructive operations
          if grep -r "DROP TABLE\|DROP COLUMN\|TRUNCATE" migrations/; then
            echo "::warning::Destructive migration detected!"
            echo "DESTRUCTIVE=true" >> $GITHUB_ENV
          fi

      - name: Request DBA Review
        if: env.DESTRUCTIVE == 'true'
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.pulls.requestReviewers({
              owner: context.repo.owner,
              repo: context.repo.repo,
              pull_number: context.issue.number,
              team_reviewers: ['dba-team']
            });

  # ============================================
  # Release Approval Gate
  # ============================================
  release-approval:
    name: Release Approval
    if: startsWith(github.head_ref, 'release/')
    runs-on: ubuntu-latest
    environment:
      name: release-approval
      url: https://github.com/${{ github.repository }}/releases
    steps:
      - name: Generate Release Notes
        run: |
          echo "## Release Summary" > release-notes.md
          echo "- Changes in this release..." >> release-notes.md

      - name: Request Release Approval
        uses: trstringer/manual-approval@v1
        with:
          secret: ${{ secrets.GITHUB_TOKEN }}
          approvers: release-managers
          minimum-approvals: 2
          issue-title: 'Release Approval Required'
```

---

## Compliance Gates

### SOC2 Compliance Checks

```yaml
compliance_gates:
  soc2:
    access_control:
      - Verify RBAC implementation
      - Check authentication mechanisms
      - Audit authorization rules

    change_management:
      - Require PR for all changes
      - Enforce code review
      - Maintain audit logs

    risk_assessment:
      - Security scan results
      - Vulnerability assessment
      - Penetration test status

    monitoring:
      - Logging enabled
      - Alerting configured
      - Incident response plan

  gdpr:
    data_protection:
      - PII handling review
      - Data encryption verification
      - Consent mechanism check

    data_subject_rights:
      - Export functionality
      - Deletion capability
      - Access request handling

  hipaa:
    phi_protection:
      - Encryption at rest
      - Encryption in transit
      - Access logging

    audit_controls:
      - Activity logging
      - Access reviews
      - Breach notification
```

### Compliance Workflow

```yaml
# .github/workflows/compliance.yml
name: Compliance Gates

on:
  pull_request:
    branches: [main]
    paths:
      - 'src/auth/**'
      - 'src/data/**'
      - 'src/api/**'

jobs:
  compliance-check:
    name: Compliance Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: PII Detection
        run: |
          # Scan for potential PII exposure
          grep -rn "ssn\|social.*security\|credit.*card" src/ && exit 1 || true

      - name: Encryption Verification
        run: |
          # Verify sensitive data is encrypted
          ./scripts/verify-encryption.sh

      - name: Access Control Audit
        run: |
          # Check RBAC implementation
          ./scripts/audit-access-control.sh

      - name: Generate Compliance Report
        run: |
          echo "# Compliance Report" > compliance-report.md
          echo "Date: $(date)" >> compliance-report.md
          echo "## Checks Performed" >> compliance-report.md
          echo "- [x] PII Detection" >> compliance-report.md
          echo "- [x] Encryption Verification" >> compliance-report.md
          echo "- [x] Access Control Audit" >> compliance-report.md

      - name: Upload Compliance Report
        uses: actions/upload-artifact@v4
        with:
          name: compliance-report
          path: compliance-report.md
```

---

## Quality Gate Configuration

```yaml
# quality-gates.yaml
quality_gates:
  code_quality:
    coverage:
      minimum: 80%
      blocking: true
    complexity:
      max_cyclomatic: 10
      blocking: true
    duplication:
      max_percentage: 3%
      blocking: false
    code_smells:
      max_count: 0
      severity: critical
      blocking: true

  security:
    vulnerabilities:
      critical: 0
      high: 0
      medium: 5
      blocking: true
    secrets:
      count: 0
      blocking: true
    sast_findings:
      critical: 0
      high: 0
      blocking: true

  performance:
    build_time:
      max_minutes: 10
      blocking: false
    test_time:
      max_minutes: 15
      blocking: false
    bundle_size:
      max_kb: 500
      blocking: true

  documentation:
    api_coverage:
      minimum: 90%
      blocking: false
    readme_present:
      required: true
      blocking: true
```

---

## Execution Phases

### Phase 1: Foundation

- [ ] Configure branch protection rules
- [ ] Set up CODEOWNERS
- [ ] Create PR templates
- [ ] Implement basic CI gates

### Phase 2: Security Gates

- [ ] Add SAST scanning
- [ ] Implement dependency scanning
- [ ] Configure secret detection
- [ ] Set up container scanning

### Phase 3: Approval Workflows

- [ ] Configure environment approvals
- [ ] Set up team-based reviews
- [ ] Implement conditional gates
- [ ] Add release approval process

### Phase 4: Compliance

- [ ] Implement compliance checks
- [ ] Generate audit reports
- [ ] Configure policy enforcement
- [ ] Set up continuous monitoring

---

_Last updated: 2024-11-30_
