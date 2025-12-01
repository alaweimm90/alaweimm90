---
name: "Governance & Compliance Superprompt"
version: "1.0"
category: "project"
tags: ["governance", "compliance", "standards", "ethics", "access-control"]
created: "2024-11-30"
---

# Governance & Compliance Superprompt

## Purpose

Comprehensive framework for enforcing governance standards, compliance requirements, ethical AI practices, and access controls across all projects and repositories.

---

## System Prompt

```
You are a Governance and Compliance Architect with expertise in:
- Enterprise governance frameworks (COBIT, ITIL)
- Compliance standards (SOC2, HIPAA, GDPR, PCI-DSS, ISO 27001)
- Ethical AI principles and responsible AI practices
- Access control models (RBAC, ABAC, Zero Trust)
- Policy-as-code and automated enforcement
- Audit trails and compliance reporting

Your mission is to implement governance that:
1. Ensures regulatory compliance
2. Enforces ethical AI practices
3. Maintains security and access controls
4. Provides comprehensive audit trails
5. Enables automated policy enforcement
```

---

## Governance Framework

### Repository Governance Structure

```yaml
governance_hierarchy:
  organization_level:
    policies:
      - security-policy.yaml
      - data-handling-policy.yaml
      - access-control-policy.yaml
      - ai-ethics-policy.yaml
    enforcement: mandatory
    
  team_level:
    policies:
      - code-review-policy.yaml
      - deployment-policy.yaml
      - testing-standards.yaml
    enforcement: required
    
  project_level:
    policies:
      - project-specific-rules.yaml
      - custom-linting.yaml
    enforcement: recommended
```

### GOVERNANCE.md Template

```markdown
# Project Governance

## Overview

This document defines the governance standards, compliance requirements, and ethical guidelines for this project.

## Compliance Requirements

### Applicable Standards
- [ ] SOC2 Type II
- [ ] GDPR
- [ ] HIPAA
- [ ] PCI-DSS
- [ ] ISO 27001

### Data Classification
| Classification | Description | Handling Requirements |
|---------------|-------------|----------------------|
| Public | Non-sensitive data | Standard controls |
| Internal | Business data | Access logging |
| Confidential | Sensitive business data | Encryption required |
| Restricted | PII/PHI/Financial | Full compliance controls |

## Access Control

### Role Definitions
| Role | Permissions | Scope |
|------|-------------|-------|
| Owner | Full access | All resources |
| Admin | Manage settings | Project-wide |
| Developer | Read/Write code | Assigned repos |
| Viewer | Read-only | Assigned repos |

### Authentication Requirements
- Multi-factor authentication: Required
- SSO integration: Enabled
- Session timeout: 8 hours
- Password policy: 12+ chars, complexity required

## Code Review Requirements

### Mandatory Reviews
- All production code changes
- Security-sensitive modifications
- Infrastructure changes
- Database migrations

### Review Criteria
- [ ] Code quality standards met
- [ ] Security best practices followed
- [ ] Test coverage adequate
- [ ] Documentation updated

## Ethical AI Guidelines

### Principles
1. **Transparency**: AI decisions must be explainable
2. **Fairness**: No discriminatory outcomes
3. **Privacy**: Data minimization and protection
4. **Accountability**: Clear ownership of AI systems
5. **Safety**: Robust testing and monitoring

### Required Practices
- Bias testing for ML models
- Data provenance documentation
- Model versioning and audit trails
- Human oversight mechanisms

## Audit Requirements

### Logging
- All access attempts logged
- Data modifications tracked
- API calls recorded
- Authentication events captured

### Retention
- Security logs: 1 year
- Access logs: 90 days
- Audit trails: 7 years
- Compliance reports: Permanent

## Incident Response

### Severity Levels
| Level | Description | Response Time |
|-------|-------------|---------------|
| P1 | Critical security breach | 15 minutes |
| P2 | Major compliance violation | 1 hour |
| P3 | Minor policy violation | 24 hours |
| P4 | Informational | 1 week |

### Escalation Path
1. On-call engineer
2. Team lead
3. Security team
4. Compliance officer
5. Executive leadership

## Review Schedule

- Quarterly: Policy review
- Semi-annual: Access audit
- Annual: Full compliance assessment
```

---

## Policy-as-Code Implementation

### OPA Policies

```rego
# policies/repository-governance.rego
package governance.repository

# Require README.md
deny[msg] {
    not input.files["README.md"]
    msg := "Repository must have a README.md file"
}

# Require LICENSE
deny[msg] {
    not input.files["LICENSE"]
    msg := "Repository must have a LICENSE file"
}

# Require CODEOWNERS
deny[msg] {
    not input.files[".github/CODEOWNERS"]
    msg := "Repository must have CODEOWNERS defined"
}

# Require CI/CD workflow
deny[msg] {
    not input.files[".github/workflows/ci.yml"]
    msg := "Repository must have CI/CD workflow"
}

# Enforce branch protection
deny[msg] {
    branch := input.branches[_]
    branch.name == "main"
    not branch.protected
    msg := "Main branch must be protected"
}

# Require minimum reviewers
deny[msg] {
    branch := input.branches[_]
    branch.name == "main"
    branch.required_reviews < 2
    msg := "Main branch requires at least 2 reviewers"
}

# Enforce signed commits
deny[msg] {
    branch := input.branches[_]
    branch.name == "main"
    not branch.require_signed_commits
    msg := "Main branch must require signed commits"
}
```

### Security Policies

```rego
# policies/security-governance.rego
package governance.security

# No secrets in code
deny[msg] {
    file := input.files[_]
    contains(file.content, "password")
    contains(file.content, "=")
    not contains(file.path, ".env.example")
    msg := sprintf("Potential secret found in %s", [file.path])
}

# Require dependency scanning
deny[msg] {
    not input.security_scanning.dependency_check
    msg := "Dependency scanning must be enabled"
}

# Require SAST
deny[msg] {
    not input.security_scanning.sast
    msg := "Static Application Security Testing must be enabled"
}

# Enforce vulnerability remediation
deny[msg] {
    vuln := input.vulnerabilities[_]
    vuln.severity == "critical"
    vuln.age_days > 7
    msg := sprintf("Critical vulnerability %s unpatched for %d days", [vuln.id, vuln.age_days])
}

# Require security review for sensitive changes
deny[msg] {
    pr := input.pull_request
    contains(pr.files_changed[_], "auth/")
    not pr.reviewers[_].team == "security"
    msg := "Security team review required for auth changes"
}
```

### Data Governance Policies

```rego
# policies/data-governance.rego
package governance.data

# PII handling requirements
deny[msg] {
    data := input.data_fields[_]
    data.classification == "pii"
    not data.encrypted
    msg := sprintf("PII field %s must be encrypted", [data.name])
}

# Data retention compliance
deny[msg] {
    data := input.data_stores[_]
    data.contains_pii
    not data.retention_policy
    msg := sprintf("Data store %s with PII must have retention policy", [data.name])
}

# Cross-border data transfer
deny[msg] {
    transfer := input.data_transfers[_]
    transfer.destination_region != input.home_region
    not transfer.legal_basis
    msg := "Cross-border data transfer requires legal basis"
}

# Data minimization
warn[msg] {
    collection := input.data_collections[_]
    count(collection.fields) > 20
    msg := sprintf("Data collection %s may violate data minimization principle", [collection.name])
}
```

---

## Access Control Implementation

### RBAC Configuration

```yaml
# access-control/rbac.yaml
roles:
  organization_owner:
    description: "Full organization access"
    permissions:
      - org:*
      - repo:*
      - team:*
      - billing:*
    
  security_admin:
    description: "Security management"
    permissions:
      - security:*
      - audit:read
      - repo:security_settings
      
  team_lead:
    description: "Team management"
    permissions:
      - team:manage
      - repo:admin
      - pr:merge
      - deploy:staging
      
  developer:
    description: "Standard development"
    permissions:
      - repo:read
      - repo:write
      - pr:create
      - pr:review
      - issues:*
      
  contractor:
    description: "Limited external access"
    permissions:
      - repo:read
      - pr:create
      - issues:create
    restrictions:
      - no_admin_access
      - no_deploy_access
      - audit_all_actions

# Role assignments
assignments:
  - user: "@security-team"
    role: security_admin
    scope: organization
    
  - user: "@team-leads"
    role: team_lead
    scope: team
    
  - user: "@developers"
    role: developer
    scope: assigned_repos
```

### Zero Trust Configuration

```yaml
# access-control/zero-trust.yaml
zero_trust_policy:
  principles:
    - never_trust_always_verify
    - least_privilege_access
    - assume_breach
    - verify_explicitly
    
  authentication:
    mfa_required: true
    session_duration: 8h
    re_auth_for_sensitive: true
    device_trust_required: true
    
  authorization:
    continuous_verification: true
    context_aware: true
    just_in_time_access: true
    
  network:
    micro_segmentation: true
    encrypted_traffic: true
    no_implicit_trust: true
    
  monitoring:
    all_access_logged: true
    anomaly_detection: true
    real_time_alerts: true
```

---

## Ethical AI Framework

### AI Ethics Policy

```yaml
# policies/ai-ethics.yaml
ethical_ai_principles:
  transparency:
    requirements:
      - Model decisions must be explainable
      - Training data sources documented
      - Model limitations clearly stated
    implementation:
      - Use interpretable models where possible
      - Implement SHAP/LIME for explanations
      - Maintain model cards for all deployments
      
  fairness:
    requirements:
      - No discriminatory outcomes
      - Regular bias testing
      - Diverse training data
    implementation:
      - Fairness metrics in CI/CD
      - Protected attribute monitoring
      - Bias mitigation techniques
      
  privacy:
    requirements:
      - Data minimization
      - Consent management
      - Right to be forgotten
    implementation:
      - Differential privacy where applicable
      - Data anonymization pipelines
      - Consent tracking system
      
  accountability:
    requirements:
      - Clear ownership
      - Audit trails
      - Incident response
    implementation:
      - Model registry with owners
      - Decision logging
      - Escalation procedures
      
  safety:
    requirements:
      - Robust testing
      - Monitoring and alerts
      - Human oversight
    implementation:
      - Adversarial testing
      - Drift detection
      - Human-in-the-loop for high-stakes

model_governance:
  required_documentation:
    - model_card
    - data_sheet
    - risk_assessment
    - bias_report
    
  approval_gates:
    - data_quality_review
    - bias_assessment
    - security_review
    - ethics_review
    
  monitoring:
    - performance_metrics
    - fairness_metrics
    - drift_detection
    - usage_patterns
```

### Model Card Template

```markdown
# Model Card: [Model Name]

## Model Details
- **Developer**: [Team/Individual]
- **Model Date**: [Date]
- **Model Version**: [Version]
- **Model Type**: [Classification/Regression/etc.]
- **License**: [License]

## Intended Use
- **Primary Use Cases**: [Description]
- **Primary Users**: [Description]
- **Out-of-Scope Uses**: [Description]

## Training Data
- **Dataset**: [Name/Description]
- **Size**: [Number of samples]
- **Collection Method**: [Description]
- **Preprocessing**: [Description]

## Evaluation Data
- **Dataset**: [Name/Description]
- **Motivation**: [Why this dataset]

## Metrics
| Metric | Value |
|--------|-------|
| Accuracy | X% |
| Precision | X% |
| Recall | X% |
| F1 Score | X |

## Fairness Analysis
| Group | Metric | Value |
|-------|--------|-------|
| Group A | Accuracy | X% |
| Group B | Accuracy | X% |
| Disparity | | X% |

## Ethical Considerations
- **Potential Harms**: [Description]
- **Mitigations**: [Description]
- **Human Oversight**: [Description]

## Limitations
- [Limitation 1]
- [Limitation 2]

## Recommendations
- [Recommendation 1]
- [Recommendation 2]
```

---

## Compliance Automation

### Compliance Check Workflow

```yaml
# .github/workflows/compliance.yml
name: Compliance Check

on:
  push:
    branches: [main]
  pull_request:
  schedule:
    - cron: '0 0 * * 0'  # Weekly

jobs:
  governance-check:
    name: Governance Compliance
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run OPA policies
        uses: open-policy-agent/opa-action@v2
        with:
          policy: policies/
          input: governance-input.json
          
      - name: Check repository structure
        run: |
          ./scripts/check-structure.sh
          
      - name: Verify CODEOWNERS
        run: |
          ./scripts/verify-codeowners.sh
          
      - name: Generate compliance report
        run: |
          ./scripts/generate-compliance-report.sh
          
      - name: Upload report
        uses: actions/upload-artifact@v4
        with:
          name: compliance-report
          path: reports/compliance-report.json

  access-audit:
    name: Access Control Audit
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Audit repository access
        run: |
          gh api repos/${{ github.repository }}/collaborators \
            --jq '.[] | {login, permissions}' > access-audit.json
            
      - name: Check for stale access
        run: |
          ./scripts/check-stale-access.sh
          
      - name: Verify MFA compliance
        run: |
          ./scripts/verify-mfa.sh

  data-governance:
    name: Data Governance Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Scan for PII
        run: |
          ./scripts/scan-pii.sh
          
      - name: Check data retention
        run: |
          ./scripts/check-retention.sh
          
      - name: Verify encryption
        run: |
          ./scripts/verify-encryption.sh
```

---

## Audit Trail Implementation

```yaml
# audit/audit-config.yaml
audit_configuration:
  events_to_capture:
    authentication:
      - login_success
      - login_failure
      - logout
      - mfa_challenge
      - session_timeout
      
    authorization:
      - access_granted
      - access_denied
      - permission_change
      - role_assignment
      
    data_access:
      - read
      - write
      - delete
      - export
      
    administrative:
      - config_change
      - policy_update
      - user_management
      - system_change
      
  retention:
    security_events: 365 days
    access_logs: 90 days
    audit_trails: 7 years
    
  storage:
    primary: elasticsearch
    backup: s3
    encryption: AES-256
    
  alerting:
    critical_events:
      - multiple_failed_logins
      - privilege_escalation
      - data_exfiltration_attempt
      - policy_violation
```

---

## Execution Phases

### Phase 1: Foundation

- [ ] Create GOVERNANCE.md template
- [ ] Define role hierarchy
- [ ] Set up basic access controls
- [ ] Implement audit logging

### Phase 2: Policy Enforcement

- [ ] Write OPA policies
- [ ] Configure automated checks
- [ ] Set up compliance workflows
- [ ] Implement security scanning

### Phase 3: AI Ethics

- [ ] Define ethical AI principles
- [ ] Create model card templates
- [ ] Implement bias testing
- [ ] Set up monitoring

### Phase 4: Continuous Compliance

- [ ] Automate compliance reporting
- [ ] Set up drift detection
- [ ] Configure alerting
- [ ] Implement remediation workflows

---

*Last updated: 2024-11-30*
