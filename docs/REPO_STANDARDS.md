# Repo Standards
## 1. Repository Structure Standards
### 1.1 Global Conventions (All Repositories)
```
repo-root/
├── README.md                    # Project overview and setup
├── LICENSE                      # License information
├── .gitignore                   # Git ignore patterns
├── .editorconfig               # Code style consistency
├── .gitattributes              # Git attributes for file handling
├── .env.example                # Environment variables template
├── docs/                       # User and developer documentation
├── src/                        # All first-party code
├── tests/                      # Test suites (mirror src/ structure)
├── scripts/                    # Automation and maintenance scripts
├── tools/                      # Internal CLIs and development tools
├── infra/                      # Infrastructure as Code (Docker, K8s, Terraform)
└── .github/                    # CI/CD workflows and GitHub configuration
    └── workflows/              # GitHub Actions pipelines
```
### 1.2 Archetype-Specific Structures
#### A. Research Platform Monorepo (AlaweinOS pattern)
**Use when:** Multiple research projects sharing optimization/ML infrastructure
```
repo-root/
├── <project-name>/             # Each major project (optilibria, MEZAN, TalAI)
│   ├── src/<package>/          # Python package structure
│   ├── tests/                  # Unit, integration, benchmarks
│   ├── docs/                   # Research documentation
│   ├── examples/               # Usage examples
│   ├── benchmarks/             # Performance benchmarks
│   ├── papers/                 # LaTeX papers (optional)
│   ├── pyproject.toml          # Python configuration
│   ├── README.md               # Project documentation
│   └── CLAUDE.md               # AI context (500+ lines for gold standard)
├── docs/                       # Cross-project documentation
├── scripts/                    # Shared automation
└── benchmarks/                 # Cross-project benchmarks
```
#### B. Educational Platform Monorepo (MeatheadPhysicist pattern)
**Use when:** Library + web app + educational content
```
repo-root/
├── src/                        # Core Python library
├── frontend/                   # React/Next web application
├── visualizations/             # 3D visualization library
├── cli/                        # Command-line tools
├── projects/                   # Research/demo projects
├── notebooks/                  # Jupyter notebooks
├── papers/                     # Academic papers
├── education/                  # Tutorials and courses
└── examples/                   # Usage examples
```
#### C. Product Monorepo (alaweimm90-business pattern)
**Use when:** Multiple products sharing infrastructure
```
repo-root/
├── <product-name>/             # Each product (live-it-iconic, repz)
│   ├── .brand/                 # Brand assets and playbooks
│   ├── docs/                   # Product documentation
│   ├── infrastructure/         # Docker, Kubernetes, Terraform
│   ├── src/                    # Application code
│   ├── supabase/               # Database migrations
│   ├── tests/                  # E2E and integration tests
│   └── package.json/pyproject.toml
├── shared/                     # Shared packages and utilities
└── infrastructure/             # Shared infrastructure
```
#### D. Toolkit Monorepo (alaweimm90-tools pattern)
**Use when:** Collection of independent tools/CLIs
```
repo-root/
├── <tool-name>/                # Each tool (fitness-app, cli, sdk)
│   ├── src/                    # Source code
│   ├── tests/                  # Test suites
│   ├── README.md               # Tool documentation
│   └── pyproject.toml/package.json
└── shared/                     # Shared utilities (optional)
```
### 1.3 File Lifecycle & Storage Tiers
- **`docs/` – Active documentation**
  - Current, high-value documentation needed for day-to-day work.
  - No raw tool output, throwaway audit logs, or "mission complete" markers.
- **`.archive/` – Long-term, low-touch storage**
  - Historical reports, completion certificates, and large remediation narratives.
  - Not auto-deleted; cleanup is manual and infrequent.
  - Recommended subfolders: `.archive/reports/`, `.archive/completion/`, `.archive/security/`, `.archive/compliance/`, `.archive/performance/`.
- **`.tmp/` – Ephemeral working area**
  - Scratch artifacts, raw tool outputs, and short-lived reports.
  - Files classified as `temp` are quarantined by `scripts/files/cleanup-daily.js` once older than `thresholds.temp_grace_hours` (24–48 hours).
  - Nothing in `.tmp/` should be treated as a system of record.
- **Root hygiene requirement**
  - Non-standard or ad-hoc files must be placed in `docs/`, `.archive/`, or `.tmp/` – not left accumulating in the repository root.
## 2. Code Quality Standards
### 2.1 General Principles
- **SOLID Principles:** Mandatory for all object-oriented design
- **DRY Principle:** Maximum 5% code duplication tolerance
- **KISS Principle:** Favor simplicity over cleverness
- **YAGNI Principle:** No premature optimization
- **Separation of Concerns:** Clear architectural boundaries
### 2.2 Code Metrics Compliance
- **Cyclomatic Complexity:** Maximum 10 per function
- **Cognitive Complexity:** Maximum 15 per function
- **Function Length:** Maximum 50 lines (excluding comments)
- **Class Length:** Maximum 300 lines
- **Parameter Count:** Maximum 5 parameters per function
- **Code Coverage:** Minimum 80% for unit tests, 70% for integration
### 2.3 Naming Conventions
- **Files:** PascalCase for components, camelCase for utilities
- **Functions:** camelCase, descriptive names
- **Constants:** SCREAMING_SNAKE_CASE
- **Classes:** PascalCase
- **Interfaces:** PascalCase with 'I' prefix (TypeScript)
## 3. Documentation Standards
### 3.1 README.md Structure
```markdown
# Project Name
[Badges: Build Status, Coverage, Version, License]
## Overview
Brief description, purpose, value proposition
## Features
Bullet list of key features
## Architecture
Link to architecture documentation
## Prerequisites
- Required software and versions
- Hardware requirements
- Access requirements
## Installation
Step-by-step setup instructions
## Configuration
Environment variables and configuration options
## Usage
Basic usage examples
## API Documentation
Link to API docs
## Testing
How to run tests
## Deployment
Deployment procedures
## Contributing
Contribution guidelines
## License
License information
## Support
Contact information, issue tracking
```
### 3.2 CLAUDE.md Requirements
- **Context Provision:** 500+ lines for AI-assisted development
- **Architecture Overview:** System design and component relationships
- **Development Guidelines:** Coding standards and patterns
- **Troubleshooting:** Common issues and solutions
- **Future Plans:** Roadmap and planned features
## 4. Testing Standards
### 4.1 Test Pyramid
- **Unit Tests (70%)**: Fast, isolated, no external dependencies
- **Integration Tests (20%)**: Component interaction testing
- **End-to-End Tests (10%)**: Full user workflow testing
### 4.2 Test Naming Convention
```
[MethodName]_[Scenario]_[ExpectedBehavior]
Examples:
- CreateUser_WithValidData_ReturnsCreatedUser
- AuthenticateUser_WithInvalidPassword_ThrowsAuthenticationException
```
## 5. CI/CD Standards
### 5.1 Pipeline Requirements
1. **Checkout:** Clone repository
2. **Dependencies:** Install and cache dependencies
3. **Lint:** Code style checking
4. **Build:** Compile and bundle application
5. **Unit Tests:** Run unit test suite
6. **Integration Tests:** Run integration tests
7. **SAST:** Static Application Security Testing
8. **SCA:** Software Composition Analysis
9. **Coverage:** Generate and validate coverage reports
10. **Artifacts:** Create deployable artifacts
11. **Container Scan:** Scan Docker images
12. **Publish:** Push artifacts to registries
### 5.2 Quality Gates
- Build must succeed
- All tests must pass
- Code coverage ≥ 80%
- No critical/high security vulnerabilities
- No license compliance issues
## 6. Security Standards
### 6.1 Authentication & Authorization
- OAuth 2.0 / OpenID Connect implementation
- JWT with RS256 or ES256 (never HS256 in production)
- Multi-factor authentication (MFA) required
- Role-Based Access Control (RBAC) or Attribute-Based Access Control (ABAC)
### 6.2 Data Protection
- Encryption at rest: AES-256-GCM minimum
- Encryption in transit: TLS 1.3 only
- Secure key management via HSM or KMS
- PII/PHI data classification and handling
## 7. Performance Standards
### 7.1 Response Time SLAs
- API endpoints: p50 < 200ms, p95 < 500ms, p99 < 1000ms
- Database queries: p95 < 100ms
- Page load time: First Contentful Paint < 1.5s
### 7.2 Scalability Requirements
- Horizontal scaling capability required
- Stateless application design
- Database connection pooling
- CDN integration for static assets
## 8. Compliance Standards
### 8.1 Regulatory Compliance
- **GDPR:** Data protection and privacy rights
- **PCI-DSS:** Payment card industry security
- **SOC 2:** Service organization controls
- **HIPAA:** Health information privacy (if applicable)
### 8.2 Accessibility Standards
- **WCAG 2.1 Level AA:** Minimum accessibility compliance
- **Section 508:** Government accessibility requirements
## 9. Version Control Standards
### 9.1 Branch Strategy (GitFlow)
- **main/master:** Production-ready code only
- **develop:** Integration branch for features
- **feature/[ticket]-description:** Feature development
- **bugfix/[ticket]-description:** Bug fixes
- **hotfix/[ticket]-description:** Emergency production fixes
- **release/version:** Release preparation
### 9.2 Commit Standards (Conventional Commits)
```
<type>(<scope>): <subject>
<body>
<footer>
```
**Types:** feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert
## 10. Maintenance and Governance
### 10.1 Regular Audits
- **Monthly:** Code quality and coverage review
- **Quarterly:** Security assessment and dependency updates
- **Annually:** Architecture review and performance benchmarking
### 10.2 Documentation Maintenance
- **Automated Hygiene:** Regular scanning for redundancies
- **Version Control:** All documentation changes tracked
- **Cross-References:** Validated links and references
### 10.3 Compliance Monitoring
- **Automated Checks:** CI/CD pipeline compliance validation
- **Manual Reviews:** Quarterly compliance audits
- **Issue Tracking:** All non-compliance issues documented and tracked
---
_This document serves as the comprehensive standard for all repositories. Regular updates and compliance checks ensure ongoing adherence to these standards._</content>
</edit_file>
