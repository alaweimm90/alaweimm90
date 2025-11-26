# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this governance system, please report it responsibly:

1. **Do NOT** create a public GitHub issue for security vulnerabilities
2. **Email** the repository maintainer directly or use GitHub's private vulnerability reporting
3. **Include** detailed information about the vulnerability and steps to reproduce

We will acknowledge receipt within 48 hours and provide a detailed response within 7 days.

## Security Measures

This governance repository implements multiple layers of security:

### Automated Security Scanning

| Tool | Purpose | Frequency |
|------|---------|-----------|
| **OpenSSF Scorecard** | 18 automated security checks | Weekly + on push |
| **Trivy** | Container and filesystem vulnerability scanning | On every CI run |
| **Dependabot** | Dependency vulnerability alerts | Continuous |
| **Renovate** | Automated dependency updates | Every 3 hours |
| **CodeQL** | Static analysis security testing | On PR/push |

### Policy Enforcement

- **OPA/Rego Policies**: Enforce security best practices in Dockerfiles, Kubernetes manifests, and repository structure
- **SLSA Level 3 Provenance**: Cryptographic attestation for governance artifacts
- **Branch Protection**: Required reviews, status checks, and linear history

### Workflow Security

All GitHub Actions workflows follow security best practices:

- Explicit minimal permissions (`permissions:` block)
- Pinned action versions (no `@latest` or `@main`)
- No secrets in logs or artifacts
- Isolated job environments

## OpenSSF Scorecard

This project is monitored by [OpenSSF Scorecard](https://securityscorecards.dev/) which performs 18 automated security checks:

- Binary-Artifacts
- Branch-Protection
- CI-Tests
- CII-Best-Practices
- Code-Review
- Contributors
- Dangerous-Workflow
- Dependency-Update-Tool
- Fuzzing
- License
- Maintained
- Packaging
- Pinned-Dependencies
- SAST
- Security-Policy
- Signed-Releases
- Token-Permissions
- Vulnerabilities

View the latest scorecard results in the [GitHub Security tab](../../security).

## Supported Versions

| Version | Supported |
|---------|-----------|
| main branch | Yes |
| Tagged releases | Yes |
| Other branches | No |

## Security Best Practices for Consumers

Repositories consuming this governance contract should:

1. **Use reusable workflows** from this repo for consistent security
2. **Enable branch protection** on main/master branches
3. **Require code reviews** before merging
4. **Run security scans** in CI pipelines
5. **Keep dependencies updated** using Renovate or Dependabot
6. **Follow Docker security policies** (non-root user, healthchecks, pinned versions)

## Contact

- **Security Issues**: Use GitHub's private vulnerability reporting
- **General Questions**: Open a GitHub issue
- **Maintainer**: [@alaweimm90](https://github.com/alaweimm90)
