# Security Policy

## Supported Versions

We actively support and provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly:

### How to Report

1. **DO NOT** create a public GitHub issue for security vulnerabilities
2. Email security details to: security@example.com (replace with your email)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity
  - Critical: 1-7 days
  - High: 7-30 days
  - Medium: 30-90 days
  - Low: Next planned release

### Security Measures

This repository implements multiple security layers:

#### Automated Security Scanning

- **OpenSSF Scorecard**: Weekly automated security health checks (18 checks)
- **Renovate**: Automated dependency updates with vulnerability detection
- **Trivy**: Container image vulnerability scanning
- **Hadolint**: Dockerfile security best practices
- **Pre-commit hooks**: Code quality and security checks

#### Security Best Practices

- All Docker containers run as non-root users
- Multi-stage builds to minimize attack surface
- Health checks on all services
- Restart policies for resilience
- Dependency pinning in CI/CD
- SARIF results uploaded to GitHub Security tab

#### OpenSSF Scorecard Checks

We monitor and maintain high scores across all 18 OpenSSF security checks:

1. **Binary-Artifacts**: No unnecessary binaries in repository
2. **Branch-Protection**: Protected branches with required reviews
3. **CI-Tests**: Comprehensive CI/CD testing
4. **CII-Best-Practices**: Following industry best practices
5. **Code-Review**: Required pull request reviews
6. **Contributors**: Multiple active contributors
7. **Dangerous-Workflow**: No dangerous GitHub Actions workflows
8. **Dependency-Update-Tool**: Renovate for automated updates
9. **Fuzzing**: Fuzz testing for critical components
10. **License**: Clear licensing
11. **Maintained**: Active maintenance and updates
12. **Packaging**: Secure package publishing
13. **Pinned-Dependencies**: Pinned dependencies in workflows
14. **SAST**: Static analysis security testing
15. **Security-Policy**: This document
16. **Signed-Releases**: GPG-signed releases
17. **Token-Permissions**: Minimal GitHub Actions permissions
18. **Vulnerabilities**: Proactive vulnerability management

### Disclosure Policy

- Confirmed vulnerabilities will be disclosed publicly after a fix is released
- Credit will be given to security researchers who responsibly disclose issues
- CVE IDs will be requested for significant vulnerabilities

### Security Updates

Security updates are released as:
- Patch versions for backward-compatible fixes
- Clearly marked in release notes with `[SECURITY]` tag
- Announced via GitHub Security Advisories

## Security Champions

- Platform Team: platform@example.com
- Security Lead: security@example.com

## Additional Resources

- [OpenSSF Scorecard Results](https://github.com/mesha/GitHub/security/code-scanning)
- [Dependency Dashboard](https://github.com/mesha/GitHub/issues/renovate)
- [Security Advisories](https://github.com/mesha/GitHub/security/advisories)

---

Generated with [Claude Code](https://claude.com/claude-code)

Last Updated: November 24, 2025
