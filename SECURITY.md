# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please report it via GitHub Security Advisories or email the repository maintainer.

## Security Measures

This repository implements:

1. **Automated Security Scanning** - OpenSSF Scorecard runs weekly
2. **Dependency Management** - Renovate automates updates with security checks
3. **Policy Enforcement** - OPA policies enforce security best practices
4. **Container Security** - Docker images scanned with Trivy

## OpenSSF Scorecard

This project is monitored by [OpenSSF Scorecard](https://securityscorecards.dev/) which performs 18 automated security checks:

- Branch Protection
- Code Review
- Dependency Update Tool
- Security Policy
- Vulnerability Disclosure
- And 13 more checks...

View the latest scorecard results in the GitHub Security tab.
