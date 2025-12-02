# Security Policy

<img src="https://img.shields.io/badge/OpenSSF-Scorecard-A855F7?style=flat-square&labelColor=1a1b27" alt="OpenSSF"/>
<img src="https://img.shields.io/badge/SLSA-Level_3-EC4899?style=flat-square&labelColor=1a1b27" alt="SLSA"/>
<img src="https://img.shields.io/badge/Trivy-Enabled-4CC9F0?style=flat-square&labelColor=1a1b27" alt="Trivy"/>

---

## Reporting a Vulnerability

| Action       | Details                                                         |
| ------------ | --------------------------------------------------------------- |
| **Do NOT**   | Create public GitHub issues for security vulnerabilities        |
| **Do**       | Use GitHub's private vulnerability reporting                    |
| **Include**  | Detailed reproduction steps                                     |
| **Response** | Acknowledgment within 48 hours, detailed response within 7 days |

---

## Security Measures

### Automated Scanning

| Tool                  | Purpose                                     | Frequency        |
| --------------------- | ------------------------------------------- | ---------------- |
| **OpenSSF Scorecard** | 18 automated security checks                | Weekly + on push |
| **Trivy**             | Container/filesystem vulnerability scanning | Every CI run     |
| **Dependabot**        | Dependency vulnerability alerts             | Continuous       |
| **Renovate**          | Automated dependency updates                | Every 3 hours    |
| **CodeQL**            | Static analysis security testing            | On PR/push       |

### Policy Enforcement

- **OPA/Rego Policies**: Security best practices for Dockerfiles, K8s manifests, repo structure
- **SLSA Level 3**: Cryptographic attestation for governance artifacts
- **Branch Protection**: Required reviews, status checks, linear history

### Workflow Security

- Explicit minimal permissions (`permissions:` block)
- Pinned action versions (no `@latest`)
- No secrets in logs or artifacts
- Isolated job environments

---

## OpenSSF Scorecard Checks

| Check                  | Description            |
| ---------------------- | ---------------------- |
| Binary-Artifacts       | No checked-in binaries |
| Branch-Protection      | Protected branches     |
| CI-Tests               | Tests run in CI        |
| Code-Review            | PRs reviewed           |
| Dangerous-Workflow     | No dangerous patterns  |
| Dependency-Update-Tool | Automated updates      |
| Pinned-Dependencies    | Actions pinned         |
| SAST                   | Static analysis        |
| Security-Policy        | This file exists       |
| Token-Permissions      | Minimal permissions    |
| Vulnerabilities        | No known CVEs          |

---

## Supported Versions

| Version         | Supported |
| --------------- | --------- |
| `main` branch   | Yes       |
| Tagged releases | Yes       |
| Other branches  | No        |

---

## Best Practices for Consumers

1. **Use reusable workflows** from this repo
2. **Enable branch protection** on main/master
3. **Require code reviews** before merging
4. **Run security scans** in CI pipelines
5. **Keep dependencies updated** via Renovate/Dependabot
6. **Follow Docker security policies** (non-root user, healthchecks, pinned versions)

---

## Contact

| Channel                                      | Use For           |
| -------------------------------------------- | ----------------- |
| GitHub Private Reporting                     | Security issues   |
| GitHub Issues                                | General questions |
| [@alaweimm90](https://github.com/alaweimm90) | Maintainer        |
