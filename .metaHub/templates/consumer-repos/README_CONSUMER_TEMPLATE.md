# {{REPO_NAME}}

{{REPO_SHORT_DESCRIPTION}}

<div align="center">

[![Status](https://img.shields.io/badge/Status-{{STATUS}}-{{STATUS_COLOR}}?style=flat-square)]({{STATUS_LINK}})
[![Version](https://img.shields.io/badge/Version-{{VERSION}}-blue?style=flat-square)](./CHANGELOG.md)
[![License](https://img.shields.io/badge/License-{{LICENSE}}-green?style=flat-square)](./LICENSE)
[![Governance](https://img.shields.io/badge/Governance-{{GOVERNANCE_LEVEL}}-purple?style=flat-square)](./GOVERNANCE.md)

</div>

---

## Overview

**{{REPO_LONG_DESCRIPTION}}**

### Key Features

- {{FEATURE_1}}
- {{FEATURE_2}}
- {{FEATURE_3}}
- {{FEATURE_4}}

### Technology Stack

**Language:** {{PRIMARY_LANGUAGE}}
**Framework:** {{FRAMEWORK}}
**Runtime:** {{RUNTIME}}

---

## Quick Start

### Prerequisites

- {{PREREQUISITE_1}}
- {{PREREQUISITE_2}}
- {{PREREQUISITE_3}}

### Installation

```bash
# Clone the repository
git clone {{REPO_URL}}
cd {{REPO_NAME}}

# Install dependencies
{{INSTALL_COMMAND}}

# Run tests
{{TEST_COMMAND}}

# Start development
{{DEV_COMMAND}}
```

### Basic Usage

```{{PRIMARY_LANGUAGE}}
{{USAGE_EXAMPLE}}
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| [README](./README.md) | This file — project overview |
| [CONTRIBUTING](./CONTRIBUTING.md) | How to contribute |
| [ARCHITECTURE](./docs/ARCHITECTURE.md) | System design |
| [API Reference](./docs/API.md) | API documentation |
| [Deployment](./docs/DEPLOYMENT.md) | How to deploy |

---

## Repository Structure

```
{{REPO_NAME}}/
├── src/                    # Source code
│   ├── {{MODULE_1}}/       # {{MODULE_1_DESCRIPTION}}
│   ├── {{MODULE_2}}/       # {{MODULE_2_DESCRIPTION}}
│   └── {{MODULE_3}}/       # {{MODULE_3_DESCRIPTION}}
├── tests/                  # Test suite
├── docs/                   # Documentation
├── .github/                # GitHub Actions workflows
├── .meta/                  # Governance metadata
│   └── repo.yaml           # Repository metadata (governance contract)
├── Dockerfile              # Container definition
├── docker-compose.yml      # Local development environment
├── {{CONFIG_FILE_1}}       # {{CONFIG_FILE_1_PURPOSE}}
├── {{CONFIG_FILE_2}}       # {{CONFIG_FILE_2_PURPOSE}}
├── README.md               # This file
├── LICENSE                 # License
└── CONTRIBUTING.md         # Contribution guidelines
```

---

## Governance Metadata

This repository implements the [governance contract]({{GOVERNANCE_CONTRACT_URL}}).

**Metadata File:** [`.meta/repo.yaml`](./.meta/repo.yaml)

```yaml
type: {{REPO_TYPE}}              # lib, tool, core, research, demo, workspace
language: {{REPO_LANGUAGE}}      # python, typescript, mixed
tier: {{REPO_TIER}}              # 1=mission-critical, 2=important, 3=experimental
coverage: {{COVERAGE_THRESHOLD}} # Test coverage minimum
docs: {{DOCS_PROFILE}}           # standard, minimal
owner: {{TEAM_NAME}}             # Team responsible
```

### Compliance Status

| Check | Status | Details |
|-------|--------|---------|
| Structure | ✅ Pass | Repository follows governance standards |
| Metadata | ✅ Pass | `.meta/repo.yaml` completes |
| Security | ✅ Pass | {{SECURITY_CHECK_DETAILS}} |
| Tests | ✅ Pass | {{TEST_COVERAGE}}% coverage |
| Documentation | ✅ Pass | {{DOC_STATUS}} |

---

## Development

### Setup Development Environment

```bash
{{DEV_SETUP_COMMAND}}
```

### Running Tests

```bash
# Run all tests
{{TEST_ALL_COMMAND}}

# Run specific test file
{{TEST_SPECIFIC_COMMAND}}

# Run with coverage
{{TEST_COVERAGE_COMMAND}}
```

### Code Quality

```bash
# Run linter
{{LINT_COMMAND}}

# Format code
{{FORMAT_COMMAND}}

# Type checking
{{TYPE_CHECK_COMMAND}}
```

### Building

```bash
# Build for development
{{BUILD_DEV_COMMAND}}

# Build for production
{{BUILD_PROD_COMMAND}}

# Build Docker image
{{BUILD_DOCKER_COMMAND}}
```

---

## Deployment

### Prerequisites

- {{DEPLOY_PREREQ_1}}
- {{DEPLOY_PREREQ_2}}

### Environment Configuration

See [Deployment Guide](./docs/DEPLOYMENT.md) for full instructions.

### Production Deployment

```bash
{{DEPLOY_COMMAND}}
```

### Monitoring

- **Logs:** {{LOG_LOCATION}}
- **Metrics:** {{METRICS_URL}}
- **Alerts:** {{ALERT_CONFIG}}

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

### Code Standards

- **Language:** {{CODE_STANDARD_LANG}}
- **Linter:** {{LINTER_NAME}}
- **Formatter:** {{FORMATTER_NAME}}
- **Coverage:** {{MIN_COVERAGE}}%

### Pull Request Process

1. Create feature branch: `git checkout -b feature/{{FEATURE_NAME}}`
2. Make changes and commit: `git commit -m "{{COMMIT_MESSAGE_FORMAT}}"`
3. Push to fork: `git push origin feature/{{FEATURE_NAME}}`
4. Submit PR with description
5. Pass all CI checks
6. Require {{APPROVAL_COUNT}} approval(s)
7. Merge when ready

### Code Review

- Automated checks (linter, tests, coverage)
- Manual review by {{CODE_REVIEWER_TEAM}}
- {{ADDITIONAL_REVIEW_NOTES}}

---

## Performance

### Benchmarks

| Operation | Baseline | Status |
|-----------|----------|--------|
| {{BENCHMARK_1}} | {{BENCHMARK_1_BASELINE}} | {{BENCHMARK_1_STATUS}} |
| {{BENCHMARK_2}} | {{BENCHMARK_2_BASELINE}} | {{BENCHMARK_2_STATUS}} |
| {{BENCHMARK_3}} | {{BENCHMARK_3_BASELINE}} | {{BENCHMARK_3_STATUS}} |

See [Performance Guide](./docs/PERFORMANCE.md) for details.

---

## Known Issues

| Issue | Workaround | Status |
|-------|-----------|--------|
| {{ISSUE_1}} | {{ISSUE_1_WORKAROUND}} | {{ISSUE_1_STATUS}} |
| {{ISSUE_2}} | {{ISSUE_2_WORKAROUND}} | {{ISSUE_2_STATUS}} |

---

## Roadmap

### {{CURRENT_QUARTER}}

- [ ] {{MILESTONE_1}}
- [ ] {{MILESTONE_2}}
- [ ] {{MILESTONE_3}}

### {{NEXT_QUARTER}}

- [ ] {{MILESTONE_4}}
- [ ] {{MILESTONE_5}}

See [Roadmap](./docs/ROADMAP.md) for full details.

---

## Dependencies

**Runtime Dependencies:**
- {{DEPENDENCY_1}} ({{DEPENDENCY_1_VERSION}})
- {{DEPENDENCY_2}} ({{DEPENDENCY_2_VERSION}})
- {{DEPENDENCY_3}} ({{DEPENDENCY_3_VERSION}})

**Dev Dependencies:**
- {{DEV_DEPENDENCY_1}}
- {{DEV_DEPENDENCY_2}}
- {{DEV_DEPENDENCY_3}}

See [package.json](./package.json) or [requirements.txt](./requirements.txt) for full list.

---

## Testing

### Test Coverage

Current: {{CURRENT_COVERAGE}}% | Target: {{TARGET_COVERAGE}}%

### Test Strategy

- {{TEST_TYPE_1}}: {{TEST_TYPE_1_COUNT}} tests
- {{TEST_TYPE_2}}: {{TEST_TYPE_2_COUNT}} tests
- {{TEST_TYPE_3}}: {{TEST_TYPE_3_COUNT}} tests

### CI/CD Pipeline

Status: {{CI_STATUS}}

Runs on: {{CI_TRIGGERS}}

---

## Security

See [SECURITY.md](./SECURITY.md) for:
- Security policies
- Vulnerability reporting
- Dependency scanning
- Compliance status

---

## License

This project is licensed under {{LICENSE}} — see [LICENSE](./LICENSE) file for details.

---

## Support & Contact

**Team:** {{TEAM_NAME}}
**Slack:** {{SLACK_CHANNEL}}
**Email:** {{CONTACT_EMAIL}}

---

## Changelog

See [CHANGELOG.md](./CHANGELOG.md) for version history and release notes.

---

## Related Projects

- **[{{RELATED_PROJECT_1}}]({{RELATED_PROJECT_1_URL}})** — {{RELATED_PROJECT_1_DESC}}
- **[{{RELATED_PROJECT_2}}]({{RELATED_PROJECT_2_URL}})** — {{RELATED_PROJECT_2_DESC}}

---

<div align="center">

**Last Updated:** {{LAST_UPDATED}}
**Maintained by:** {{MAINTAINER_NAME}}
**Status:** {{PROJECT_STATUS}}

</div>
