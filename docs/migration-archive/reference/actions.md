# Golden Path Remediation Actions

**Created:** 2025-11-25
**Scope:** 55 repositories across 5 organizations
**Estimated Effort:** 120-160 engineering hours
**Timeline:** 4-6 weeks (with concurrent work)

---

## Summary of Actions by Priority

| Priority | Count | Category | Effort (hrs) |
|----------|-------|----------|-----------|
| **P0 - Critical** | 4 | Blocking issues | 40-50 |
| **P1 - Major** | 3 | Compliance gaps | 50-70 |
| **P2 - Secondary** | 5 | Enhancement | 30-40 |
| **TOTAL** | 12 | | **120-160** |

---

# P0: CRITICAL ACTIONS (Must Do First)

## P0.1: Implement .meta/repo.yaml Across All Repos

**Impact:** 55 repos
**Effort:** 15-20 hours
**Status:** BLOCKER - Cannot enforce policy without this

### Action Description
Create a standardized repository metadata file for all repositories to enable centralized governance, policy enforcement, and portfolio tracking.

### Template (`.meta/repo.yaml`)

```yaml
---
# Repository metadata for Golden Path compliance
# See: https://github.com/alaweimm90/.docs/governance/repo-metadata.md

repo:
  name: "{{ repo-name }}"
  prefix: "{{ core|lib|adapter|tool|template|demo|infra|paper|archive }}"
  type: "{{ project-type }}"

metadata:
  owner: "{{ org-name }}"
  created: "{{ YYYY-MM-DD }}"
  maintained: true
  status: "{{ active|maintenance|archived }}"

governance:
  license: "{{ license-type }}"
  codeowners: ["{{ github-handles }}"]
  security_contact: "{{ email }}"

requirements:
  languages: ["{{ language }}"]
  min_python: "3.9"  # if applicable
  min_node: "18"     # if applicable

quality:
  test_coverage_min: 70  # or 80 for libs/tools
  test_framework: "{{ pytest|jest|vitest|other }}"
  linter: "{{ eslint|ruff|other }}"
  type_checker: "{{ mypy|tsc|other }}"

ci:
  provider: "github-actions"
  workflows_enabled: true
  opa_policy_check: true
  super_linter: true

docs:
  profile: "{{ minimal|standard }}"  # per Golden Path
  location: "root"  # or "docs/"

release:
  pattern: "semver"  # or "calendar"
  auto_publish: false
  registry: "npm|pypi|crates|none"
```

### Implementation Steps

#### Step 1: Create Template File
**File:** `.meta/repo.yaml` (template)

```bash
# Create in root of each repo
cat > .meta/repo.yaml << 'EOF'
---
repo:
  name: "repo-name"
  prefix: "core"
  type: "framework"

metadata:
  owner: "alaweimm90-tools"
  created: "2025-11-25"
  maintained: true
  status: "active"

governance:
  license: "MIT"
  codeowners: ["@alaweimm90"]
  security_contact: "security@alaweimm90.dev"

requirements:
  languages: ["python"]
  min_python: "3.9"

quality:
  test_coverage_min: 80
  test_framework: "pytest"

ci:
  provider: "github-actions"
  workflows_enabled: true
  opa_policy_check: true

docs:
  profile: "standard"
EOF
```

#### Step 2: Generate for Each Repo (Automation Script)

```python
#!/usr/bin/env python3
"""Generate .meta/repo.yaml for all repos"""

import json
import os
from pathlib import Path

with open("inventory.json") as f:
    inventory = json.load(f)

for org in inventory["organizations"]:
    org_name = org["name"]
    for repo in org["repositories"]:
        repo_path = f"organizations/{org_name}/{repo['name']}"
        meta_dir = Path(repo_path) / ".meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        # Determine test coverage minimum
        test_coverage = 80 if repo["prefix"] in ["core", "lib"] else 70

        metadata = f"""---
repo:
  name: "{repo['name']}"
  prefix: "{repo['prefix']}"
  type: "{repo['type']}"

metadata:
  owner: "{org_name}"
  created: "2025-11-25"
  maintained: {str(repo['active_status'] == 'active').lower()}
  status: "{repo['active_status']}"

governance:
  license: "MIT"
  codeowners: ["@alaweimm90"]
  security_contact: "security@alaweimm90.dev"

requirements:
  languages: {json.dumps(repo['languages'])}

quality:
  test_coverage_min: {test_coverage}
  test_framework: "{detect_framework(repo)}"

ci:
  provider: "github-actions"
  workflows_enabled: {str(repo['ci_status'] != 'not-configured').lower()}
  opa_policy_check: true

docs:
  profile: "{repo['docs_profile']}"
"""

        with open(meta_dir / "repo.yaml", "w") as f:
            f.write(metadata)
```

**Timeline:** 2-3 hours (automation)
**Verification:** Check all 55 repos for .meta/repo.yaml presence

---

## P0.2: Add LICENSE Files to 34 Repos

**Impact:** 34 repos (62%)
**Effort:** 8-10 hours
**Affected Orgs:** MeatheadPhysicist (16), alaweimm90-tools (14), alaweimm90-business (3)

### Recommended License Strategy

| Organization | License | Rationale |
|--------------|---------|-----------|
| alaweimm90-science | MIT | Scientific/academic work |
| alaweimm90-tools | MIT | Developer tools |
| alaweimm90-business | Proprietary/BSL | Business apps |
| AlaweinOS | Apache-2.0 | Platform/infrastructure |
| MeatheadPhysicist | CC-BY-SA-4.0 | Research/documentation |

### Implementation

**For MIT (43 repos):**
```bash
curl -s https://raw.githubusercontent.com/github/gitignore/main/Global/LICENSE-MIT \
  | sed 's/Copyright.*/Copyright (c) 2024-2025 alaweimm90/' > LICENSE
```

**For Apache-2.0 (4 repos):**
```bash
curl -s https://www.apache.org/licenses/LICENSE-2.0.txt > LICENSE
```

**File Locations to Update:**
- alaweimm90-business: calla-lily-couture, dr-alowein-portfolio, marketing-automation
- alaweimm90-tools: admin-dashboard, alaweimm90-cli, alaweimm90-python-sdk, business-intelligence, core-framework, CrazyIdeas, devops-platform, helm-charts, LLMWorks, load-tests, marketing-center, monitoring, prompty, prompty-service
- MeatheadPhysicist: api, automation, cli, cloud, dashboard, database, education, examples, frontend, gh-pages, integrations, mlops, monitoring, notebooks, quantum, visualizations

**Timeline:** 3-4 hours
**Automation:** Batch update via script

---

## P0.3: Add SECURITY.md to 42 Repos

**Impact:** 42 repos (76%)
**Effort:** 10-12 hours
**Key Repos:** All except 13 compliant repos

### Template (`SECURITY.md`)

```markdown
# Security Policy

## Reporting Security Vulnerabilities

If you discover a security vulnerability in {{ repo-name }}, please email:

**Security Contact:** security@alaweimm90.dev

Please do NOT open a public GitHub issue for security vulnerabilities.

## Supported Versions

| Version | Status |
|---------|--------|
| {{ latest }} | ✅ Supported |
| {{ previous }} | ⚠️ Security fixes only |
| < {{ older }} | ❌ Unsupported |

## Security Best Practices

- Keep dependencies updated
- Use strong authentication
- Report vulnerabilities responsibly
- Review code for injection vulnerabilities

## Compliance

This project follows these standards:
- OWASP Top 10 awareness
- Dependency scanning via Dependabot
- Code quality via {{ linter-tool }}
- Coverage tracking

See [CONTRIBUTING.md](CONTRIBUTING.md) for development standards.
```

**Template Locations to Create:**
All 42 repos listed in gaps.md under "Missing SECURITY.md"

**Timeline:** 4-5 hours
**Automation:** Template replacement for all repos

---

## P0.4: Setup CI/CD for 18 Repos with Zero Automation

**Impact:** 18 repos (33%)
**Effort:** 15-18 hours
**Repos Affected:**

alaweimm90-business: (calla-lily-couture, dr-alowein-portfolio, marketing-automation)
alaweimm90-tools: (admin-dashboard, helm-charts, LLMWorks, load-tests, monitoring, prompty)
MeatheadPhysicist: (api, automation, cloud, dashboard, database, education, examples, frontend, gh-pages, mlops, notebooks, visualizations)

### Minimal CI/CD Workflow Template

**File:** `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up environment
        run: |
          [[ -f "package.json" ]] && npm install
          [[ -f "requirements.txt" ]] && pip install -r requirements.txt

      - name: Run tests
        run: |
          [[ -f "package.json" ]] && npm test
          [[ -f "pytest.ini" ]] && pytest

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run super-linter
        uses: github/super-linter@v4
        env:
          DEFAULT_BRANCH: main
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

**Deployment Workflow Template** (Optional for tools/services):

```yaml
name: Deploy

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Deploy to {{ platform }}
        run: |
          # Add deployment steps
          echo "Deploying to {{ target }}"
```

**Timeline:** 8-10 hours (setup + validation)

---

## P0 Summary

**Total P0 Effort:** 40-50 hours
**Timeline:** 1-2 weeks (parallel work)
**Validation Checklist:**
- [ ] .meta/repo.yaml in all 55 repos
- [ ] LICENSE in all 55 repos
- [ ] SECURITY.md in all 55 repos
- [ ] GitHub Actions workflows verified
- [ ] All repos pass policy validation

---

# P1: MAJOR ACTIONS (High Priority)

## P1.1: Add Test Coverage to 33 Repos

**Impact:** 33 repos (60%)
**Effort:** 50-70 hours
**Target:** ≥70% coverage for demos, ≥80% for libs/tools

### Strategy by Language

#### Python (20 repos)
**Framework:** pytest
**Coverage Tool:** pytest-cov

```bash
# Add to requirements-dev.txt
pytest>=7.0
pytest-cov>=4.0
pytest-mock>=3.10

# pyproject.toml configuration
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=src --cov-report=html --cov-report=term-missing"
minversion = "7.0"

[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/venv/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if TYPE_CHECKING:",
    "raise AssertionError",
    "raise NotImplementedError",
]
```

**Example Test File Structure:**
```
repo/
├── src/
│   ├── __init__.py
│   ├── module_a.py
│   └── module_b.py
├── tests/
│   ├── __init__.py
│   ├── test_module_a.py
│   └── test_module_b.py
├── pytest.ini
└── pyproject.toml
```

#### TypeScript/JavaScript (17 repos)
**Framework:** Jest or Vitest
**Coverage Tool:** Built-in

```json
{
  "jest": {
    "preset": "ts-jest",
    "testEnvironment": "node",
    "collectCoverageFrom": [
      "src/**/*.ts",
      "!src/**/*.d.ts"
    ],
    "coverageThreshold": {
      "global": {
        "branches": 70,
        "functions": 70,
        "lines": 70,
        "statements": 70
      }
    }
  }
}
```

### Repos Requiring Test Coverage (Priority Order)

**High Priority (0% coverage, core/lib repos):**
1. repz (alaweimm90-business) - Core framework
2. HELIOS (alaweimm90-tools) - Library
3. SimCore (AlaweinOS) - Core simulator
4. MEZAN (AlaweinOS) - Core framework
5. spin-circ (alaweimm90-science) - Library

**Medium Priority (Existing tests, need improvement):**
- 12 repos with <50% coverage (see inventory)

**Lower Priority (Demo repos):**
- 16 repos with 0% coverage in alaweimm90-business and MeatheadPhysicist

### Implementation Steps

1. **Add test framework configuration**
2. **Create tests/ directory structure**
3. **Write unit tests** (20-50% of lines)
4. **Write integration tests** (10-20% of lines)
5. **Verify coverage** meets threshold
6. **Add coverage badge** to README
7. **Enable coverage reporting** in CI/CD

**Timeline:** 40-50 hours (core/lib repos) + 10-20 hours (demo repos)

---

## P1.2: Add CONTRIBUTING.md to 42 Repos

**Impact:** 42 repos (76%)
**Effort:** 8-10 hours
**Affected Orgs:** Most repos missing this

### Template (`CONTRIBUTING.md`)

```markdown
# Contributing to {{ repo-name }}

Thank you for your interest in contributing! This document provides guidelines
and instructions for contributing to this project.

## Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/{{ user }}/{{ repo }}.git
   cd {{ repo }}
   ```

2. **Install dependencies**
   ```bash
   # Python projects
   pip install -e ".[dev]"

   # Node.js projects
   npm install
   ```

3. **Create a feature branch**
   ```bash
   git checkout -b feature/my-feature
   ```

## Development Workflow

### Code Standards

- **Python:** Follow PEP 8, use `ruff`, `black`, `mypy`
- **JavaScript:** Follow ESLint config, use Prettier
- **Documentation:** Use Markdown, follow Google style guide

### Testing

```bash
# Python
pytest --cov=src tests/

# JavaScript
npm test -- --coverage
```

**Minimum Coverage:** 70% (demos), 80% (libraries/tools)

### Linting

```bash
# Python
ruff check .
black --check .
mypy src/

# JavaScript
npm run lint
```

## Submitting Changes

1. **Commit with clear messages**
   ```
   type(scope): description

   Longer explanation if needed.

   Fixes #123
   ```

2. **Push to your fork**
3. **Open a Pull Request**
4. **Respond to review feedback**

## Pull Request Guidelines

- Link to related issues
- Describe your changes clearly
- Ensure all tests pass
- Maintain or improve code coverage
- Update documentation as needed

## Code Review Process

- At least 1 approval required
- All CI checks must pass
- No merge conflicts
- No stale reviews

## Reporting Issues

Please use GitHub Issues for bug reports and feature requests.

### Bug Report Template
```markdown
## Description
Brief description of the bug.

## Steps to Reproduce
1. ...
2. ...

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens.

## Environment
- OS:
- Python/Node version:
- Package version:
```

## Licensing

By contributing, you agree that your contributions will be licensed under the
project's license (see [LICENSE](LICENSE)).

---

Questions? Open an issue or contact the maintainers.
```

**Timeline:** 4-5 hours (bulk creation)

---

## P1.3: Create Reusable CI/CD Workflow Templates

**Impact:** Standardization across 55 repos
**Effort:** 8-12 hours
**Deliverables:** 5-8 reusable workflow files

### Reusable Workflows Location

**File Structure:**
```
.github/workflows/
├── _python-test.yml          # Reusable Python test workflow
├── _node-test.yml            # Reusable Node.js test workflow
├── _publish-pypi.yml         # Reusable Python package publish
├── _publish-npm.yml          # Reusable npm package publish
├── _deploy-docker.yml        # Reusable Docker deployment
└── .../
```

### Example: Reusable Python Test Workflow

**File:** `.github/workflows/_python-test.yml`

```yaml
name: Python Test (Reusable)

on:
  workflow_call:
    inputs:
      python_versions:
        description: "Python versions to test"
        required: false
        default: '["3.9", "3.10", "3.11"]'
        type: string
      test_command:
        description: "Command to run tests"
        required: false
        default: "pytest --cov=src tests/"
        type: string

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ${{ fromJSON(inputs.python_versions) }}

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
          cache: "pip"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Run tests
        run: ${{ inputs.test_command }}

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
```

### Usage in Individual Repos

**File:** `.github/workflows/ci.yml` (in each repo)

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    uses: {{ owner }}/.github/.github/workflows/_python-test.yml@main
    with:
      python_versions: '["3.9", "3.10", "3.11"]'
      test_command: "pytest --cov=src tests/"
```

**Timeline:** 8-10 hours

---

## P1 Summary

**Total P1 Effort:** 50-70 hours
**Timeline:** 2-4 weeks
**Validation:**
- [ ] Tests added to all 33 repos
- [ ] Coverage ≥70% across portfolio
- [ ] CONTRIBUTING.md in all 42 repos
- [ ] Reusable workflows tested and documented

---

# P2: SECONDARY ACTIONS (Enhancement)

## P2.1: Consolidate/Archive MeatheadPhysicist Organization

**Impact:** 14 undocumented repos
**Effort:** 10-15 hours
**Recommendation:** Move to .archive/ or consolidate into parent repos

### Candidates for Archival
```
MeatheadPhysicist/
├── api/              → Archive (no README, no CI)
├── automation/       → Archive (undocumented)
├── cloud/            → Archive or consolidate
├── database/         → Archive or consolidate
├── education/        → Archive (no README)
├── mlops/            → Archive (no README)
├── notebooks/        → Archive (no README)
├── visualizations/   → Archive (no README)
└── [other stubs]
```

### Implementation

```bash
# For each repo to archive:
mkdir -p .archive/{{ repo-name }}-{{ date }}
mv {{ repo-path }} .archive/{{ repo-name }}-{{ date }}/
git add .archive/
git commit -m "archive: Move {{ repo }} to archive (no active development)"
```

**Timeline:** 5-8 hours

---

## P2.2: Documentation Standardization

**Impact:** All 55 repos
**Effort:** 8-10 hours
**Deliverables:** Style guide + README template

### README.md Template

```markdown
# {{ Project Name }}

> {{ One-line description }}

{{ Badge row: license, build, coverage }}

## Features

- Feature 1
- Feature 2
- Feature 3

## Installation

\`\`\`bash
# For Python
pip install {{ package-name }}

# For Node.js
npm install {{ package-name }}
\`\`\`

## Quick Start

\`\`\`python
from {{ module }} import {{ function }}

result = {{ function }}(arg1, arg2)
\`\`\`

## Documentation

- [Getting Started](docs/getting-started.md)
- [API Reference](docs/api.md)
- [Examples](examples/)

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md)

## Security

See [SECURITY.md](SECURITY.md)

## License

[{{ LICENSE }}](LICENSE) © 2024-2025

## Contact

- **Issues:** [GitHub Issues]({{ repo }}/issues)
- **Security:** [security@alaweimm90.dev](mailto:security@alaweimm90.dev)
```

**Timeline:** 3-4 hours

---

## P2.3: OPA Policy Validation

**Impact:** Enforcement across portfolio
**Effort:** 8-12 hours
**Deliverables:** OPA policy rules + CI integration

### Example OPA Policy (`.meta/policies/golden-path.rego`)

```rego
package golden_path

# Rule 1: Must have README
violations[msg] {
    not input.files["README.md"]
    msg := "Missing required file: README.md"
}

# Rule 2: Must have LICENSE
violations[msg] {
    not input.files["LICENSE"]
    msg := "Missing required file: LICENSE"
}

# Rule 3: Must have SECURITY.md
violations[msg] {
    not input.files["SECURITY.md"]
    msg := "Missing required file: SECURITY.md"
}

# Rule 4: Must have .meta/repo.yaml
violations[msg] {
    not input.files[".meta/repo.yaml"]
    msg := "Missing required file: .meta/repo.yaml"
}

# Rule 5: Must have tests for libs/tools
violations[msg] {
    input.repo.prefix in ["core", "lib"]
    input.test_coverage < 80
    msg := sprintf("Test coverage too low: %d%% (minimum 80%% for libs)", [input.test_coverage])
}

# Rule 6: Must have CI/CD
violations[msg] {
    not input.ci_configured
    msg := "GitHub Actions not configured"
}
```

**Timeline:** 5-8 hours

---

## P2 Summary

**Total P2 Effort:** 30-40 hours
**Timeline:** 2-3 weeks
**Validation:**
- [ ] Archive recommendations reviewed
- [ ] Documentation standardized
- [ ] OPA policies deployed and tested

---

# Detailed Implementation Patches

## Patch 1: .meta/repo.yaml Generator Script

**File:** `scripts/generate-repo-metadata.py`

```python
#!/usr/bin/env python3
"""Generate .meta/repo.yaml for all repositories"""

import json
import os
from pathlib import Path
from typing import Dict, Any

def load_inventory(path: str = "inventory.json") -> Dict[str, Any]:
    """Load inventory.json"""
    with open(path) as f:
        return json.load(f)

def get_test_framework(repo: Dict) -> str:
    """Detect test framework based on repo type"""
    languages = repo.get("languages", [])

    if "Python" in languages:
        return "pytest"
    elif any(lang in languages for lang in ["TypeScript", "JavaScript"]):
        return "jest"
    else:
        return "pytest"  # default

def generate_metadata(org: str, repo: Dict) -> str:
    """Generate YAML metadata for a repo"""

    # Determine test coverage minimum
    min_coverage = 80 if repo["prefix"] in ["core", "lib"] else 70
    test_framework = get_test_framework(repo)

    yaml_content = f"""---
# Repository metadata for Golden Path compliance
# Generated: 2025-11-25
# See: docs/governance/repo-metadata.md

repo:
  name: "{repo['name']}"
  prefix: "{repo['prefix']}"
  type: "{repo['type']}"

metadata:
  owner: "{org}"
  created: "2025-11-25"
  maintained: {str(repo['active_status'] == 'active').lower()}
  status: "{repo['active_status']}"

governance:
  license: "MIT"
  codeowners: ["@alaweimm90"]
  security_contact: "security@alaweimm90.dev"

requirements:
  languages: {json.dumps(repo['languages'])}
  min_python: "3.9"
  min_node: "18"

quality:
  test_coverage_min: {min_coverage}
  test_framework: "{test_framework}"
  linter: "ruff"
  type_checker: "mypy"

ci:
  provider: "github-actions"
  workflows_enabled: {str(repo['ci_status'] != 'not-configured').lower()}
  opa_policy_check: true
  super_linter: true

docs:
  profile: "{repo['docs_profile']}"
  location: "root"

release:
  pattern: "semver"
  auto_publish: false
  registry: "none"
"""
    return yaml_content

def main():
    """Generate .meta/repo.yaml for all repos"""
    inventory = load_inventory()

    for org in inventory["organizations"]:
        org_name = org["name"]
        org_path = Path(f"organizations/{org_name}")

        for repo in org["repositories"]:
            repo_path = org_path / repo["name"]
            meta_dir = repo_path / ".meta"

            # Create .meta directory if it doesn't exist
            meta_dir.mkdir(parents=True, exist_ok=True)

            # Generate metadata
            metadata = generate_metadata(org_name, repo)

            # Write to file
            meta_file = meta_dir / "repo.yaml"
            with open(meta_file, "w") as f:
                f.write(metadata)

            print(f"✓ Created {meta_file}")

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
cd organizations
python ../scripts/generate-repo-metadata.py
```

---

## Patch 2: License Template Batch Script

**File:** `scripts/add-licenses.sh`

```bash
#!/bin/bash
# Add LICENSE files to repos

set -e

# Define organizations and their license preferences
declare -A ORG_LICENSE
ORG_LICENSE["alaweimm90-science"]="MIT"
ORG_LICENSE["alaweimm90-tools"]="MIT"
ORG_LICENSE["alaweimm90-business"]="MIT"
ORG_LICENSE["AlaweinOS"]="Apache-2.0"
ORG_LICENSE["MeatheadPhysicist"]="CC-BY-SA-4.0"

# Function to add MIT license
add_mit_license() {
    local repo_path=$1
    cat > "${repo_path}/LICENSE" << 'EOF'
MIT License

Copyright (c) 2024-2025 alaweimm90

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF
    echo "✓ Added MIT license to ${repo_path}"
}

# Main loop
for org_dir in organizations/*/; do
    org_name=$(basename "$org_dir")

    # Skip if not in our org list
    [[ ! ${ORG_LICENSE[$org_name]+_} ]] && continue

    license=${ORG_LICENSE[$org_name]}

    for repo_dir in "${org_dir}"*/; do
        # Skip hidden dirs and template dirs
        [[ $(basename "$repo_dir") == .* ]] && continue
        [[ $(basename "$repo_dir") == "docs" ]] && continue

        # Check if license exists
        if [[ ! -f "${repo_dir}LICENSE" ]]; then
            echo "Adding ${license} to $(basename "$repo_dir")..."

            if [[ "$license" == "MIT" ]]; then
                add_mit_license "$repo_dir"
            fi
        fi
    done
done

echo "✓ License batch update complete"
```

---

## Patch 3: SECURITY.md Template Script

**File:** `scripts/add-security-docs.sh`

```bash
#!/bin/bash
# Add SECURITY.md to all repos

for org_dir in organizations/*/; do
    org_name=$(basename "$org_dir")

    for repo_dir in "${org_dir}"*/; do
        repo_name=$(basename "$repo_dir")

        # Skip if exists
        [[ -f "${repo_dir}SECURITY.md" ]] && continue

        cat > "${repo_dir}SECURITY.md" << 'EOF'
# Security Policy

## Reporting Security Vulnerabilities

If you discover a security vulnerability in this repository, please email:

**security@alaweimm90.dev**

Please do **NOT** open a public GitHub issue for security vulnerabilities.

## Supported Versions

| Version | Status |
|---------|--------|
| Latest | ✅ Actively maintained |
| Previous | ⚠️ Security fixes only |
| Older | ❌ Unsupported |

## Security Best Practices

- Keep dependencies updated via Dependabot
- Run regular security audits
- Use strong authentication
- Report vulnerabilities responsibly
- Never commit secrets or credentials

## Compliance

This project follows:
- OWASP Top 10 awareness
- Dependency scanning (Dependabot)
- Code quality checks in CI/CD
- Security headers enforcement

See [CONTRIBUTING.md](CONTRIBUTING.md) for development standards.
EOF

        echo "✓ Added SECURITY.md to ${repo_dir}"
    done
done

echo "✓ Security documentation batch update complete"
```

---

# Execution Plan

## Phase 1: Critical (Week 1-2)
1. **Run .meta/repo.yaml generator** (2 hours)
2. **Batch add LICENSE files** (3 hours)
3. **Batch add SECURITY.md** (2 hours)
4. **Setup CI/CD for 18 repos** (15 hours)
5. **Validation & testing** (5 hours)

## Phase 2: Major (Week 3-4)
1. **Add test frameworks** (25 hours)
2. **Write core library tests** (25 hours)
3. **Batch add CONTRIBUTING.md** (5 hours)

## Phase 3: Enhancement (Week 5-6)
1. **Create reusable workflow templates** (8 hours)
2. **Document standardization** (4 hours)
3. **Archive old repos** (5 hours)
4. **OPA policy implementation** (8 hours)

## Ongoing
- Code review and validation
- CI/CD testing
- Documentation updates

---

# Success Metrics

By completion of all actions:

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Repos with README | 46 (84%) | 55 (100%) | ⚠️ |
| Repos with LICENSE | 21 (38%) | 55 (100%) | ❌ |
| Repos with SECURITY.md | 13 (24%) | 55 (100%) | ❌ |
| Repos with CONTRIBUTING.md | 13 (24%) | 55 (100%) | ❌ |
| Repos with .meta/repo.yaml | 0 (0%) | 55 (100%) | ❌ |
| Repos with CI/CD | 37 (67%) | 55 (100%) | ⚠️ |
| Repos with tests | 22 (40%) | 55 (100%) | ❌ |
| Avg test coverage | 42% | 75% | ❌ |
| Golden Path compliant | 17 (31%) | 55 (100%) | ❌ |

---

# Risk Mitigation

1. **Breaking Changes:** Review all workflow changes on staging branch first
2. **Merge Conflicts:** Use automation to minimize manual conflicts
3. **Coverage Regressions:** Keep baseline metrics tracked
4. **Deployment Issues:** Test each workflow individually before deployment
5. **Resource Constraints:** Parallelize work across team members

