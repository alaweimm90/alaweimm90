# CLAUDE.md - metaHub

## Project Overview

**Purpose:** Central coordination hub and documentation center for a multi-organization monorepo ecosystem (project-agnostic, reusable)
**Type:** Coordination layer (no code, pure documentation and tooling)
**Tech Stack:** Markdown, Bash scripts, YAML configurations, GitHub Actions
**Status:** Active - Core Infrastructure

## What is metaHub?

metaHub is the **nerve center** of your monorepo ecosystem, containing:

- Unified documentation across 5 organizations and 42+ projects
- Shared automation scripts (setup, compliance, testing)
- Project templates for Python and TypeScript
- Compliance tracking and reporting tools
- Standards enforcement mechanisms

**metaHub does NOT contain production code.** It exists solely to coordinate and document the ecosystem.

## Key Commands

```bash
# Documentation
cat metaHub/docs/getting-started/overview.md      # Monorepo introduction
cat metaHub/docs/organizations/project-index.md   # All 42+ projects
cat metaHub/docs/standards/compliance.md          # Compliance guide

# Setup and verification
./metaHub/scripts/setup/install-dependencies.sh   # Install all dependencies
./metaHub/scripts/setup/setup-pre-commit.sh       # Setup pre-commit hooks
./metaHub/scripts/setup/verify-environment.sh     # Verify environment

# Compliance checking
./metaHub/scripts/compliance/check-all-orgs.sh    # Check all organizations
./metaHub/scripts/compliance/generate-report.sh   # Generate reports
./metaHub/scripts/compliance/fix-common-issues.sh # Auto-fix simple issues

# Testing
./metaHub/scripts/testing/run-all-tests.sh        # Run all tests
./metaHub/scripts/testing/check-coverage.sh       # Coverage reports

# Templates
cp -r metaHub/templates/python-project/ {org}/{project}/
cp -r metaHub/templates/typescript-project/ {org}/{project}/

# Navigation
cd ../org-alpha/                  # Example organization
cd ../org-beta/                   # Example organization
cd ../org-gamma/                  # Example organization
```

## Architecture

### Directory Structure

```
metaHub/
├── docs/                        # Documentation hub
│   ├── getting-started/         # Onboarding (overview, setup, first-contribution)
│   ├── organizations/           # Org overviews + project index
│   ├── standards/               # Standards and compliance guides
│   ├── architecture/            # Architecture decisions and design
│   └── archive/                 # Historical documentation
│
├── scripts/                     # Automation tools
│   ├── setup/                   # Environment setup scripts
│   ├── compliance/              # Compliance checking and fixing
│   ├── testing/                 # Test automation
│   └── utils/                   # General utilities
│
├── config/                      # Shared base configurations
│   ├── .pre-commit-config.yaml  # Pre-commit hooks template
│   ├── .eslintrc.base.js        # ESLint base config
│   ├── .prettierrc.base.json    # Prettier base config
│   ├── pyproject.base.toml      # Python config template
│   └── tsconfig.base.json       # TypeScript base config
│
├── templates/                   # Project scaffolding templates
│   ├── python-project/          # Full Python project template
│   ├── typescript-project/      # Full TypeScript template
│   └── organization/            # New organization template
│
├── tools/                       # Development tools
│   ├── compliance-checker/      # Compliance automation
│   ├── readme-generator/        # README generation
│   └── superprompt-builder/     # Superprompt creation tools
│
├── .github/                     # GitHub workflows and templates
│   ├── ISSUE_TEMPLATE/          # Issue templates
│   ├── PULL_REQUEST_TEMPLATE.md # PR template
│   └── workflows/               # GitHub Actions (monorepo-wide)
│
├── README.md                    # Main metaHub README
├── CLAUDE.md                    # This file
├── LICENSE                      # MIT License
└── .gitignore                   # Git ignore rules
```

### Organizations Overview (Example)

| Organization ID | Example Name        | Example Projects | Description                                  |
| --------------- | ------------------- | ---------------- | -------------------------------------------- |
| **org-one**     | Example Org One     | 3                | Placeholder description for organization one |
| **org-two**     | Example Org Two     | 5                | Placeholder description for organization two |
| **org-three**   | Example Org Three   | 4                | Placeholder description for organization three |
| **org-four**    | Example Org Four    | 2                | Placeholder description for organization four |
| **org-five**    | Example Org Five    | 1                | Placeholder description for organization five |

**Note:** Replace these example organizations with your own real organizations when adopting metaHub.

## Important Constraints

### Directory Structure Rules

**metaHub is NOT for code:**

- No `src/` directory - metaHub contains NO production code
- No `tests/` for application code - only script validation
- No language-specific packages (no npm, no pyproject.toml)

**metaHub IS for coordination:**

- Documentation in `docs/`
- Automation scripts in `scripts/`
- Configuration templates in `config/`
- Project templates in `templates/`
- Development tools in `tools/`

### File Organization Standards

1. **Documentation:** All docs in `docs/` organized by category
2. **Scripts:** All scripts in `scripts/` with clear naming
3. **Templates:** Reusable templates in `templates/`
4. **Tools:** Standalone tools in `tools/` (may have their own dependencies)

### Cross-Reference Requirements

- All documentation MUST link to relevant files in parent directory (`../REPO_STANDARDS.md`)
- Organization docs MUST reference actual organization READMEs
- Compliance docs MUST link to actual compliance reports
- Setup guides MUST point to real scripts in `scripts/`

## Testing Requirements

**metaHub testing is different:**

- NO application code tests (metaHub has no application code)
- Script validation: Ensure bash scripts are syntactically correct
- Documentation validation: Check markdown links aren't broken
- Template validation: Verify templates have all mandatory files

**Validation Commands:**

```bash
# Validate bash scripts
shellcheck scripts/**/*.sh

# Check markdown links
markdown-link-check docs/**/*.md

# Verify templates
./tools/compliance-checker/validate-template.sh templates/python-project/
./tools/compliance-checker/validate-template.sh templates/typescript-project/
```

## Code Style

### Documentation Style

**Markdown Standards:**

- Use ATX-style headings (`#`, `##`, `###`)
- Include table of contents for long documents (>500 lines)
- Use fenced code blocks with language identifiers
- Include practical examples in all guides
- Link to actual files, not placeholders
- Keep line length reasonable (<120 chars for readability)

**Writing Style:**

- Professional but accessible
- Assume reader is intelligent but new to the monorepo
- Include "why" not just "what"
- Provide both quick reference and deep explanations
- Use tables for comparisons and structured data

### Script Style

**Bash Scripts:**

```bash
#!/usr/bin/env bash
# Brief description of what this script does
#
# Usage: ./script-name.sh [options]
#
# Options:
#   -h, --help     Show this help message
#   -v, --verbose  Verbose output

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Function definitions before main logic
function show_help() {
    grep '^#' "$0" | tail -n +2 | cut -c 3-
    exit 0
}

# Main logic here
```

**Script Requirements:**

- Always include help text
- Use `set -euo pipefail` for safety
- Validate inputs before processing
- Provide clear error messages
- Log actions in verbose mode

## Common Tasks

### Adding New Documentation

1. **Determine category:**
   - Getting started → `docs/getting-started/`
   - Organization info → `docs/organizations/`
   - Standards → `docs/standards/`
   - Architecture → `docs/architecture/`

2. **Create the file:**

```bash
touch metaHub/docs/{category}/{name}.md
```

3. **Follow template structure:**

```markdown
# Title

> One-sentence description

## Overview

[Introduction and context]

## [Main Sections]

[Content organized logically]

## Quick Reference

[Tables, commands, key information]

## See Also

- [Related Doc 1](./related.md)
- [Related Doc 2](../other/doc.md)
```

4. **Link from relevant places:**
   - Update README.md navigation
   - Link from related docs
   - Update project index if needed

### Adding New Script

1. **Choose correct directory:**

```bash
# Setup scripts (install, configure)
scripts/setup/new-setup-script.sh

# Compliance scripts (check, fix, report)
scripts/compliance/new-compliance-script.sh

# Testing scripts (run tests, coverage)
scripts/testing/new-test-script.sh

# Utilities (general purpose)
scripts/utils/new-util-script.sh
```

2. **Create script with template:**

```bash
#!/usr/bin/env bash
# Description of script
set -euo pipefail

# Implementation
```

3. **Make executable:**

```bash
chmod +x scripts/{category}/{script-name}.sh
```

4. **Test the script:**

```bash
shellcheck scripts/{category}/{script-name}.sh
./scripts/{category}/{script-name}.sh --help
```

5. **Document in README:**

```markdown
### Category Scripts

\`\`\`bash
./scripts/{category}/{script-name}.sh # Brief description
\`\`\`
```

### Creating New Project Template

1. **Create template directory:**

```bash
mkdir -p metaHub/templates/{template-name}/
```

2. **Include ALL mandatory files:**

```
templates/{template-name}/
├── README.md              # Template README with placeholders
├── CLAUDE.md              # Template CLAUDE.md
├── LICENSE                # MIT or Apache 2.0
├── .gitignore             # Comprehensive ignore rules
├── CHANGELOG.md           # Initial changelog
├── CONTRIBUTING.md        # Contribution guidelines
├── SECURITY.md            # Security policy
├── pyproject.toml         # (Python) or package.json (TS)
├── .pre-commit-config.yaml # Pre-commit hooks
├── src/                   # Source structure
├── tests/                 # Test structure
└── .github/workflows/     # CI/CD workflows
```

3. **Use placeholders for customization:**

```markdown
# {{PROJECT_NAME}}

> {{PROJECT_DESCRIPTION}}

**Author:** {{AUTHOR_NAME}}
**Email:** {{AUTHOR_EMAIL}}
```

4. **Document template usage:**

```bash
cat > templates/{template-name}/TEMPLATE_USAGE.md <<EOF
# Template Usage

1. Copy template: cp -r metaHub/templates/{template-name}/ {org}/{project}/
2. Customize placeholders (PROJECT_NAME, AUTHOR_NAME, etc.)
3. Initialize git: cd {org}/{project}/ && git init
4. Install dependencies
5. Start developing
EOF
```

### Updating Compliance Reports

1. **Run compliance check:**

```bash
./metaHub/scripts/compliance/check-all-orgs.sh
```

2. **Generate new report:**

```bash
./metaHub/scripts/compliance/generate-report.sh > ../MONOREPO_COMPLIANCE_REPORT_$(date +%Y-%m-%d).md
```

3. **Update metaHub documentation:**

```bash
# Update compliance.md with new findings
# Update project-index.md with any new projects
# Update organization docs with new compliance scores
```

4. **Commit report:**

```bash
git add ../MONOREPO_COMPLIANCE_REPORT_*.md
git add metaHub/docs/standards/compliance.md
git commit -m "docs: update compliance report for $(date +%Y-%m-%d)"
```

## Navigation Patterns

### Context Loading Order for AI Assistants

**Level 1: Monorepo Overview**

```bash
1. metaHub/README.md                      # What is this monorepo?
2. metaHub/docs/getting-started/overview.md  # Detailed introduction
3. ../REPO_STANDARDS.md                   # Universal standards
```

**Level 2: Organization Context**

```bash
# For working in specific organization (e.g., AlaweinOS):
4. AlaweinOS/README.md                    # Organization overview
5. AlaweinOS/ALAWEIN_OS_SUPERPROMPT.md    # Comprehensive AI context
6. AlaweinOS/COMPLIANCE_AUDIT_*.md        # Known issues
```

**Level 3: Project-Specific Context**

```bash
# For working in specific project (e.g., optilibria):
7. AlaweinOS/optilibria/README.md         # Project README
8. AlaweinOS/optilibria/CLAUDE.md         # Project AI context
9. AlaweinOS/optilibria/docs/             # Project documentation
```

### Quick Reference Navigation

**From metaHub to Organizations (example):**

```bash
cd ../org-one/             # Example organization A
cd ../org-two/             # Example organization B
cd ../org-three/           # Example organization C
cd ../org-four/            # Example organization D
cd ../org-five/            # Example organization E
```

**From metaHub to Standards:**

```bash
cat ../REPO_STANDARDS.md                           # Repository standards
cat ../MONOREPO_COMPLIANCE_REPORT_2025-11-20.md    # Compliance report
cat ../CRITICAL_VIOLATIONS_PRIORITY_ACTION_PLAN.md # Priority fixes
```

**Within metaHub:**

```bash
cat docs/getting-started/overview.md          # Monorepo introduction
cat docs/organizations/project-index.md       # All 42+ projects
cat docs/standards/compliance.md              # Compliance guide
cat docs/architecture/monorepo-design.md      # Architecture decisions
```

## Compliance Status

**Current Overall Compliance (example):** 60.6% (Target: 95%+)

**By Organization (example):**

- org-one: 72.0% (C)
- org-two: 68.8% (D+)
- org-three: 61.0% (D+)
- org-four: 58.0% (F)
- org-five: 45.8% (F)

**Critical Issues:**

- 18 projects missing LICENSE files (43%) - LEGAL BLOCKER
- 23 projects missing SECURITY.md (55%) - SECURITY RISK
- 15 projects with missing/broken .gitignore (36%) - CREDENTIAL RISK
- 18 projects missing CLAUDE.md (43%) - OPERATIONAL
- 16 projects without CI/CD (38%) - QUALITY RISK

[Full Report](../MONOREPO_COMPLIANCE_REPORT_2025-11-20.md)

## AI Assistant Guidance

### When Working with metaHub

**DO:**

- Read existing documentation thoroughly before creating new docs
- Link to actual files in parent directory (../REPO_STANDARDS.md)
- Cross-reference between related documentation
- Keep documentation synchronized with reality
- Update metaHub docs when standards change
- Use templates when creating new projects
- Run scripts to automate repetitive tasks
- Follow markdown best practices
- Include practical examples in all guides

**DON'T:**

- Create duplicate documentation
- Add production code to metaHub (it's docs-only)
- Create placeholder docs without content
- Break links between documents
- Forget to update cross-references
- Skip validation of scripts/templates
- Use relative links that break context
- Create docs that contradict REPO_STANDARDS.md

### Documentation Best Practices

1. **Start with Purpose:**
   - Every doc should answer "Why does this exist?"
   - Include one-sentence summary at top

2. **Structure Logically:**
   - Overview → Details → Examples → Reference
   - Use headings to create scannable structure
   - Include TOC for long documents

3. **Link Intelligently:**
   - Link to actual files, not abstractions
   - Use relative paths from document location
   - Verify links aren't broken

4. **Provide Context:**
   - Explain "why" not just "what"
   - Include background for decisions
   - Reference related documentation

5. **Keep Fresh:**
   - Update when underlying systems change
   - Mark deprecated information clearly
   - Archive old docs, don't delete

### Script Development Best Practices

1. **Safety First:**

   ```bash
   set -euo pipefail  # Always use this
   ```

2. **Validate Inputs:**

   ```bash
   if [ $# -eq 0 ]; then
       echo "Error: No arguments provided"
       show_help
       exit 1
   fi
   ```

3. **Provide Help:**

   ```bash
   if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
       show_help
       exit 0
   fi
   ```

4. **Clear Error Messages:**

   ```bash
   echo "Error: Directory not found: $dir_path" >&2
   echo "Please ensure the organization directory exists." >&2
   exit 1
   ```

5. **Test Before Committing:**
   ```bash
   shellcheck script.sh
   ./script.sh --help
   ./script.sh  # Test actual functionality
   ```

## Common Patterns

### Documentation Cross-Reference Pattern

```markdown
## See Also

**Within metaHub:**

- [Getting Started](../getting-started/overview.md)
- [Project Index](../organizations/project-index.md)

**Organization Level (example):**

- [Example Org One README](../../org-one/README.md)
- [Example Org One Superprompt](../../org-one/SUPERPROMPT_EXAMPLE.md)

**Repository Level:**

- [Repository Standards](../../REPO_STANDARDS.md)
- [Compliance Report](../../MONOREPO_COMPLIANCE_REPORT_2025-11-20.md)
```

### Script Validation Pattern

```bash
#!/usr/bin/env bash
# Validate organization structure
set -euo pipefail

ORG_DIR="${1:?Error: Organization directory required}"

if [ ! -d "$ORG_DIR" ]; then
    echo "Error: Directory not found: $ORG_DIR" >&2
    exit 1
fi

echo "Validating $ORG_DIR..."

# Check mandatory files
for file in README.md LICENSE .gitignore; do
    if [ ! -f "$ORG_DIR/$file" ]; then
        echo "  ✗ Missing: $file" >&2
        ERRORS=$((ERRORS + 1))
    else
        echo "  ✓ Found: $file"
    fi
done

if [ $ERRORS -eq 0 ]; then
    echo "✓ Validation passed"
    exit 0
else
    echo "✗ Validation failed with $ERRORS errors" >&2
    exit 1
fi
```

## Contact & Resources

**Owner:** (Your Name Here)
**Email:** contact@example.com
**Website:** https://example.com
**GitHub:** [@your-handle](https://github.com/your-handle)
**Location:** (Your Location)

**For Issues:** Open an issue in the metaHub repository
**For Questions:** See organization-specific documentation in your own monorepo
**For Contributions:** See [docs/getting-started/first-contribution.md](./docs/getting-started/first-contribution.md)

## Revision History

- **v1.0** (2025-11-21): Initial metaHub CLAUDE.md creation
  - Comprehensive documentation structure
  - Script and template guidelines
  - Navigation patterns
  - AI assistant guidance

---

**Last Updated:** 2025-11-21
**Next Review:** 2025-12-21 (monthly)
**Status:** Active - Core Infrastructure

---

_metaHub: Coordinating excellence across 5 organizations and 42+ projects_
