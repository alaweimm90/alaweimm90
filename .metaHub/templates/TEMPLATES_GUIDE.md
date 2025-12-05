# Template Customization Guide

This directory contains reusable README and documentation templates for:
- **Personal GitHub profiles**
- **Organization profiles**
- **Consumer repositories**
- **Enterprise governance repos**

All templates use **variable substitution** for easy customization.

---

## Template Structure

```
.metaHub/templates/
├── profiles/                     # Personal profile README templates
│   └── README_PROFILE_TEMPLATE.md
├── organizations/                # Organization profile templates
│   └── README_ORG_TEMPLATE.md
├── consumer-repos/               # Consumer repository templates
│   └── README_CONSUMER_TEMPLATE.md
└── TEMPLATES_GUIDE.md           # This file
```

---

## How to Use Templates

### Step 1: Copy Template

```bash
# For personal profile
cp .metaHub/templates/profiles/README_PROFILE_TEMPLATE.md README.md

# For organization
cp .metaHub/templates/organizations/README_ORG_TEMPLATE.md README.md

# For consumer repo
cp .metaHub/templates/consumer-repos/README_CONSUMER_TEMPLATE.md README.md
```

### Step 2: Replace Variables

All templates use `{{VARIABLE_NAME}}` syntax for placeholders.

**Option A: Manual Replacement**
- Open the copied file in an editor
- Find and replace each `{{VARIABLE}}` with actual values

**Option B: Automated Replacement (Bash)**
```bash
# Example: Replace with a variables file
source variables.env
envsubst < README_PROFILE_TEMPLATE.md > README.md
```

**Option C: Python Script**
```python
import re

def substitute_template(template_path, variables):
    with open(template_path, 'r') as f:
        content = f.read()

    for key, value in variables.items():
        content = content.replace(f"{{{{{key}}}}}", str(value))

    return content

# Example usage
variables = {
    'FULL_NAME': 'Meshal Alawein',
    'GITHUB_USERNAME': 'alawein',
    'EMAIL': 'meshal@berkeley.edu',
    # ... more variables
}

result = substitute_template('README_PROFILE_TEMPLATE.md', variables)
with open('README.md', 'w') as f:
    f.write(result)
```

---

## Profile Template Variables

**Identity & Contact**
```
{{FULL_NAME}}              # Your full name
{{GITHUB_USERNAME}}        # GitHub username
{{EMAIL}}                  # Email address
{{WEBSITE_URL}}            # Portfolio website URL
{{WEBSITE_DOMAIN}}         # Domain name (e.g., malawein.com)
{{LINKEDIN_HANDLE}}        # LinkedIn profile handle
{{SPOTIFY_USERNAME}}       # Spotify username (for music widget)
```

**Branding & Colors**
```
{{PRIMARY_COLOR}}          # Primary badge color (hex, e.g., A855F7)
{{SECONDARY_COLOR}}        # Secondary badge color (hex, e.g., EC4899)
{{ACCENT_COLOR}}           # Accent color (hex, e.g., 4CC9F0)
{{GITHUB_THEME}}           # GitHub stats theme (e.g., tokyonight)
{{GRAPH_THEME}}            # Activity graph theme
```

**Profile Content**
```
{{PROFESSIONAL_DESCRIPTOR}}  # One-line job title (e.g., "Computational physicist & systems engineer")
{{ELEVATOR_PITCH}}          # Short pitch (1-2 sentences)
{{BACKGROUND_STORY}}        # Your background and journey
{{PERSONAL_PHILOSOPHY}}     # Philosophy or approach
{{PERSONAL_QUOTE}}          # Inspirational quote
```

**Skills & Expertise**
```
{{EXPERTISE_1}} / {{EXPERTISE_1_BAR}} / {{EXPERTISE_1_PCT}}
{{EXPERTISE_2}} / {{EXPERTISE_2_BAR}} / {{EXPERTISE_2_PCT}}
# Example: Quantum Mechanics / ██████████████████░░ / 95%
```

**Tech Stack**
```
{{TECH_BADGE_1}}           # e.g., ![Python](https://img.shields.io/badge/...)
{{TECH_BADGE_2}}
# ... up to 8 badges
```

**Projects**
```
{{PROJECT_N_NAME}}         # Project name
{{PROJECT_N_TAGLINE}}      # One-line description
{{PROJECT_N_DESCRIPTION}}  # Full description
{{PROJECT_N_TECH}}         # Technologies used
{{PROJECT_N_CODE_SNIPPET}} # Code example
{{PROJECT_N_STATUS}}       # Active, Beta, Archived
{{PROJECT_N_COLOR}}        # Status badge color
{{PROJECT_N_BADGE_1}}      # Custom badge
```

**Current Work**
```
{{CURRENT_RESEARCH}}       # Current research focus
{{BUILDING_LIST}}          # List of projects building (comma-separated)
{{LEARNING_GOAL}}          # Current learning goal
{{LEARNING_BOOK}}          # Currently reading
{{SPRINT_TITLE}}           # Sprint/project name
{{GOAL_1}}, {{GOAL_2}}, {{GOAL_3}}  # Sprint goals
```

**Engagement**
```
{{RESPONSE_CATEGORY_1}}    # e.g., research_collab
{{RESPONSE_TIME_1}}        # e.g., 24-48 hours
{{FUN_FACT_1}} ... {{FUN_FACT_6}}  # Personal tidbits
{{EASTER_EGG_MESSAGE}}     # Hidden message for readers
```

**Metadata**
```
{{LAST_UPDATED}}           # Update timestamp (YYYY)
{{MUSIC_PREFERENCE}}       # Music preference description
```

---

## Organization Template Variables

**Organization Identity**
```
{{ORG_NAME}}               # Organization name
{{ORG_DESCRIPTION}}        # Short description
{{ORG_MISSION}}            # Organization mission
{{ORG_FOUNDED}}            # Founding year
{{ORG_LOCATION}}           # Geographic location
{{ORG_CONTACT_EMAIL}}      # Contact email
{{ORG_STATUS}}             # Status (Active, Inactive, etc.)
{{STATUS_COLOR}}           # Status badge color
```

**Organization Stats**
```
{{REPO_COUNT}}             # Number of repositories
{{TEAM_COUNT}}             # Number of teams
{{CONTRIBUTOR_COUNT}}      # Active contributors
{{MONTHLY_COMMITS}}        # Monthly commit average
{{LANGUAGE_COUNT}}         # Number of programming languages
{{COMPLIANCE_SCORE}}       # Compliance percentage
```

**Teams**
```
{{TEAM_N_NAME}}            # Team name
{{TEAM_N_FOCUS}}           # Team focus area
{{TEAM_N_COUNT}}           # Team member count
{{TEAM_N_LEAD}}            # Team lead name
```

**Repositories**
```
{{REPO_N_NAME}}            # Repository name
{{REPO_N_URL}}             # Repository URL
{{REPO_N_DESCRIPTION}}     # Repository description
{{REPO_N_LANG}}            # Programming language
{{REPO_N_LANG_COLOR}}      # Language badge color
{{REPO_N_STATUS}}          # Status (Active, Archived, etc.)
{{REPO_N_STATUS_COLOR}}    # Status badge color
```

**Infrastructure & Tools**
```
{{INFRA_N_NAME}}           # Tool name
{{INFRA_N_URL}}            # Tool URL
{{INFRA_N_DESCRIPTION}}    # Tool description
{{RESEARCH_N_NAME}}        # Research project name
{{RESEARCH_N_URL}}         # Research project URL
{{RESEARCH_N_DESCRIPTION}} # Research project description
```

**Tech Stack**
```
{{TECH_LANGUAGE_N}}        # Programming language badge/description
{{TECH_FRAMEWORK_N}}       # Framework badge/description
{{TECH_INFRA_N}}           # Infrastructure tool
{{TECH_DEVOPS_N}}          # DevOps tool
```

**Governance & Links**
```
{{GOVERNANCE_CONTRACT_URL}}   # URL to governance contract repo
{{ORG_DASHBOARD_URL}}         # Organization dashboard URL
{{CATALOG_URL}}               # Repository catalog URL
{{ORG_SLACK_CHANNEL}}         # Slack channel name
{{ORG_LICENSE}}               # License type (MIT, Apache 2.0, etc.)
```

**Organization Leadership**
```
{{ORG_LEAD}}               # Lead person name
{{ORG_EMAIL}}              # Organization email
{{ORG_MAINTAINER}}         # Maintainer name
```

**Metadata**
```
{{LAST_UPDATED}}           # Last update date
```

---

## Consumer Repository Template Variables

**Repository Identity**
```
{{REPO_NAME}}              # Repository name
{{REPO_SHORT_DESCRIPTION}} # One-line description
{{REPO_LONG_DESCRIPTION}}  # Full description
{{REPO_URL}}               # Repository clone URL
{{VERSION}}                # Current version
{{STATUS}}                 # Status (Active, Beta, Experimental, etc.)
{{STATUS_COLOR}}           # Status badge color
{{STATUS_LINK}}            # Status page link
```

**Governance**
```
{{GOVERNANCE_LEVEL}}       # Tier level or compliance status
{{GOVERNANCE_CONTRACT_URL}} # Link to governance contract
{{REPO_TYPE}}              # lib, tool, core, research, demo, workspace
{{REPO_LANGUAGE}}          # primary language
{{REPO_TIER}}              # 1, 2, or 3
{{COVERAGE_THRESHOLD}}     # Test coverage minimum (e.g., 80%)
{{DOCS_PROFILE}}           # standard or minimal
{{TEAM_NAME}}              # Responsible team
{{CODE_REVIEWER_TEAM}}     # Code review team
{{APPROVAL_COUNT}}         # Required approvals
```

**Tech Stack**
```
{{PRIMARY_LANGUAGE}}       # Main programming language
{{FRAMEWORK}}              # Framework used
{{RUNTIME}}                # Runtime environment
{{DEPENDENCY_N}}           # Dependencies
{{DEV_DEPENDENCY_N}}       # Dev dependencies
{{DEPENDENCY_N_VERSION}}   # Dependency version
```

**Quick Start**
```
{{PREREQUISITE_N}}         # Prerequisites
{{INSTALL_COMMAND}}        # Installation command
{{TEST_COMMAND}}           # Test command
{{DEV_COMMAND}}            # Dev server command
{{USAGE_EXAMPLE}}          # Code example
```

**Project Structure**
```
{{MODULE_N}}               # Module/package name
{{MODULE_N_DESCRIPTION}}   # Module purpose
{{CONFIG_FILE_N}}          # Config file name
{{CONFIG_FILE_N_PURPOSE}}  # Config file purpose
```

**Compliance**
```
{{SECURITY_CHECK_DETAILS}} # Security check results
{{TEST_COVERAGE}}          # Current test coverage %
{{DOC_STATUS}}             # Documentation status
{{CURRENT_COVERAGE}}       # Current coverage %
{{TARGET_COVERAGE}}        # Target coverage %
```

**Development**
```
{{DEV_SETUP_COMMAND}}      # Development setup
{{TEST_ALL_COMMAND}}       # Run all tests
{{TEST_SPECIFIC_COMMAND}}  # Run specific test
{{TEST_COVERAGE_COMMAND}}  # Coverage command
{{LINT_COMMAND}}           # Linter command
{{FORMAT_COMMAND}}         # Code formatter
{{TYPE_CHECK_COMMAND}}     # Type checker
{{BUILD_DEV_COMMAND}}      # Dev build
{{BUILD_PROD_COMMAND}}     # Production build
{{BUILD_DOCKER_COMMAND}}   # Docker build
{{CODE_STANDARD_LANG}}     # Code standard language
{{LINTER_NAME}}            # Linter tool
{{FORMATTER_NAME}}         # Formatter tool
{{MIN_COVERAGE}}           # Minimum coverage %
{{COMMIT_MESSAGE_FORMAT}}  # Commit message format
```

**Deployment**
```
{{DEPLOY_PREREQ_N}}        # Deployment prerequisites
{{LOG_LOCATION}}           # Logs location
{{METRICS_URL}}            # Metrics URL
{{ALERT_CONFIG}}           # Alert configuration
{{DEPLOY_COMMAND}}         # Deployment command
{{CI_STATUS}}              # CI/CD status
{{CI_TRIGGERS}}            # CI triggers
```

**Performance & Quality**
```
{{BENCHMARK_N}}            # Benchmark name
{{BENCHMARK_N_BASELINE}}   # Baseline value
{{BENCHMARK_N_STATUS}}     # Status
{{ISSUE_N}}                # Known issue
{{ISSUE_N_WORKAROUND}}     # Workaround
{{ISSUE_N_STATUS}}         # Issue status
```

**Roadmap & Future**
```
{{CURRENT_QUARTER}}        # Current quarter (Q1, Q2, etc.)
{{NEXT_QUARTER}}           # Next quarter
{{MILESTONE_N}}            # Milestone description
```

**Testing**
```
{{TEST_TYPE_N}}            # Test type (unit, integration, etc.)
{{TEST_TYPE_N_COUNT}}      # Number of tests
```

**Contact & Metadata**
```
{{SLACK_CHANNEL}}          # Slack channel
{{CONTACT_EMAIL}}          # Contact email
{{MAINTAINER_NAME}}        # Maintainer name
{{PROJECT_STATUS}}         # Project status
{{LICENSE}}                # License type
{{LAST_UPDATED}}           # Last update
```

---

## Enterprise Governance Template

For an enterprise governance repository, use the consumer repo template with these additional governance-specific variables:

```
{{GOVERNANCE_VERSION}}        # Governance framework version
{{SLSA_LEVEL}}               # SLSA level (2, 3, or 4)
{{COMPLIANCE_FRAMEWORKS}}    # NIST, EO 14028, SOC 2, etc.
{{POLICY_COUNT}}             # Number of policies
{{WORKFLOW_COUNT}}           # Number of reusable workflows
{{SCHEMA_COUNT}}             # Number of schemas
{{MONITORING_STATUS}}        # Active, Pilot, Planning
{{ENFORCEMENT_MODE}}         # warning-only, blocking, auto-fix
```

---

## Example: Personalizing the Profile Template

### Variables File (variables.env)

```bash
# Identity
FULL_NAME="Meshal Alawein"
GITHUB_USERNAME="alawein"
EMAIL="meshal@berkeley.edu"
WEBSITE_URL="https://malawein.com"
WEBSITE_DOMAIN="malawein.com"
LINKEDIN_HANDLE="alawein"

# Branding
PRIMARY_COLOR="A855F7"
SECONDARY_COLOR="EC4899"
ACCENT_COLOR="4CC9F0"

# Profile Content
PROFESSIONAL_DESCRIPTOR="Computational physicist & systems engineer"
ELEVATOR_PITCH="I use math and code to untangle hard problems."
PERSONAL_QUOTE="The best code is like a physics equation—minimal, elegant, and captures the essence of truth."

# Projects
PROJECT_1_NAME="Optilibria"
PROJECT_1_TAGLINE="Universal optimization framework"
PROJECT_1_TECH="Python, JAX, CUDA, NumPy"
PROJECT_1_STATUS="Active"

# ... more variables
```

### Substitution Command

```bash
# Bash
source variables.env
envsubst < README_PROFILE_TEMPLATE.md > README.md

# Or with sed
sed -i 's|{{FULL_NAME}}|Meshal Alawein|g' README.md
sed -i 's|{{GITHUB_USERNAME}}|alawein|g' README.md
# ... etc
```

---

## Best Practices

1. **Use consistent colors** — Pick 3-4 colors and use throughout
2. **Keep bios concise** — 2-3 sentences maximum
3. **Update regularly** — Refresh quarterly at minimum
4. **Link to governance** — Always reference the governance contract repo
5. **Test rendering** — Check on GitHub before committing
6. **Version templates** — Update TEMPLATES_GUIDE.md when templates change

---

## IDE Integration

### VSCode Snippet

Create `.vscode/snippets/template.code-snippets`:

```json
{
  "Template Variable": {
    "prefix": "tvar",
    "body": "{{${1:VARIABLE_NAME}}}",
    "description": "Template variable placeholder"
  },
  "Profile Template": {
    "prefix": "profile",
    "body": ["# Profile Template", "", "Copy from .metaHub/templates/profiles/"],
    "description": "Profile README template"
  }
}
```

### GitHub Codespaces

Create `.devcontainer/devcontainer.json`:

```json
{
  "name": "Template Editor",
  "image": "mcr.microsoft.com/devcontainers/universal:latest",
  "features": {
    "ghcr.io/devcontainers/features/github-cli:1": {}
  },
  "customizations": {
    "vscode": {
      "extensions": ["GitHub.copilot"]
    }
  }
}
```

---

## Contributing New Templates

To add new templates:

1. Create file: `.metaHub/templates/{category}/README_{TYPE}_TEMPLATE.md`
2. Use consistent `{{VARIABLE}}` naming
3. Document variables in this guide
4. Test with example substitution
5. Submit PR with examples

---

## Support

For template questions:
- Check variable definitions in this guide
- Review example profiles in `.metaHub/examples/`
- See governance contract for organization examples

---

**Last Updated:** 2025-11-26
**Maintainer:** Governance Team
