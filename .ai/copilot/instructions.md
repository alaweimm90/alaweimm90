# GitHub Copilot Instructions

## Project Context
This is a multi-organization GitHub portfolio governance system.

### Organizations Managed
- alaweimm90-tools
- alaweimm90-science
- alaweimm90-internal
- alaweimm90-oss
- alaweimm90-labs

### Architecture
- Central governance repo with templates and policies
- Organization monorepos synced via push_monorepos.py
- ADR documentation in docs/adr/

## Code Standards

### Python
- Version: 3.11+
- Type hints required
- Use pathlib for file operations
- Click for CLI interfaces
- PyYAML for configuration

### File Organization
- Scripts: .metaHub/scripts/
- Policies: .metaHub/policies/
- Templates: .metaHub/templates/
- Documentation: docs/

## Preferences

### Auto-Approve Actions
- Apply suggested code changes
- Create/modify files in standard locations
- Run read-only git commands
- Execute Python scripts

### Response Style
- Concise, actionable responses
- Show code, not explanations
- Prefer editing existing files over creating new ones

## Key Files to Reference
- .metaHub/policies/root-structure.yaml - Root directory policy
- .metaHub/templates/structures/portfolio-structure.yaml - Structure templates
- docs/adr/ADR-001-organization-monorepo-architecture.md
- docs/adr/ADR-002-root-structure-policy.md