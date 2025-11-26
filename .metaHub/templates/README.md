# README Templates & Documentation Generator

Complete template system for generating customized documentation across your portfolio.

## What's Included

### Templates

1. **[README_PROFILE_TEMPLATE.md](./profiles/README_PROFILE_TEMPLATE.md)**
   - Personal GitHub profile README
   - Features: Projects, skills, stats, philosophy
   - Use case: Your GitHub profile
   - Variables: 100+ customizable fields

2. **[README_ORG_TEMPLATE.md](./organizations/README_ORG_TEMPLATE.md)**
   - Organization/collective profile README
   - Features: Teams, repositories, governance, compliance
   - Use case: Organization profile
   - Variables: 80+ customizable fields

3. **[README_CONSUMER_TEMPLATE.md](./consumer-repos/README_CONSUMER_TEMPLATE.md)**
   - Individual repository README
   - Features: Docs, setup, deployment, governance, compliance
   - Use case: Any consumer repository
   - Variables: 120+ customizable fields

### Tools

- **[template_generator.py](./template_generator.py)** — Automated template substitution
- **[TEMPLATES_GUIDE.md](./TEMPLATES_GUIDE.md)** — Complete variable reference
- **[examples/PROFILE_VARIABLES.json](./examples/PROFILE_VARIABLES.json)** — Example configuration

---

## Quick Start

### Option 1: Interactive Mode (Easiest)

```bash
# Generate your profile README interactively
python template_generator.py profile --interactive --output README.md

# Follow prompts for each variable
# Your custom README.md is created!
```

### Option 2: Configuration File

```bash
# Copy example variables
cp examples/PROFILE_VARIABLES.json my-vars.json

# Edit my-vars.json with your values
# Then generate:
python template_generator.py profile --config my-vars.json --output README.md
```

### Option 3: Manual Editing

```bash
# Copy template
cp profiles/README_PROFILE_TEMPLATE.md README.md

# Edit README.md and replace {{VARIABLES}} manually
# Find and replace using your editor
```

---

## Available Templates

### Profile Template

**For:** Personal GitHub profiles

**Command:**
```bash
python template_generator.py profile --interactive --output README.md
```

**Example:** [examples/PROFILE_VARIABLES.json](./examples/PROFILE_VARIABLES.json)

**Key Sections:**
- Profile header and contact
- About section with mission and philosophy
- Featured projects (4-6 projects)
- Expertise and tech stack
- GitHub statistics
- Philosophy and values
- Current work and learning
- Fun facts and easter eggs
- Contact information

**Variables:** 100+

---

### Organization Template

**For:** Organization/collective READMEs

**Command:**
```bash
python template_generator.py org --interactive --output README.md
```

**Key Sections:**
- Organization identity and mission
- Core teams and structure
- Repository catalog
- Governance framework
- Tech stack
- Compliance and standards
- Contact and leadership

**Variables:** 80+

---

### Consumer Repository Template

**For:** Individual repository READMEs

**Command:**
```bash
python template_generator.py consumer --interactive --output README.md
```

**Key Sections:**
- Project overview and features
- Quick start and installation
- Documentation links
- Repository structure
- Governance metadata
- Development guide
- Deployment instructions
- Testing and quality
- Roadmap and issues
- Contributing guidelines
- Support and contact

**Variables:** 120+

---

## Configuration Files

### YAML Format

Create `variables.yaml`:

```yaml
FULL_NAME: "Meshal Alawein"
GITHUB_USERNAME: "alaweimm90"
EMAIL: "meshal@berkeley.edu"
PRIMARY_COLOR: "A855F7"
# ... more variables
```

Generate:
```bash
python template_generator.py profile --config variables.yaml --output README.md
```

### JSON Format

Create `variables.json`:

```json
{
  "FULL_NAME": "Meshal Alawein",
  "GITHUB_USERNAME": "alaweimm90",
  "EMAIL": "meshal@berkeley.edu",
  "PRIMARY_COLOR": "A855F7"
}
```

Generate:
```bash
python template_generator.py profile --config variables.json --output README.md
```

### Environment Variables

Create `.env`:

```bash
export FULL_NAME="Meshal Alawein"
export GITHUB_USERNAME="alaweimm90"
export EMAIL="meshal@berkeley.edu"
export PRIMARY_COLOR="A855F7"
```

Generate with bash:
```bash
source .env
envsubst < profiles/README_PROFILE_TEMPLATE.md > README.md
```

---

## Template Variable Categories

### Identity & Contact
- `FULL_NAME` — Your full name
- `GITHUB_USERNAME` — GitHub username
- `EMAIL` — Email address
- `WEBSITE_URL` — Portfolio/website URL

### Branding
- `PRIMARY_COLOR` — Main color (hex, e.g., A855F7)
- `SECONDARY_COLOR` — Secondary color
- `ACCENT_COLOR` — Accent color

### Content
- `PROFESSIONAL_DESCRIPTOR` — Your job title/role
- `ELEVATOR_PITCH` — Short introduction (1-2 sentences)
- `BACKGROUND_STORY` — Your background
- `PERSONAL_PHILOSOPHY` — Your approach/philosophy

### Projects
- `PROJECT_N_NAME` — Project name
- `PROJECT_N_TAGLINE` — One-liner
- `PROJECT_N_DESCRIPTION` — Full description
- `PROJECT_N_TECH` — Technologies used
- `PROJECT_N_STATUS` — Active, Beta, Archived

### Skills
- `EXPERTISE_N` — Skill name
- `EXPERTISE_N_PCT` — Percentage (0-100)
- `EXPERTISE_N_BAR` — ASCII bar (████░░)

### Current Work
- `CURRENT_RESEARCH` — Current focus
- `LEARNING_GOAL` — What you're learning
- `READING_MATERIAL` — Current reading

See [TEMPLATES_GUIDE.md](./TEMPLATES_GUIDE.md) for **complete variable reference**.

---

## Generator Features

### Validation

The generator checks for:
- **Missing variables:** Variables in template but not in configuration
- **Unused variables:** Variables in configuration but not in template

Example:
```
⚠️  Missing variables: FULL_NAME, EMAIL
ℹ️  Unused variables: EXTRA_FIELD
```

### Interactive Prompts

When using `--interactive`, the generator:
1. Shows template name
2. Lists number of variables
3. Prompts for each variable
4. Skips empty responses
5. Generates output file

### Customization

The generator preserves:
- Markdown formatting
- Links and references
- Code blocks and syntax highlighting
- Badge definitions
- HTML formatting

---

## Use Cases

### Scenario 1: Personal Profile

```bash
# Generate your GitHub profile README
python template_generator.py profile --interactive --output README.md

# Your profile is now ready!
git add README.md
git commit -m "feat(profile): customize personal README"
```

### Scenario 2: Organization Setup

```bash
# Create organization variables
cat > org-vars.json << EOF
{
  "ORG_NAME": "My Org",
  "ORG_DESCRIPTION": "Description here",
  "REPO_COUNT": "15",
  ...
}
EOF

# Generate org README
python template_generator.py org --config org-vars.json --output README.md
```

### Scenario 3: Consumer Repo

```bash
# Copy template and customize
cp consumer-repos/README_CONSUMER_TEMPLATE.md README.md

# Or use generator
python template_generator.py consumer --interactive --output README.md

# Customize repo metadata
cat > .meta/repo.yaml << EOF
type: lib
language: python
tier: 2
EOF
```

---

## Best Practices

1. **Start with examples** — Copy `examples/PROFILE_VARIABLES.json` as template
2. **Use consistent colors** — Pick 3-4 colors and use throughout
3. **Keep descriptions concise** — 2-3 sentences maximum for elevator pitch
4. **Update regularly** — Quarterly refresh recommended
5. **Version your config** — Track variables in git (consider secrets handling)
6. **Test rendering** — Preview on GitHub before final commit
7. **Reference governance** — Always link to governance contract

---

## IDE Integration

### VSCode

Add to `.vscode/settings.json`:

```json
{
  "[markdown]": {
    "editor.defaultFormatter": "prettier.prettier",
    "editor.formatOnSave": true
  }
}
```

Add to `.vscode/extensions.json`:

```json
{
  "recommendations": [
    "DavidAnson.vscode-markdownlint",
    "yzhang.markdown-all-in-one"
  ]
}
```

### GitHub Codespaces

Create `.devcontainer/devcontainer.json`:

```json
{
  "image": "mcr.microsoft.com/devcontainers/universal:latest",
  "customizations": {
    "vscode": {
      "extensions": ["yzhang.markdown-all-in-one"]
    }
  }
}
```

---

## Advanced Usage

### Custom Template

Create `templates/custom/README_CUSTOM.md` with your own `{{VARIABLES}}`.

Use generator:
```bash
python template_generator.py custom --interactive --output README.md
```

### Batch Generation

```bash
# Generate for multiple repos
for repo in repo1 repo2 repo3; do
  echo "Generating $repo..."
  python template_generator.py consumer \
    --config "configs/$repo.json" \
    --output "$repo/README.md"
done
```

### CI/CD Integration

GitHub Actions workflow:

```yaml
- name: Generate READMEs
  run: |
    python .metaHub/templates/template_generator.py profile \
      --config variables.json \
      --output README.md

    git add README.md
    git commit -m "chore: update README from template"
    git push
```

---

## Troubleshooting

### Missing variables warning

```
⚠️  Missing variables: FULL_NAME
```

**Solution:** Add missing variables to config file or provide in interactive prompt.

### Template not found

```
Error: Template not found: profiles/README_PROFILE_TEMPLATE.md
```

**Solution:** Ensure you're running from repo root where templates exist.

### Encoding issues

```
UnicodeDecodeError: 'utf-8' codec can't decode
```

**Solution:** Ensure config file is UTF-8 encoded.

---

## Contributing

To add new templates:

1. Create file: `templates/category/README_TYPE_TEMPLATE.md`
2. Use consistent `{{VARIABLE}}` naming
3. Document variables in [TEMPLATES_GUIDE.md](./TEMPLATES_GUIDE.md)
4. Add example in `examples/`
5. Update this README
6. Submit PR

---

## Reference

- **[TEMPLATES_GUIDE.md](./TEMPLATES_GUIDE.md)** — Complete variable documentation
- **[Governance Contract](../)** — Governance policies and standards
- **[Examples](./examples/)** — Working configuration examples

---

## Support

For questions:
- Check variable definitions in [TEMPLATES_GUIDE.md](./TEMPLATES_GUIDE.md)
- Review example configs in `examples/`
- See governance contract for standards

---

**Last Updated:** 2025-11-26
**Status:** Production-ready
**Maintained by:** Governance Team
