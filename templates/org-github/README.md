# Organization .github Template

This template creates a `.github` repository for GitHub organizations, providing:

- **Organization profile** (`profile/README.md`)
- **Issue templates** (bug reports, feature requests)
- **Pull request template**
- **Security policy**
- **Workflow templates** (starter workflows for new repos)

## Usage

### 1. Create the .github repo

```bash
# In your organization
gh repo create ORG_NAME/.github --public
```

### 2. Copy template files

```bash
# From this repo
cp -r templates/org-github/* /path/to/.github/
```

### 3. Replace placeholders

| Placeholder | Replace with |
|-------------|-------------|
| `{{ORG_NAME}}` | Your org name (e.g., `AlaweinLabs`) |
| `{{ORG_DESCRIPTION}}` | One-line org description |
| `{{ORG_DOMAIN}}` | Domain for security emails |

### 4. Push to GitHub

```bash
cd /path/to/.github
git add .
git commit -m "feat: initialize org .github repo"
git push
```

## What Each File Does

| File | Purpose |
|------|---------|
| `profile/README.md` | Shown on org's GitHub profile page |
| `ISSUE_TEMPLATE/*.md` | Default issue templates for all repos |
| `PULL_REQUEST_TEMPLATE.md` | Default PR template for all repos |
| `SECURITY.md` | Security policy shown on all repos |
| `workflow-templates/*.yml` | Starter workflows in "Actions" tab |

## Inheritance

Files in `.github` repo apply to ALL repos in the org unless overridden:

```
org/.github/                    # Org-wide defaults
├── ISSUE_TEMPLATE/
└── PULL_REQUEST_TEMPLATE.md

org/some-repo/                  # Can override
├── .github/
│   └── ISSUE_TEMPLATE/        # Overrides org template
└── ...
```

## Related

- [Architecture](../../docs/ARCHITECTURE.md)
- [Reusable Workflows](../../.github/workflows/)
