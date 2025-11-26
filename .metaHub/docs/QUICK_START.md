# Quick Start Guide

Get up and running with the portfolio governance system in 5 minutes.

## Installation

```bash
# Install dependencies
pip install -r .metaHub/scripts/requirements.txt

# Make scripts executable
chmod +x .metaHub/scripts/*.py
```

## Common Tasks

### 1. Check Repository Compliance Locally

```bash
# Run enforcement on current repo
python .metaHub/scripts/enforce.py .

# Run in strict mode (warnings fail)
python .metaHub/scripts/enforce.py . --strict

# Generate JSON report
python .metaHub/scripts/enforce.py . --report json
```

### 2. View Portfolio Catalog

```bash
# Generate current portfolio catalog
python .metaHub/scripts/catalog.py

# View catalog.json
cat .metaHub/catalog/catalog.json | jq '.organizations[] | {name, repo_count}'
```

### 3. Create Weekly Checkpoint

```bash
# Generate weekly drift report
python .metaHub/scripts/checkpoint.py

# View latest drift report
cat .metaHub/catalog/weekly-reports/latest-drift.md
```

### 4. Add Metadata to Repository

```bash
# Create .meta/repo.yaml in your repository
mkdir -p .meta
cat > .meta/repo.yaml << 'YAML'
type: app
language: python
tier: 2
owner: your-team

YAML

# Validate the metadata
python .metaHub/scripts/enforce.py .
```

## Understanding Reports

### Enforcement Report (JSON)

```json
{
  "status": "pass|fail",
  "violations": ["list of failing checks"],
  "warnings": ["list of warnings"],
  "summary": {
    "violations": 0,
    "warnings": 2
  }
}
```

**Exit Codes:**
- `0` = Success (or warnings in non-strict mode)
- `1` = Violations found

### Catalog Report

Shows all repositories in organizations/:
- Total repos discovered
- Repos with metadata
- Repos with violations
- Compliance percentage

### Drift Report

Shows changes from previous week:
- New repositories added
- Repositories archived/deleted
- Status changes (compliant → violation, etc.)
- Intentional vs unintentional drift

## Troubleshooting

### "enforce.py not found"

```bash
# Make sure you're in the repository root
cd /path/to/governance-contract

# Check script exists
ls -l .metaHub/scripts/enforce.py
```

### "Missing .meta/repo.yaml"

This is optional for the governance contract itself, but required for consumer repos:

```bash
# Consumer repos should have:
ls .meta/repo.yaml
```

### "Python modules not found"

```bash
# Install requirements
pip install -r .metaHub/scripts/requirements.txt

# Verify
python -c "import yaml; print('OK')"
```

### "Permission denied"

```bash
# Make scripts executable
chmod +x .metaHub/scripts/*.py
```

## Integration Examples

### Pre-commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
python .metaHub/scripts/enforce.py . --strict
```

### CI/CD Pipeline

In `.github/workflows/ci.yml`:

```yaml
- name: Enforce governance
  run: python .metaHub/scripts/enforce.py . --report json --output enforcement-report.json

- name: Upload report
  uses: actions/upload-artifact@v3
  with:
    name: enforcement-report
    path: enforcement-report.json
```

### Container Build

Validate before building Docker image:

```bash
python .metaHub/scripts/enforce.py . || exit 1
docker build -t myapp:latest .
```

## Next Steps

1. **Read** [ARCHITECTURE.md](./ARCHITECTURE.md) for system design
2. **Review** [Consumer Guide](./.metaHub/guides/consumer-guide.md) for governance adoption
3. **Explore** [Example Repo](./.metaHub/examples/consumer-repo/) to see working implementation
4. **Configure** your repository with `.meta/repo.yaml`
5. **Integrate** enforce.py into your CI/CD

## Key Files

| File | Purpose |
|------|---------|
| `.metaHub/scripts/enforce.py` | Enforcement validation |
| `.metaHub/scripts/catalog.py` | Portfolio cataloging |
| `.metaHub/scripts/checkpoint.py` | Drift detection |
| `.metaHub/catalog/catalog.json` | Current catalog |
| `.metaHub/catalog/weekly-reports/` | Drift archives |
| `ARCHITECTURE.md` | System design |

## Support

- **Questions about policies?** See [.metaHub/policies/README.md](./.metaHub/policies/README.md)
- **Questions about metadata?** See [.metaHub/schemas/README.md](./.metaHub/schemas/README.md)
- **Need examples?** See [.metaHub/examples/consumer-repo/](./.metaHub/examples/consumer-repo/)

---

**Last Updated:** 2025-11-26
