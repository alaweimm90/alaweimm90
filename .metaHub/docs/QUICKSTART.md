# Quick Start Guide

Get up and running with the alaweimm90-tools toolkit in 5 minutes.

## Prerequisites

- Bash 4.0+
- Python 3.9+
- Node.js 18+
- jq
- Git

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/alaweimm90/alaweimm90-tools.git
cd alaweimm90-tools

# 2. Install dependencies
pip install pyyaml requests  # Python basics
npm install                   # Node dependencies

# 3. Make scripts executable
chmod +x bin/* tools/**/*.sh tools/**/*.py

# 4. Add to PATH (optional but recommended)
echo 'export PATH="$PATH:$HOME/alaweimm90-tools/bin"' >> ~/.bashrc
source ~/.bashrc
```

## Verify Installation

```bash
# Check toolkit is accessible
toolkit --help

# List available tools
ls tools/
```

## Your First Commands

### AI Task Routing

```bash
# Route a task to the best AI tool
toolkit ai route "Create a REST API with authentication"

# See routing explanation
toolkit ai route --explain "Fix memory leak in auth module"

# View dashboard
toolkit ai dashboard
```

### Governance

```bash
# Run compliance validation
toolkit governance validate

# Check structure
python tools/governance/structure_validator.py
```

### DevOps Templates

```bash
# List available templates
toolkit devops list

# Generate from template
toolkit devops build --template=k8s
```

### Security Scanning

```bash
# Run all security scans
toolkit security scan-all

# Individual scans
./tools/security/secret-scan.sh
./tools/security/dependency-scan.sh
```

## Common Workflows

### New Project Setup

```bash
# 1. Create project structure
toolkit devops build --template=demo-k8s-node

# 2. Set up CI/CD
python tools/automation/setup_repo_ci.py

# 3. Validate compliance
toolkit governance validate
```

### Daily Development

```bash
# Route your task
toolkit ai route "Add user authentication"

# Check costs
toolkit ai cost report

# Run tests
./tools/ai-orchestration/test-runner.sh
```

### Security Audit

```bash
# Full security scan
toolkit security scan-all

# Container scanning
./tools/security/trivy-scan.sh

# Secret detection
./tools/security/secret-scan.sh
```

## Next Steps

- Read [CATALOG.md](CATALOG.md) for full tool list
- Explore [docs/guides/](docs/guides/) for detailed documentation
- Check [templates/](templates/) for infrastructure templates

## Getting Help

```bash
# Tool-specific help
./tools/ai-orchestration/task-router.sh --help
python tools/governance/compliance_validator.py --help

# View catalog
cat CATALOG.md
```
