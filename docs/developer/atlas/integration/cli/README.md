# ORCHEX-KILO CLI Reference

The ORCHEX-KILO integration provides a unified command-line interface that combines ORCHEX analysis capabilities with KILO governance and DevOps automation. This reference covers all integrated commands and their usage.

## Command Structure

```bash
ORCHEX <command> [subcommand] [options] [arguments]
```

## Global Options

| Option            | Description                   | Default              |
| ----------------- | ----------------------------- | -------------------- |
| `--help`, `-h`    | Show help information         | -                    |
| `--version`, `-v` | Show version information      | -                    |
| `--config <file>` | Specify configuration file    | `ORCHEX.config.json` |
| `--verbose`       | Enable verbose output         | `false`              |
| `--quiet`         | Suppress non-error output     | `false`              |
| `--json`          | Output results in JSON format | `false`              |

## Analysis Commands

### `ORCHEX analyze repo`

Analyze a repository for code quality metrics with optional KILO governance validation.

```bash
ORCHEX analyze repo <path> [options]
```

**Arguments:**

- `path`: Path to the repository directory

**Options:**

- `--format <format>`: Output format (`table`, `json`, `summary`) (default: `table`)
- `--depth <depth>`: Analysis depth (`shallow`, `medium`, `deep`) (default: `medium`)
- `--include-patterns <patterns>`: File patterns to include (comma-separated)
- `--exclude-patterns <patterns>`: File patterns to exclude (comma-separated)
- `--governance-check`: Validate results against KILO governance policies
- `--auto-refactor`: Apply KILO-validated refactoring operations
- `--compliance-level <level>`: Compliance strictness (`lenient`, `standard`, `strict`)

**Examples:**

```bash
# Basic repository analysis
ORCHEX analyze repo ./my-project

# Deep analysis with governance validation
ORCHEX analyze repo ./my-project --depth deep --governance-check

# Analyze specific file types with JSON output
ORCHEX analyze repo ./my-project --include-patterns "*.ts,*.js" --format json

# Analysis with auto-refactoring
ORCHEX analyze repo ./my-project --auto-refactor --compliance-level strict
```

**Output:**

```
Repository Analysis Results
┌─────────────────┬─────────────┬─────────┐
│ Metric          │ Value       │ Status  │
├─────────────────┼─────────────┼─────────┤
│ Files Analyzed  │ 245         │ info    │
│ Total Lines     │ 15432       │ info    │
│ Complexity Score│ 3.2         │ warning │
│ Chaos Level     │ 2.1         │ good    │
│ Maintainability │ 7.8         │ good    │
└─────────────────┴─────────────┴─────────┘

Governance Check: PASSED
Compliance Score: 8.5/10
```

### `ORCHEX analyze complexity`

Analyze code complexity metrics with KILO validation.

```bash
ORCHEX analyze complexity <path> [options]
```

**Arguments:**

- `path`: Path to analyze

**Options:**

- `--threshold <number>`: Complexity threshold (default: 10)
- `--language <lang>`: Target programming language
- `--validate-policies`: Check against KILO complexity policies

**Examples:**

```bash
# Analyze complexity with default threshold
ORCHEX analyze complexity ./src

# Strict complexity analysis with policy validation
ORCHEX analyze complexity ./src --threshold 5 --validate-policies
```

### `ORCHEX analyze chaos`

Analyze code maintainability and chaos levels.

```bash
ORCHEX analyze chaos <path> [options]
```

**Arguments:**

- `path`: Path to analyze

**Options:**

- `--detailed`: Show detailed breakdown
- `--governance-integration`: Include KILO governance metrics

**Examples:**

```bash
# Basic chaos analysis
ORCHEX analyze chaos ./src

# Detailed analysis with governance integration
ORCHEX analyze chaos ./src --detailed --governance-integration
```

### `ORCHEX analyze scan`

Quick repository scan for basic metrics.

```bash
ORCHEX analyze scan <path> [options]
```

**Arguments:**

- `path`: Path to scan

**Options:**

- `--health-check`: Include repository health assessment
- `--governance-summary`: Include KILO governance summary

**Examples:**

```bash
# Quick scan
ORCHEX analyze scan ./my-project

# Health check with governance summary
ORCHEX analyze scan ./my-project --health-check --governance-summary
```

## Template Commands

### `ORCHEX template list`

List available KILO DevOps templates.

```bash
ORCHEX template list [category] [options]
```

**Arguments:**

- `category`: Template category (`cicd`, `db`, `iac`, `k8s`, `logging`, `monitoring`, `ui`)

**Options:**

- `--all`: List all templates across categories
- `--search <term>`: Search templates by name or description
- `--format <format>`: Output format (`table`, `json`, `list`)

**Examples:**

```bash
# List all CI/CD templates
ORCHEX template list cicd

# Search for database templates
ORCHEX template list db --search postgres

# List all templates in JSON format
ORCHEX template list --all --format json
```

### `ORCHEX template get`

Retrieve and generate a KILO DevOps template.

```bash
ORCHEX template get <category>/<name> [options]
```

**Arguments:**

- `category/name`: Template identifier (e.g., `cicd/github-actions`)

**Options:**

- `--version <version>`: Template version (default: `latest`)
- `--output <dir>`: Output directory (default: `./generated`)
- `--parameters <file>`: Parameter file (JSON)
- `--validate`: Validate generated files against policies
- `--apply`: Automatically apply the template

**Parameter Options:**

- `--param.<key>=<value>`: Set template parameters

**Examples:**

```bash
# Get GitHub Actions CI/CD template
ORCHEX template get cicd/github-actions

# Get specific version with custom parameters
ORCHEX template get cicd/github-actions --version 1.2.0 --param.nodeVersion=18

# Generate and validate template
ORCHEX template get k8s/deployment --validate --output ./k8s

# Use parameter file
ORCHEX template get db/postgres --parameters ./db-params.json
```

### `ORCHEX template validate`

Validate a template configuration.

```bash
ORCHEX template validate <category>/<name> [options]
```

**Arguments:**

- `category/name`: Template identifier

**Options:**

- `--parameters <file>`: Parameter file to validate
- `--strict`: Enable strict validation
- `--policy-check`: Validate against KILO policies

**Examples:**

```bash
# Validate template parameters
ORCHEX template validate cicd/github-actions --parameters ./ci-params.json

# Strict validation with policy check
ORCHEX template validate k8s/deployment --strict --policy-check
```

## Bridge Commands

### `ORCHEX bridge status`

Check the status of integration bridges.

```bash
ORCHEX bridge status [options]
```

**Options:**

- `--k2a`: Show only K2A bridge status
- `--a2k`: Show only A2K bridge status
- `--detailed`: Show detailed status information
- `--health-check`: Perform health checks

**Examples:**

```bash
# Overall bridge status
ORCHEX bridge status

# Detailed K2A bridge status
ORCHEX bridge status --k2a --detailed

# Health check for both bridges
ORCHEX bridge status --health-check
```

**Output:**

```
ORCHEX-KILO Bridge Status
========================

K2A Bridge (KILO → ORCHEX)
├── Status: Active
├── Events Processed: 1,247
├── Last Event: 2024-01-15T10:30:00Z
├── Error Count: 0
└── Health: Good

A2K Bridge (ORCHEX → KILO)
├── Status: Active
├── Validations: 892
├── Templates Served: 156
├── Cache Hit Rate: 94%
└── Health: Good
```

### `ORCHEX bridge configure`

Configure bridge settings.

```bash
ORCHEX bridge configure <bridge> [options]
```

**Arguments:**

- `bridge`: Bridge to configure (`k2a` or `a2k`)

**Options:**

- `--config <file>`: Configuration file
- `--set <key>=<value>`: Set configuration value
- `--reset`: Reset to default configuration

**Examples:**

```bash
# Configure K2A bridge
ORCHEX bridge configure k2a --set validation.strictness=strict

# Load configuration from file
ORCHEX bridge configure a2k --config ./bridge-config.json

# Reset A2K bridge to defaults
ORCHEX bridge configure a2k --reset
```

### `ORCHEX bridge test`

Test bridge connectivity and functionality.

```bash
ORCHEX bridge test [bridge] [options]
```

**Arguments:**

- `bridge`: Bridge to test (`k2a`, `a2k`, or both if omitted)

**Options:**

- `--comprehensive`: Run comprehensive tests
- `--performance`: Include performance benchmarks
- `--report <file>`: Save test report to file

**Examples:**

```bash
# Test both bridges
ORCHEX bridge test

# Comprehensive K2A bridge test
ORCHEX bridge test k2a --comprehensive

# Performance test with report
ORCHEX bridge test --performance --report ./bridge-test-report.json
```

## Compliance Commands

### `ORCHEX compliance check`

Check code compliance against KILO policies.

```bash
ORCHEX compliance check <path> [options]
```

**Arguments:**

- `path`: Path to check (file or directory)

**Options:**

- `--policies <list>`: Comma-separated list of policies to check
- `--language <lang>`: Programming language
- `--format <format>`: Output format (`table`, `json`, `summary`)
- `--fix`: Automatically fix violations (where possible)
- `--strict`: Treat warnings as errors

**Examples:**

```bash
# Check compliance for a file
ORCHEX compliance check ./src/auth.js --policies security,code_quality

# Check directory with auto-fix
ORCHEX compliance check ./src --fix

# Strict compliance check
ORCHEX compliance check ./src --strict --format json
```

### `ORCHEX compliance report`

Generate compliance reports.

```bash
ORCHEX compliance report [options]
```

**Options:**

- `--path <path>`: Target path (default: current directory)
- `--output <file>`: Output file
- `--format <format>`: Report format (`html`, `pdf`, `json`, `markdown`)
- `--period <days>`: Report period in days
- `--policies <list>`: Focus on specific policies

**Examples:**

```bash
# Generate HTML compliance report
ORCHEX compliance report --output ./compliance-report.html

# JSON report for last 30 days
ORCHEX compliance report --format json --period 30

# Security-focused report
ORCHEX compliance report --policies security --format pdf
```

## Workflow Commands

### `ORCHEX workflow run`

Execute predefined integrated workflows.

```bash
ORCHEX workflow run <workflow> [options]
```

**Arguments:**

- `workflow`: Workflow name

**Options:**

- `--config <file>`: Workflow configuration file
- `--param.<key>=<value>`: Workflow parameters
- `--dry-run`: Show what would be executed without running
- `--verbose`: Enable verbose output

**Examples:**

```bash
# Run code quality workflow
ORCHEX workflow run code-quality

# Run CI/CD setup workflow with parameters
ORCHEX workflow run cicd-setup --param.projectName=my-app --param.language=typescript

# Dry run deployment workflow
ORCHEX workflow run deploy --dry-run
```

### `ORCHEX workflow list`

List available workflows.

```bash
ORCHEX workflow list [options]
```

**Options:**

- `--category <category>`: Filter by category
- `--search <term>`: Search workflows
- `--format <format>`: Output format

**Examples:**

```bash
# List all workflows
ORCHEX workflow list

# List CI/CD workflows
ORCHEX workflow list --category cicd

# Search for security workflows
ORCHEX workflow list --search security
```

## Configuration Commands

### `ORCHEX config show`

Display current configuration.

```bash
ORCHEX config show [section] [options]
```

**Arguments:**

- `section`: Configuration section to show

**Options:**

- `--format <format>`: Output format (`json`, `yaml`, `table`)
- `--defaults`: Show default values
- `--effective`: Show effective configuration (with overrides)

**Examples:**

```bash
# Show all configuration
ORCHEX config show

# Show bridge configuration
ORCHEX config show bridges

# Show effective configuration in JSON
ORCHEX config show --effective --format json
```

### `ORCHEX config set`

Set configuration values.

```bash
ORCHEX config set <key> <value> [options]
```

**Arguments:**

- `key`: Configuration key
- `value`: Configuration value

**Options:**

- `--global`: Set global configuration
- `--local`: Set local configuration (default)
- `--type <type>`: Value type (`string`, `number`, `boolean`, `json`)

**Examples:**

```bash
# Set compliance level
ORCHEX config set compliance.level strict

# Set bridge timeout
ORCHEX config set bridges.a2k.timeoutMs 60000

# Set global verbose mode
ORCHEX config set global.verbose true --global
```

### `ORCHEX config reset`

Reset configuration to defaults.

```bash
ORCHEX config reset [section] [options]
```

**Arguments:**

- `section`: Configuration section to reset

**Options:**

- `--confirm`: Require confirmation before reset
- `--backup`: Create backup before reset

**Examples:**

```bash
# Reset all configuration
ORCHEX config reset --confirm

# Reset bridge configuration
ORCHEX config reset bridges

# Reset with backup
ORCHEX config reset --backup
```

## Utility Commands

### `ORCHEX init`

Initialize ORCHEX-KILO integration in a project.

```bash
ORCHEX init [options]
```

**Options:**

- `--template <template>`: Initialization template
- `--force`: Overwrite existing configuration
- `--skip-tests`: Skip integration tests

**Examples:**

```bash
# Initialize with default settings
ORCHEX init

# Initialize with custom template
ORCHEX init --template enterprise

# Force re-initialization
ORCHEX init --force
```

### `ORCHEX doctor`

Diagnose integration issues.

```bash
ORCHEX doctor [options]
```

**Options:**

- `--fix`: Automatically fix issues where possible
- `--report <file>`: Save diagnostic report
- `--verbose`: Show detailed diagnostic information

**Examples:**

```bash
# Run diagnostics
ORCHEX doctor

# Auto-fix issues
ORCHEX doctor --fix

# Generate detailed report
ORCHEX doctor --report ./ORCHEX-doctor-report.json --verbose
```

### `ORCHEX version`

Show version information for ORCHEX and KILO components.

```bash
ORCHEX version [options]
```

**Options:**

- `--all`: Show versions of all components
- `--check-updates`: Check for available updates
- `--format <format>`: Output format

**Examples:**

```bash
# Show current versions
ORCHEX version

# Check for updates
ORCHEX version --check-updates

# Show all component versions
ORCHEX version --all --format json
```

## Exit Codes

| Code | Description                |
| ---- | -------------------------- |
| 0    | Success                    |
| 1    | General error              |
| 2    | Configuration error        |
| 3    | Bridge communication error |
| 4    | Validation error           |
| 5    | Template error             |
| 6    | Compliance violation       |
| 10   | KILO service unavailable   |
| 11   | ORCHEX service unavailable |

## Environment Variables

| Variable             | Description                 | Default                |
| -------------------- | --------------------------- | ---------------------- |
| `ATLAS_CONFIG`       | Configuration file path     | `./ORCHEX.config.json` |
| `KILO_ENDPOINT`      | KILO service endpoint       | -                      |
| `ATLAS_BRIDGE_DEBUG` | Enable bridge debug logging | `false`                |
| `ATLAS_CACHE_DIR`    | Cache directory             | `~/.orchex/cache`      |
| `ATLAS_LOG_LEVEL`    | Logging level               | `info`                 |

## Examples

### Complete Development Workflow

```bash
# Initialize project
ORCHEX init

# Analyze codebase with governance
ORCHEX analyze repo . --governance-check --format json > analysis.json

# Check compliance
ORCHEX compliance check . --format summary

# Generate CI/CD pipeline
ORCHEX template get cicd/github-actions --param.nodeVersion=18 --apply

# Run integrated workflow
ORCHEX workflow run quality-gate

# Check bridge status
ORCHEX bridge status --health-check
```

### Automated CI/CD Integration

```yaml
# .github/workflows/ORCHEX-kilo-analysis.yml
name: ORCHEX-KILO Analysis
on: [push, pull_request]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run ORCHEX-KILO Analysis
        run: |
          ORCHEX analyze repo . --governance-check --auto-refactor
      - name: Generate Compliance Report
        run: ORCHEX compliance report --output compliance.html
      - name: Upload Report
        uses: actions/upload-artifact@v3
        with:
          name: compliance-report
          path: compliance.html
```

This CLI reference covers the complete integrated command set. For more detailed information about specific commands, use `ORCHEX <command> --help`.
