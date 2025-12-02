# ATLAS-KILO CLI Reference

The ATLAS-KILO integration provides a unified command-line interface that combines ATLAS analysis capabilities with KILO governance and DevOps automation. This reference covers all integrated commands and their usage.

## Command Structure

```bash
atlas <command> [subcommand] [options] [arguments]
```

## Global Options

| Option            | Description                   | Default             |
| ----------------- | ----------------------------- | ------------------- |
| `--help`, `-h`    | Show help information         | -                   |
| `--version`, `-v` | Show version information      | -                   |
| `--config <file>` | Specify configuration file    | `atlas.config.json` |
| `--verbose`       | Enable verbose output         | `false`             |
| `--quiet`         | Suppress non-error output     | `false`             |
| `--json`          | Output results in JSON format | `false`             |

## Analysis Commands

### `atlas analyze repo`

Analyze a repository for code quality metrics with optional KILO governance validation.

```bash
atlas analyze repo <path> [options]
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
atlas analyze repo ./my-project

# Deep analysis with governance validation
atlas analyze repo ./my-project --depth deep --governance-check

# Analyze specific file types with JSON output
atlas analyze repo ./my-project --include-patterns "*.ts,*.js" --format json

# Analysis with auto-refactoring
atlas analyze repo ./my-project --auto-refactor --compliance-level strict
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

### `atlas analyze complexity`

Analyze code complexity metrics with KILO validation.

```bash
atlas analyze complexity <path> [options]
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
atlas analyze complexity ./src

# Strict complexity analysis with policy validation
atlas analyze complexity ./src --threshold 5 --validate-policies
```

### `atlas analyze chaos`

Analyze code maintainability and chaos levels.

```bash
atlas analyze chaos <path> [options]
```

**Arguments:**

- `path`: Path to analyze

**Options:**

- `--detailed`: Show detailed breakdown
- `--governance-integration`: Include KILO governance metrics

**Examples:**

```bash
# Basic chaos analysis
atlas analyze chaos ./src

# Detailed analysis with governance integration
atlas analyze chaos ./src --detailed --governance-integration
```

### `atlas analyze scan`

Quick repository scan for basic metrics.

```bash
atlas analyze scan <path> [options]
```

**Arguments:**

- `path`: Path to scan

**Options:**

- `--health-check`: Include repository health assessment
- `--governance-summary`: Include KILO governance summary

**Examples:**

```bash
# Quick scan
atlas analyze scan ./my-project

# Health check with governance summary
atlas analyze scan ./my-project --health-check --governance-summary
```

## Template Commands

### `atlas template list`

List available KILO DevOps templates.

```bash
atlas template list [category] [options]
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
atlas template list cicd

# Search for database templates
atlas template list db --search postgres

# List all templates in JSON format
atlas template list --all --format json
```

### `atlas template get`

Retrieve and generate a KILO DevOps template.

```bash
atlas template get <category>/<name> [options]
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
atlas template get cicd/github-actions

# Get specific version with custom parameters
atlas template get cicd/github-actions --version 1.2.0 --param.nodeVersion=18

# Generate and validate template
atlas template get k8s/deployment --validate --output ./k8s

# Use parameter file
atlas template get db/postgres --parameters ./db-params.json
```

### `atlas template validate`

Validate a template configuration.

```bash
atlas template validate <category>/<name> [options]
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
atlas template validate cicd/github-actions --parameters ./ci-params.json

# Strict validation with policy check
atlas template validate k8s/deployment --strict --policy-check
```

## Bridge Commands

### `atlas bridge status`

Check the status of integration bridges.

```bash
atlas bridge status [options]
```

**Options:**

- `--k2a`: Show only K2A bridge status
- `--a2k`: Show only A2K bridge status
- `--detailed`: Show detailed status information
- `--health-check`: Perform health checks

**Examples:**

```bash
# Overall bridge status
atlas bridge status

# Detailed K2A bridge status
atlas bridge status --k2a --detailed

# Health check for both bridges
atlas bridge status --health-check
```

**Output:**

```
ATLAS-KILO Bridge Status
========================

K2A Bridge (KILO → ATLAS)
├── Status: Active
├── Events Processed: 1,247
├── Last Event: 2024-01-15T10:30:00Z
├── Error Count: 0
└── Health: Good

A2K Bridge (ATLAS → KILO)
├── Status: Active
├── Validations: 892
├── Templates Served: 156
├── Cache Hit Rate: 94%
└── Health: Good
```

### `atlas bridge configure`

Configure bridge settings.

```bash
atlas bridge configure <bridge> [options]
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
atlas bridge configure k2a --set validation.strictness=strict

# Load configuration from file
atlas bridge configure a2k --config ./bridge-config.json

# Reset A2K bridge to defaults
atlas bridge configure a2k --reset
```

### `atlas bridge test`

Test bridge connectivity and functionality.

```bash
atlas bridge test [bridge] [options]
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
atlas bridge test

# Comprehensive K2A bridge test
atlas bridge test k2a --comprehensive

# Performance test with report
atlas bridge test --performance --report ./bridge-test-report.json
```

## Compliance Commands

### `atlas compliance check`

Check code compliance against KILO policies.

```bash
atlas compliance check <path> [options]
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
atlas compliance check ./src/auth.js --policies security,code_quality

# Check directory with auto-fix
atlas compliance check ./src --fix

# Strict compliance check
atlas compliance check ./src --strict --format json
```

### `atlas compliance report`

Generate compliance reports.

```bash
atlas compliance report [options]
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
atlas compliance report --output ./compliance-report.html

# JSON report for last 30 days
atlas compliance report --format json --period 30

# Security-focused report
atlas compliance report --policies security --format pdf
```

## Workflow Commands

### `atlas workflow run`

Execute predefined integrated workflows.

```bash
atlas workflow run <workflow> [options]
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
atlas workflow run code-quality

# Run CI/CD setup workflow with parameters
atlas workflow run cicd-setup --param.projectName=my-app --param.language=typescript

# Dry run deployment workflow
atlas workflow run deploy --dry-run
```

### `atlas workflow list`

List available workflows.

```bash
atlas workflow list [options]
```

**Options:**

- `--category <category>`: Filter by category
- `--search <term>`: Search workflows
- `--format <format>`: Output format

**Examples:**

```bash
# List all workflows
atlas workflow list

# List CI/CD workflows
atlas workflow list --category cicd

# Search for security workflows
atlas workflow list --search security
```

## Configuration Commands

### `atlas config show`

Display current configuration.

```bash
atlas config show [section] [options]
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
atlas config show

# Show bridge configuration
atlas config show bridges

# Show effective configuration in JSON
atlas config show --effective --format json
```

### `atlas config set`

Set configuration values.

```bash
atlas config set <key> <value> [options]
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
atlas config set compliance.level strict

# Set bridge timeout
atlas config set bridges.a2k.timeoutMs 60000

# Set global verbose mode
atlas config set global.verbose true --global
```

### `atlas config reset`

Reset configuration to defaults.

```bash
atlas config reset [section] [options]
```

**Arguments:**

- `section`: Configuration section to reset

**Options:**

- `--confirm`: Require confirmation before reset
- `--backup`: Create backup before reset

**Examples:**

```bash
# Reset all configuration
atlas config reset --confirm

# Reset bridge configuration
atlas config reset bridges

# Reset with backup
atlas config reset --backup
```

## Utility Commands

### `atlas init`

Initialize ATLAS-KILO integration in a project.

```bash
atlas init [options]
```

**Options:**

- `--template <template>`: Initialization template
- `--force`: Overwrite existing configuration
- `--skip-tests`: Skip integration tests

**Examples:**

```bash
# Initialize with default settings
atlas init

# Initialize with custom template
atlas init --template enterprise

# Force re-initialization
atlas init --force
```

### `atlas doctor`

Diagnose integration issues.

```bash
atlas doctor [options]
```

**Options:**

- `--fix`: Automatically fix issues where possible
- `--report <file>`: Save diagnostic report
- `--verbose`: Show detailed diagnostic information

**Examples:**

```bash
# Run diagnostics
atlas doctor

# Auto-fix issues
atlas doctor --fix

# Generate detailed report
atlas doctor --report ./atlas-doctor-report.json --verbose
```

### `atlas version`

Show version information for ATLAS and KILO components.

```bash
atlas version [options]
```

**Options:**

- `--all`: Show versions of all components
- `--check-updates`: Check for available updates
- `--format <format>`: Output format

**Examples:**

```bash
# Show current versions
atlas version

# Check for updates
atlas version --check-updates

# Show all component versions
atlas version --all --format json
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
| 11   | ATLAS service unavailable  |

## Environment Variables

| Variable             | Description                 | Default               |
| -------------------- | --------------------------- | --------------------- |
| `ATLAS_CONFIG`       | Configuration file path     | `./atlas.config.json` |
| `KILO_ENDPOINT`      | KILO service endpoint       | -                     |
| `ATLAS_BRIDGE_DEBUG` | Enable bridge debug logging | `false`               |
| `ATLAS_CACHE_DIR`    | Cache directory             | `~/.atlas/cache`      |
| `ATLAS_LOG_LEVEL`    | Logging level               | `info`                |

## Examples

### Complete Development Workflow

```bash
# Initialize project
atlas init

# Analyze codebase with governance
atlas analyze repo . --governance-check --format json > analysis.json

# Check compliance
atlas compliance check . --format summary

# Generate CI/CD pipeline
atlas template get cicd/github-actions --param.nodeVersion=18 --apply

# Run integrated workflow
atlas workflow run quality-gate

# Check bridge status
atlas bridge status --health-check
```

### Automated CI/CD Integration

```yaml
# .github/workflows/atlas-kilo-analysis.yml
name: ATLAS-KILO Analysis
on: [push, pull_request]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run ATLAS-KILO Analysis
        run: |
          atlas analyze repo . --governance-check --auto-refactor
      - name: Generate Compliance Report
        run: atlas compliance report --output compliance.html
      - name: Upload Report
        uses: actions/upload-artifact@v3
        with:
          name: compliance-report
          path: compliance.html
```

This CLI reference covers the complete integrated command set. For more detailed information about specific commands, use `atlas <command> --help`.
