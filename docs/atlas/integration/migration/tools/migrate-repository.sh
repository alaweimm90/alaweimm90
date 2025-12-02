#!/bin/bash
# migrate-repository.sh
# Automated migration of individual repositories to ATLAS-KILO integration

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/../migration-config.json"
BACKUP_DIR="${MIGRATION_BACKUP_DIR:-./migration-backup}"
LOG_FILE="${SCRIPT_DIR}/migration.log"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Error handling
error() {
    log "ERROR: $*" >&2
    exit 1
}

# Validation
validate_prerequisites() {
    log "Validating prerequisites..."

    # Check if atlas CLI is available
    if ! command -v atlas &> /dev/null; then
        error "ATLAS CLI not found. Please install @atlas/cli"
    fi

    # Check if kilo CLI is available
    if ! command -v kilo &> /dev/null; then
        error "KILO CLI not found. Please install @kilo/cli"
    fi

    # Check if jq is available for JSON processing
    if ! command -v jq &> /dev/null; then
        error "jq not found. Please install jq for JSON processing"
    fi

    log "Prerequisites validation complete"
}

# Backup existing configuration
backup_configuration() {
    local repo_path="$1"
    log "Creating configuration backup for $repo_path"

    mkdir -p "$BACKUP_DIR"

    # Backup ATLAS config
    if [ -f "$repo_path/atlas.config.json" ]; then
        cp "$repo_path/atlas.config.json" "$BACKUP_DIR/atlas.config.$(date +%s).backup.json"
        log "ATLAS config backed up"
    fi

    # Backup KILO config
    if [ -f "$repo_path/kilo.config.json" ]; then
        cp "$repo_path/kilo.config.json" "$BACKUP_DIR/kilo.config.$(date +%s).backup.json"
        log "KILO config backed up"
    fi

    # Backup CI/CD configurations
    if [ -d "$repo_path/.github/workflows" ]; then
        mkdir -p "$BACKUP_DIR/cicd"
        cp -r "$repo_path/.github" "$BACKUP_DIR/cicd/"
        log "CI/CD configurations backed up"
    fi
}

# Apply integrated configuration
apply_integration_config() {
    local repo_path="$1"
    log "Applying integrated configuration to $repo_path"

    cd "$repo_path" || error "Cannot change to repository directory: $repo_path"

    # Create integrated configuration
    cat > atlas-kilo.config.json << EOF
{
  "integration": {
    "enabled": true,
    "version": "1.0",
    "bridges": {
      "k2a": {
        "enabled": true,
        "eventTypes": ["policy_violation", "security_issue"],
        "analysis": {
          "autoTrigger": true,
          "timeout": 300000
        }
      },
      "a2k": {
        "enabled": true,
        "validation": {
          "strictness": "standard",
          "timeout": 60000
        },
        "templates": {
          "cacheEnabled": true,
          "basePath": "./templates"
        }
      }
    },
    "endpoints": {
      "kilo": {
        "api": "${KILO_ENDPOINT:-https://kilo-api.company.com}",
        "apiKey": "${KILO_API_KEY}"
      }
    }
  }
}
EOF

    # Apply configuration
    atlas config apply atlas-kilo.config.json || error "Failed to apply integrated configuration"

    log "Integrated configuration applied successfully"
}

# Update CI/CD workflows
update_cicd_workflows() {
    local repo_path="$1"
    log "Updating CI/CD workflows in $repo_path"

    cd "$repo_path" || error "Cannot change to repository directory: $repo_path"

    # Check for GitHub Actions
    if [ -d ".github/workflows" ]; then
        log "Found GitHub Actions workflows"

        # Create integrated workflow
        cat > .github/workflows/integrated-analysis.yml << 'EOF'
name: Integrated ATLAS-KILO Analysis
on: [push, pull_request]

jobs:
  integrated-analysis:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Integration
        run: |
          npm install -g @atlas/cli @kilo/cli
          atlas config set kilo.endpoint ${{ secrets.KILO_ENDPOINT }}
          atlas config set kilo.apiKey ${{ secrets.KILO_API_KEY }}

      - name: Integrated Analysis
        run: atlas analyze repo . --governance-check --format json > analysis.json

      - name: Quality Gate
        run: |
          COMPLIANCE=$(jq '.complianceScore' analysis.json)
          if [ "$COMPLIANCE" -lt 7 ]; then
            echo "Quality gate failed: Score $COMPLIANCE < 7"
            exit 1
          fi

      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: analysis-results
          path: analysis.json
EOF

        log "Integrated GitHub Actions workflow created"
    fi

    # Check for other CI/CD systems
    if [ -f ".gitlab-ci.yml" ]; then
        log "GitLab CI detected - manual update required"
    fi

    if [ -f "Jenkinsfile" ]; then
        log "Jenkins detected - manual update required"
    fi
}

# Validate migration
validate_migration() {
    local repo_path="$1"
    log "Validating migration for $repo_path"

    cd "$repo_path" || error "Cannot change to repository directory: $repo_path"

    # Test bridge connectivity
    if ! atlas bridge test --quiet; then
        error "Bridge connectivity test failed"
    fi

    # Test integrated analysis
    if ! atlas analyze repo . --governance-check --format json --dry-run > /dev/null; then
        error "Integrated analysis test failed"
    fi

    # Test template access
    if ! atlas template list cicd --limit 1 --quiet > /dev/null; then
        error "Template access test failed"
    fi

    log "Migration validation successful"
}

# Main migration function
migrate_repository() {
    local repo_path="$1"
    local dry_run="${2:-false}"

    log "Starting migration for repository: $repo_path"
    log "Dry run mode: $dry_run"

    # Validate repository exists
    if [ ! -d "$repo_path" ]; then
        error "Repository path does not exist: $repo_path"
    fi

    # Check if it's a git repository
    if [ ! -d "$repo_path/.git" ]; then
        error "Not a git repository: $repo_path"
    fi

    if [ "$dry_run" = "true" ]; then
        log "Dry run complete - no changes made"
        return 0
    fi

    # Execute migration steps
    validate_prerequisites
    backup_configuration "$repo_path"
    apply_integration_config "$repo_path"
    update_cicd_workflows "$repo_path"
    validate_migration "$repo_path"

    log "Migration completed successfully for $repo_path"
}

# Main script execution
main() {
    local repo_path=""
    local dry_run="false"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                dry_run="true"
                shift
                ;;
            --help)
                echo "Usage: $0 <repository-path> [--dry-run]"
                echo ""
                echo "Migrate a repository to ATLAS-KILO integration"
                echo ""
                echo "Arguments:"
                echo "  repository-path    Path to the repository to migrate"
                echo ""
                echo "Options:"
                echo "  --dry-run         Show what would be done without making changes"
                echo "  --help           Show this help message"
                exit 0
                ;;
            *)
                if [ -z "$repo_path" ]; then
                    repo_path="$1"
                else
                    error "Unexpected argument: $1"
                fi
                shift
                ;;
        esac
    done

    # Validate required arguments
    if [ -z "$repo_path" ]; then
        error "Repository path is required. Use --help for usage information."
    fi

    # Run migration
    migrate_repository "$repo_path" "$dry_run"
}

# Run main function
main "$@"