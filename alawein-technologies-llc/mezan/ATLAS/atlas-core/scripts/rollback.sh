#!/bin/bash
# Rollback script for MEZAN ATLAS deployments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
ROLLBACK_TO=${1:-"previous"}
BACKUP_DIR=${BACKUP_DIR:-"/backup"}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-""}
HEALTH_CHECK_URL=${HEALTH_CHECK_URL:-"http://localhost:8080/health"}
SLACK_WEBHOOK=${SLACK_WEBHOOK:-""}
DRY_RUN=${DRY_RUN:-false}

# State tracking
CURRENT_VERSION=""
TARGET_VERSION=""
ROLLBACK_SUCCESSFUL=false

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

send_notification() {
    local message="$1"
    local status="$2"

    if [ -n "$SLACK_WEBHOOK" ]; then
        local color="warning"
        [ "$status" = "success" ] && color="good"
        [ "$status" = "error" ] && color="danger"

        curl -X POST -H 'Content-type: application/json' \
            --data "{
                \"text\":\"ATLAS Rollback\",
                \"attachments\": [{
                    \"color\": \"${color}\",
                    \"fields\": [
                        {\"title\": \"Action\", \"value\": \"Rollback\", \"short\": true},
                        {\"title\": \"Status\", \"value\": \"${status}\", \"short\": true},
                        {\"title\": \"From Version\", \"value\": \"${CURRENT_VERSION}\", \"short\": true},
                        {\"title\": \"To Version\", \"value\": \"${TARGET_VERSION}\", \"short\": true},
                        {\"title\": \"Message\", \"value\": \"${message}\"},
                        {\"title\": \"Time\", \"value\": \"$(date)\"}
                    ]
                }]
            }" \
            "$SLACK_WEBHOOK" 2>/dev/null || true
    fi
}

get_current_version() {
    log_info "Getting current version..."

    # Get version from running container
    CURRENT_VERSION=$(docker inspect atlas-api 2>/dev/null | \
        grep -o '"org.opencontainers.image.version": "[^"]*' | \
        cut -d'"' -f4) || CURRENT_VERSION="unknown"

    if [ -z "$CURRENT_VERSION" ] || [ "$CURRENT_VERSION" = "unknown" ]; then
        # Try to get from image tag
        CURRENT_VERSION=$(docker-compose ps atlas-api 2>/dev/null | \
            grep atlas-api | awk '{print $2}' | cut -d: -f2) || CURRENT_VERSION="unknown"
    fi

    log_info "Current version: ${CURRENT_VERSION}"
}

list_available_versions() {
    log_info "Available versions for rollback:"

    # List local Docker images
    echo -e "${BLUE}Local Docker images:${NC}"
    docker images --filter "reference=*atlas*" --format "table {{.Repository}}\t{{.Tag}}\t{{.CreatedAt}}\t{{.Size}}"

    # List available backups
    echo -e "\n${BLUE}Available backup files:${NC}"
    ls -lht ${BACKUP_DIR}/atlas-backup-* 2>/dev/null | head -10 || echo "No backups found"

    # List tags from registry if configured
    if [ -n "$DOCKER_REGISTRY" ]; then
        echo -e "\n${BLUE}Registry tags:${NC}"
        # This would require registry API access
        echo "Registry listing not implemented"
    fi
}

validate_target_version() {
    local target="$1"

    log_info "Validating target version: ${target}"

    # Check if it's a valid Docker image
    if docker image inspect "mezan-atlas:${target}" >/dev/null 2>&1; then
        TARGET_VERSION="$target"
        return 0
    fi

    # Check if it's a backup timestamp
    if [ -f "${BACKUP_DIR}/atlas-backup-${target}-manifest.json" ]; then
        TARGET_VERSION="backup-${target}"
        return 0
    fi

    # Check if it's "previous" (special case)
    if [ "$target" = "previous" ]; then
        # Get the second most recent image
        TARGET_VERSION=$(docker images --filter "reference=mezan-atlas" --format "{{.Tag}}" | \
            grep -v latest | head -2 | tail -1)

        if [ -z "$TARGET_VERSION" ]; then
            log_error "No previous version found"
            return 1
        fi
        return 0
    fi

    log_error "Target version not found: ${target}"
    return 1
}

create_rollback_backup() {
    log_info "Creating backup of current state before rollback..."

    local backup_name="rollback-backup-$(date +%Y%m%d-%H%M%S)"

    # Quick backup of critical data
    docker-compose exec -T postgres pg_dump -U atlas atlas > \
        "${BACKUP_DIR}/${backup_name}-db.sql" 2>/dev/null || true

    docker-compose exec -T redis redis-cli SAVE 2>/dev/null || true
    docker cp atlas-redis:/data/atlas.rdb "${BACKUP_DIR}/${backup_name}-redis.rdb" 2>/dev/null || true

    # Save current container state
    docker-compose ps > "${BACKUP_DIR}/${backup_name}-state.txt"

    log_info "Rollback backup created: ${backup_name}"
}

stop_services() {
    log_info "Stopping current services..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would stop services"
        return 0
    fi

    docker-compose stop
    log_info "Services stopped"
}

rollback_docker_images() {
    log_info "Rolling back to Docker image version: ${TARGET_VERSION}"

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would rollback to ${TARGET_VERSION}"
        return 0
    fi

    # Update docker-compose.yml or .env file with target version
    if [ -f ".env" ]; then
        sed -i.bak "s/IMAGE_TAG=.*/IMAGE_TAG=${TARGET_VERSION}/" .env
    fi

    # Tag the target version as latest for docker-compose
    docker tag "mezan-atlas:${TARGET_VERSION}" "mezan-atlas:latest"

    log_info "Docker images updated to version: ${TARGET_VERSION}"
}

rollback_from_backup() {
    local backup_timestamp="$1"
    log_info "Rolling back from backup: ${backup_timestamp}"

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would restore from backup ${backup_timestamp}"
        return 0
    fi

    # Restore database
    if [ -f "${BACKUP_DIR}/atlas-backup-${backup_timestamp}-postgres.sql" ]; then
        log_info "Restoring database..."
        docker-compose exec -T postgres psql -U atlas -d postgres -c "DROP DATABASE IF EXISTS atlas;"
        docker-compose exec -T postgres psql -U atlas -d postgres -c "CREATE DATABASE atlas;"
        docker-compose exec -T postgres psql -U atlas atlas < \
            "${BACKUP_DIR}/atlas-backup-${backup_timestamp}-postgres.sql"
    elif [ -f "${BACKUP_DIR}/atlas-backup-${backup_timestamp}-postgres.sql.gz" ]; then
        log_info "Restoring compressed database..."
        gunzip -c "${BACKUP_DIR}/atlas-backup-${backup_timestamp}-postgres.sql.gz" | \
            docker-compose exec -T postgres psql -U atlas atlas
    fi

    # Restore Redis data
    if [ -f "${BACKUP_DIR}/atlas-backup-${backup_timestamp}-redis.rdb" ]; then
        log_info "Restoring Redis data..."
        docker-compose stop redis
        docker cp "${BACKUP_DIR}/atlas-backup-${backup_timestamp}-redis.rdb" atlas-redis:/data/atlas.rdb
        docker-compose start redis
    fi

    # Restore configuration if available
    if [ -f "${BACKUP_DIR}/atlas-backup-${backup_timestamp}-config.tar" ]; then
        log_info "Restoring configuration..."
        tar -xf "${BACKUP_DIR}/atlas-backup-${backup_timestamp}-config.tar" -C ..
    elif [ -f "${BACKUP_DIR}/atlas-backup-${backup_timestamp}-config.tar.gz" ]; then
        tar -xzf "${BACKUP_DIR}/atlas-backup-${backup_timestamp}-config.tar.gz" -C ..
    fi

    log_info "Backup restoration completed"
}

start_services() {
    log_info "Starting services with rollback version..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would start services"
        return 0
    fi

    docker-compose up -d

    # Wait for services to initialize
    sleep 10

    log_info "Services started"
}

verify_rollback() {
    log_info "Verifying rollback..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would verify rollback"
        ROLLBACK_SUCCESSFUL=true
        return 0
    fi

    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -f "$HEALTH_CHECK_URL" >/dev/null 2>&1; then
            log_info "Health check passed"

            # Verify version
            local running_version=$(docker inspect atlas-api 2>/dev/null | \
                grep -o '"org.opencontainers.image.version": "[^"]*' | \
                cut -d'"' -f4) || ""

            if [ -n "$running_version" ]; then
                log_info "Running version: ${running_version}"
            fi

            # Run basic smoke tests
            local test_endpoints=("/api/v1/status" "/health" "/metrics")
            local all_passed=true

            for endpoint in "${test_endpoints[@]}"; do
                if ! curl -f "http://localhost:8080${endpoint}" >/dev/null 2>&1; then
                    log_error "Smoke test failed: ${endpoint}"
                    all_passed=false
                fi
            done

            if [ "$all_passed" = true ]; then
                ROLLBACK_SUCCESSFUL=true
                return 0
            fi
        fi

        log_warn "Verification attempt ${attempt}/${max_attempts} failed"
        sleep 5
        attempt=$((attempt + 1))
    done

    log_error "Rollback verification failed"
    return 1
}

cleanup_failed_rollback() {
    log_error "Rollback failed, attempting to restore original state..."

    # Try to restore to the original version
    if [ -n "$CURRENT_VERSION" ] && [ "$CURRENT_VERSION" != "unknown" ]; then
        docker tag "mezan-atlas:${CURRENT_VERSION}" "mezan-atlas:latest"
        docker-compose up -d

        sleep 10

        if curl -f "$HEALTH_CHECK_URL" >/dev/null 2>&1; then
            log_info "Successfully restored to original version: ${CURRENT_VERSION}"
        else
            log_error "Failed to restore original version"
        fi
    fi
}

generate_rollback_report() {
    local report_file="rollback-report-$(date +%Y%m%d-%H%M%S).txt"

    cat > "$report_file" <<EOF
ATLAS Rollback Report
=====================
Date: $(date)
Operator: $(whoami)
Hostname: $(hostname)

Rollback Details:
- From Version: ${CURRENT_VERSION}
- To Version: ${TARGET_VERSION}
- Status: $([ "$ROLLBACK_SUCCESSFUL" = true ] && echo "SUCCESS" || echo "FAILED")
- Dry Run: ${DRY_RUN}

Container Status:
$(docker-compose ps)

Recent Logs:
$(docker-compose logs --tail=50 atlas-api 2>&1)

Health Check:
$(curl -s "$HEALTH_CHECK_URL" 2>&1 || echo "Health check failed")
EOF

    log_info "Rollback report saved: ${report_file}"
}

main() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}MEZAN ATLAS Rollback Procedure${NC}"
    echo -e "${BLUE}========================================${NC}"

    # Get current state
    get_current_version

    # Validate target
    if ! validate_target_version "$ROLLBACK_TO"; then
        list_available_versions
        exit 1
    fi

    echo ""
    echo -e "${YELLOW}Rollback Configuration:${NC}"
    echo "  Current Version: ${CURRENT_VERSION}"
    echo "  Target Version: ${TARGET_VERSION}"
    echo "  Dry Run: ${DRY_RUN}"
    echo ""

    # Confirmation prompt (unless in dry-run mode)
    if [ "$DRY_RUN" != true ]; then
        read -p "Are you sure you want to proceed with rollback? (yes/no): " confirmation
        if [ "$confirmation" != "yes" ]; then
            log_info "Rollback cancelled by user"
            exit 0
        fi
    fi

    # Send notification
    send_notification "Starting rollback from ${CURRENT_VERSION} to ${TARGET_VERSION}" "info"

    # Perform rollback
    create_rollback_backup
    stop_services

    if [[ "$TARGET_VERSION" == backup-* ]]; then
        rollback_from_backup "${TARGET_VERSION#backup-}"
    else
        rollback_docker_images
    fi

    start_services

    # Verify rollback
    if verify_rollback; then
        log_info "Rollback completed successfully!"
        send_notification "Rollback successful to version ${TARGET_VERSION}" "success"
    else
        log_error "Rollback verification failed"
        cleanup_failed_rollback
        send_notification "Rollback failed - attempted restore to ${CURRENT_VERSION}" "error"
        ROLLBACK_SUCCESSFUL=false
    fi

    # Generate report
    generate_rollback_report

    # Final status
    echo ""
    if [ "$ROLLBACK_SUCCESSFUL" = true ]; then
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}Rollback Successful!${NC}"
        echo -e "${GREEN}========================================${NC}"
        exit 0
    else
        echo -e "${RED}========================================${NC}"
        echo -e "${RED}Rollback Failed!${NC}"
        echo -e "${RED}========================================${NC}"
        exit 1
    fi
}

# Handle script arguments
case "$1" in
    --list|-l)
        list_available_versions
        exit 0
        ;;
    --dry-run|-d)
        DRY_RUN=true
        ROLLBACK_TO="${2:-previous}"
        main
        ;;
    --help|-h)
        echo "Usage: $0 [VERSION|OPTIONS]"
        echo ""
        echo "Arguments:"
        echo "  VERSION     Version to rollback to (tag, timestamp, or 'previous')"
        echo ""
        echo "Options:"
        echo "  -l, --list     List available versions"
        echo "  -d, --dry-run  Perform dry run without actual changes"
        echo "  -h, --help     Show this help message"
        echo ""
        echo "Environment Variables:"
        echo "  DRY_RUN            Enable dry run mode"
        echo "  BACKUP_DIR         Directory containing backups"
        echo "  SLACK_WEBHOOK      Slack webhook for notifications"
        echo ""
        echo "Examples:"
        echo "  $0 previous           # Rollback to previous version"
        echo "  $0 v1.2.3            # Rollback to specific version"
        echo "  $0 20240101-120000   # Rollback from backup timestamp"
        echo "  $0 --dry-run v1.2.3  # Dry run rollback"
        exit 0
        ;;
    *)
        main
        ;;
esac