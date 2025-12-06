#!/bin/bash

# CrazyIdeas Platform - Rollback Script
# This script handles safe rollback to previous deployment version

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-staging}
TARGET_VERSION=${2:-previous}
ROLLBACK_TYPE=${3:-fast} # fast or safe

# Function to print colored output
log() {
    echo -e "${2:-$NC}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# Function to get previous version
get_previous_version() {
    log "Identifying previous version..." "$YELLOW"

    # Get deployment history from tags or database
    if [ -f /var/deployments/history.json ]; then
        PREVIOUS_VERSION=$(jq -r ".deployments[-2].version" /var/deployments/history.json)
        log "Previous version identified: $PREVIOUS_VERSION" "$GREEN"
        echo "$PREVIOUS_VERSION"
    else
        log "Could not identify previous version" "$RED"
        exit 1
    fi
}

# Function to check current state
check_current_state() {
    log "Checking current deployment state..." "$YELLOW"

    # Save current state
    docker ps -a > /tmp/current_state.txt
    docker images > /tmp/current_images.txt

    # Check for unhealthy services
    UNHEALTHY=$(docker ps --filter "health=unhealthy" -q | wc -l)
    if [ "$UNHEALTHY" -gt 0 ]; then
        log "Found $UNHEALTHY unhealthy containers" "$YELLOW"
    fi

    log "Current state saved" "$GREEN"
}

# Function for fast rollback (switch images)
fast_rollback() {
    local version=$1
    log "Performing fast rollback to version $version..." "$YELLOW"

    # Update docker-compose with target version
    export IMAGE_TAG=$version

    # Restart services with new version
    if [[ "$ENVIRONMENT" == "production" ]]; then
        docker-compose -f docker-compose.prod.yml up -d --no-deps
    else
        docker-compose -f docker-compose.dev.yml up -d --no-deps
    fi

    log "Fast rollback completed" "$GREEN"
}

# Function for safe rollback (blue-green switch)
safe_rollback() {
    local version=$1
    log "Performing safe rollback to version $version..." "$YELLOW"

    # Deploy old version alongside current
    log "Deploying previous version in parallel..."
    docker-compose -f docker-compose.rollback.yml up -d

    # Wait for old version to be healthy
    sleep 30

    # Switch traffic to old version
    log "Switching traffic to previous version..."
    ./switch_traffic.sh previous

    # Wait for traffic drain
    sleep 60

    # Stop current version
    log "Stopping current version..."
    docker-compose -f docker-compose.prod.yml down

    # Rename rollback to production
    mv docker-compose.rollback.yml docker-compose.prod.yml

    log "Safe rollback completed" "$GREEN"
}

# Function to verify rollback
verify_rollback() {
    log "Verifying rollback..." "$YELLOW"

    # Check all services are running
    for service in ghost-researcher scientific-tinder chaos-engine api-backend; do
        if docker ps | grep -q "$service"; then
            log "$service is running" "$GREEN"
        else
            log "$service is not running" "$RED"
            return 1
        fi
    done

    # Run health checks
    ./health_check.sh || {
        log "Health checks failed after rollback" "$RED"
        return 1
    }

    log "Rollback verified successfully" "$GREEN"
}

# Function to restore database
restore_database() {
    local backup_date=$1
    log "Restoring database from backup $backup_date..." "$YELLOW"

    # Find backup file
    BACKUP_FILE="/var/backups/postgres/backup-${backup_date}.sql"

    if [ ! -f "$BACKUP_FILE" ]; then
        log "Backup file not found: $BACKUP_FILE" "$RED"
        return 1
    fi

    # Create restore point
    docker exec postgres-primary pg_dump -U postgres crazyideas > /tmp/pre_rollback_backup.sql

    # Restore database
    docker exec -i postgres-primary psql -U postgres crazyideas < "$BACKUP_FILE" || {
        log "Database restore failed" "$RED"
        return 1
    }

    log "Database restored successfully" "$GREEN"
}

# Function to clear cache
clear_cache() {
    log "Clearing application caches..." "$YELLOW"

    # Clear Redis cache
    docker exec redis-master redis-cli FLUSHALL

    # Clear CDN cache
    if [ -n "$CLOUDFLARE_API_TOKEN" ]; then
        curl -X POST "https://api.cloudflare.com/client/v4/zones/$CLOUDFLARE_ZONE_ID/purge_cache" \
            -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
            -H "Content-Type: application/json" \
            --data '{"purge_everything":true}'
    fi

    log "Caches cleared" "$GREEN"
}

# Function to notify rollback
notify_rollback() {
    local status=$1
    local reason=$2

    # Log to monitoring system
    curl -X POST http://localhost:9090/api/v1/alerts \
        -H "Content-Type: application/json" \
        -d "{
            \"labels\": {
                \"alertname\": \"RollbackExecuted\",
                \"severity\": \"warning\",
                \"environment\": \"$ENVIRONMENT\"
            },
            \"annotations\": {
                \"status\": \"$status\",
                \"reason\": \"$reason\"
            }
        }"

    # Send to Slack
    if [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🔄 Rollback executed on $ENVIRONMENT\nStatus: $status\nReason: $reason\"}" \
            "$SLACK_WEBHOOK"
    fi
}

# Main rollback flow
main() {
    log "=== Starting Rollback Process ===" "$YELLOW"
    log "Environment: $ENVIRONMENT"
    log "Target Version: $TARGET_VERSION"
    log "Rollback Type: $ROLLBACK_TYPE"
    log "================================" "$YELLOW"

    # Check current state
    check_current_state

    # Determine target version
    if [ "$TARGET_VERSION" == "previous" ]; then
        TARGET_VERSION=$(get_previous_version)
    fi

    # Perform rollback based on type
    case "$ROLLBACK_TYPE" in
        fast)
            fast_rollback "$TARGET_VERSION"
            ;;
        safe)
            safe_rollback "$TARGET_VERSION"
            ;;
        *)
            log "Unknown rollback type: $ROLLBACK_TYPE" "$RED"
            exit 1
            ;;
    esac

    # Verify rollback
    if verify_rollback; then
        log "Rollback successful" "$GREEN"
        notify_rollback "SUCCESS" "Rolled back to version $TARGET_VERSION"

        # Clear caches after successful rollback
        clear_cache

        # Optional: Restore database if needed
        read -p "Do you want to restore the database? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            read -p "Enter backup date (YYYYMMDD): " BACKUP_DATE
            restore_database "$BACKUP_DATE"
        fi
    else
        log "Rollback verification failed" "$RED"
        notify_rollback "FAILED" "Rollback verification failed"
        exit 1
    fi

    log "=== Rollback Complete ===" "$GREEN"
    log "Current Version: $TARGET_VERSION"
    log "Status: SUCCESS"
    log "Time: $(date)"
    log "========================" "$GREEN"
}

# Run main function
main "$@"