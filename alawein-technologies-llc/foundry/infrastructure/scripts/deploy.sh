#!/bin/bash

# CrazyIdeas Platform - Deployment Script
# This script handles automated deployment with health checks and rollback capability

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-staging}
VERSION=${2:-latest}
ROLLBACK_ON_FAILURE=${ROLLBACK_ON_FAILURE:-true}
HEALTH_CHECK_RETRIES=10
HEALTH_CHECK_INTERVAL=30
DEPLOYMENT_TIMEOUT=600

# Services to deploy
SERVICES=("ghost-researcher" "scientific-tinder" "chaos-engine" "api-backend")

# Function to print colored output
log() {
    echo -e "${2:-$NC}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking prerequisites..." "$YELLOW"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log "Docker is not installed" "$RED"
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log "Docker Compose is not installed" "$RED"
        exit 1
    fi

    # Check AWS CLI (if using AWS)
    if [[ "$ENVIRONMENT" == "production" ]]; then
        if ! command -v aws &> /dev/null; then
            log "AWS CLI is not installed" "$RED"
            exit 1
        fi
    fi

    log "Prerequisites check passed" "$GREEN"
}

# Function to backup current deployment
backup_current_deployment() {
    log "Creating backup of current deployment..." "$YELLOW"

    BACKUP_DIR="/var/backups/deployments/$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$BACKUP_DIR"

    # Backup docker-compose files
    cp docker-compose*.yml "$BACKUP_DIR/" 2>/dev/null || true

    # Backup environment files
    cp .env* "$BACKUP_DIR/" 2>/dev/null || true

    # Save current container states
    docker ps -a > "$BACKUP_DIR/container_states.txt"

    # Save current images
    docker images > "$BACKUP_DIR/images.txt"

    log "Backup created at $BACKUP_DIR" "$GREEN"
    echo "$BACKUP_DIR" > /tmp/last_backup_dir.txt
}

# Function to pull new images
pull_images() {
    log "Pulling new images for version $VERSION..." "$YELLOW"

    for service in "${SERVICES[@]}"; do
        log "Pulling $service:$VERSION"
        docker pull "ghcr.io/crazyideas/$service:$VERSION" || {
            log "Failed to pull $service:$VERSION" "$RED"
            return 1
        }
    done

    log "All images pulled successfully" "$GREEN"
}

# Function to deploy services
deploy_services() {
    log "Deploying services to $ENVIRONMENT..." "$YELLOW"

    if [[ "$ENVIRONMENT" == "production" ]]; then
        docker-compose -f docker-compose.prod.yml up -d --remove-orphans
    else
        docker-compose -f docker-compose.dev.yml up -d --remove-orphans
    fi

    log "Services deployed" "$GREEN"
}

# Function to run database migrations
run_migrations() {
    log "Running database migrations..." "$YELLOW"

    docker-compose exec -T api-backend npm run migrate || {
        log "Migration failed" "$RED"
        return 1
    }

    log "Migrations completed" "$GREEN"
}

# Function to perform health checks
health_check() {
    local service=$1
    local port=$2
    local endpoint=${3:-/api/health}

    for i in $(seq 1 $HEALTH_CHECK_RETRIES); do
        log "Health check attempt $i/$HEALTH_CHECK_RETRIES for $service..."

        if curl -f -s -o /dev/null -w "%{http_code}" "http://localhost:$port$endpoint" | grep -q "200"; then
            log "$service is healthy" "$GREEN"
            return 0
        fi

        if [ $i -lt $HEALTH_CHECK_RETRIES ]; then
            log "Health check failed, retrying in $HEALTH_CHECK_INTERVAL seconds..."
            sleep $HEALTH_CHECK_INTERVAL
        fi
    done

    log "$service health check failed after $HEALTH_CHECK_RETRIES attempts" "$RED"
    return 1
}

# Function to perform all health checks
perform_health_checks() {
    log "Performing health checks on all services..." "$YELLOW"

    local all_healthy=true

    health_check "ghost-researcher" 3000 "/api/health" || all_healthy=false
    health_check "scientific-tinder" 3001 "/api/health" || all_healthy=false
    health_check "chaos-engine" 3002 "/api/health" || all_healthy=false
    health_check "api-backend" 8000 "/health" || all_healthy=false

    if $all_healthy; then
        log "All services are healthy" "$GREEN"
        return 0
    else
        log "Some services failed health checks" "$RED"
        return 1
    fi
}

# Function to run smoke tests
run_smoke_tests() {
    log "Running smoke tests..." "$YELLOW"

    # Run basic smoke tests
    npm run test:smoke:$ENVIRONMENT || {
        log "Smoke tests failed" "$RED"
        return 1
    }

    log "Smoke tests passed" "$GREEN"
}

# Function to rollback deployment
rollback() {
    log "Initiating rollback..." "$RED"

    if [ -f /tmp/last_backup_dir.txt ]; then
        BACKUP_DIR=$(cat /tmp/last_backup_dir.txt)
        log "Rolling back to backup: $BACKUP_DIR" "$YELLOW"

        # Restore docker-compose files
        cp "$BACKUP_DIR"/docker-compose*.yml . 2>/dev/null || true

        # Restore environment files
        cp "$BACKUP_DIR"/.env* . 2>/dev/null || true

        # Restart services with old configuration
        if [[ "$ENVIRONMENT" == "production" ]]; then
            docker-compose -f docker-compose.prod.yml up -d --force-recreate
        else
            docker-compose -f docker-compose.dev.yml up -d --force-recreate
        fi

        log "Rollback completed" "$GREEN"
    else
        log "No backup found for rollback" "$RED"
        exit 1
    fi
}

# Function to notify deployment status
notify_deployment() {
    local status=$1
    local message=$2

    # Send to Slack
    if [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"Deployment to $ENVIRONMENT: $status\n$message\"}" \
            "$SLACK_WEBHOOK"
    fi

    # Send email notification
    if [ -n "$NOTIFICATION_EMAIL" ]; then
        echo "$message" | mail -s "Deployment $status - $ENVIRONMENT" "$NOTIFICATION_EMAIL"
    fi
}

# Function to update monitoring
update_monitoring() {
    log "Updating monitoring configuration..." "$YELLOW"

    # Update Prometheus targets
    docker-compose restart prometheus

    # Update Grafana dashboards
    curl -X POST http://admin:admin@localhost:3003/api/admin/provisioning/dashboards/reload

    log "Monitoring updated" "$GREEN"
}

# Function to clean up old images
cleanup_old_images() {
    log "Cleaning up old images..." "$YELLOW"

    # Remove unused images
    docker image prune -af --filter "until=72h"

    # Remove unused volumes
    docker volume prune -f

    log "Cleanup completed" "$GREEN"
}

# Main deployment flow
main() {
    log "Starting deployment to $ENVIRONMENT with version $VERSION" "$GREEN"

    # Pre-deployment checks
    check_prerequisites

    # Create backup
    backup_current_deployment

    # Pull new images
    if ! pull_images; then
        log "Failed to pull images" "$RED"
        notify_deployment "FAILED" "Failed to pull Docker images"
        exit 1
    fi

    # Deploy services
    if ! deploy_services; then
        log "Deployment failed" "$RED"
        if $ROLLBACK_ON_FAILURE; then
            rollback
        fi
        notify_deployment "FAILED" "Failed to deploy services"
        exit 1
    fi

    # Run migrations
    if ! run_migrations; then
        log "Migrations failed" "$RED"
        if $ROLLBACK_ON_FAILURE; then
            rollback
        fi
        notify_deployment "FAILED" "Database migrations failed"
        exit 1
    fi

    # Health checks
    if ! perform_health_checks; then
        log "Health checks failed" "$RED"
        if $ROLLBACK_ON_FAILURE; then
            rollback
        fi
        notify_deployment "FAILED" "Health checks failed"
        exit 1
    fi

    # Smoke tests
    if ! run_smoke_tests; then
        log "Smoke tests failed" "$RED"
        if $ROLLBACK_ON_FAILURE; then
            rollback
        fi
        notify_deployment "FAILED" "Smoke tests failed"
        exit 1
    fi

    # Post-deployment tasks
    update_monitoring
    cleanup_old_images

    # Success notification
    log "Deployment completed successfully!" "$GREEN"
    notify_deployment "SUCCESS" "All services deployed and healthy"

    # Print deployment summary
    echo ""
    log "=== Deployment Summary ===" "$GREEN"
    log "Environment: $ENVIRONMENT"
    log "Version: $VERSION"
    log "Services: ${SERVICES[*]}"
    log "Status: SUCCESS"
    log "Time: $(date)"
    log "======================="
}

# Run main function
main "$@"