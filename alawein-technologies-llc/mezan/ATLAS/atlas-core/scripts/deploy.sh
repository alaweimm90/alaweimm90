#!/bin/bash
# Production deployment script for MEZAN ATLAS

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-production}
DEPLOY_HOST=${DEPLOY_HOST:-""}
DEPLOY_USER=${DEPLOY_USER:-"deploy"}
DEPLOY_PATH=${DEPLOY_PATH:-"/opt/atlas"}
BACKUP_BEFORE_DEPLOY=${BACKUP_BEFORE_DEPLOY:-true}
HEALTH_CHECK_URL=${HEALTH_CHECK_URL:-"http://localhost:8080/health"}
ROLLBACK_ON_FAILURE=${ROLLBACK_ON_FAILURE:-true}
SLACK_WEBHOOK=${SLACK_WEBHOOK:-""}

# Versioning
VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo "dev")
PREVIOUS_VERSION=""

# Logging
LOG_FILE="deploy-$(date +%Y%m%d-%H%M%S).log"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

send_notification() {
    local message="$1"
    local status="$2"

    # Send to Slack if webhook is configured
    if [ -n "$SLACK_WEBHOOK" ]; then
        local color="good"
        [ "$status" = "error" ] && color="danger"
        [ "$status" = "warning" ] && color="warning"

        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"ATLAS Deployment: ${message}\",\"color\":\"${color}\"}" \
            "$SLACK_WEBHOOK" 2>/dev/null || true
    fi
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check required tools
    local required_tools=("docker" "docker-compose" "git" "curl")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is not installed"
            exit 1
        fi
    done

    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon is not running"
        exit 1
    fi

    # Check environment file
    if [ ! -f ".env.$ENVIRONMENT" ]; then
        log_error "Environment file .env.$ENVIRONMENT not found"
        exit 1
    fi

    log_info "Prerequisites check passed"
}

backup_current() {
    if [ "$BACKUP_BEFORE_DEPLOY" = true ]; then
        log_info "Creating backup of current deployment..."

        local backup_name="backup-$(date +%Y%m%d-%H%M%S)"
        mkdir -p backups

        # Backup database
        docker-compose exec -T postgres pg_dump -U atlas atlas > "backups/${backup_name}-db.sql" 2>/dev/null || true

        # Backup Redis
        docker-compose exec -T redis redis-cli --rdb "backups/${backup_name}-redis.rdb" 2>/dev/null || true

        # Backup volumes
        docker run --rm -v atlas-data:/data -v $(pwd)/backups:/backup \
            alpine tar czf "/backup/${backup_name}-volumes.tar.gz" /data 2>/dev/null || true

        # Store current version for rollback
        PREVIOUS_VERSION=$(docker-compose ps -q atlas-api | xargs docker inspect -f '{{.Config.Image}}' | cut -d: -f2)

        log_info "Backup created: ${backup_name}"
    fi
}

build_images() {
    log_info "Building Docker images..."

    # Run build script
    ./scripts/build.sh --target production --version "$VERSION"

    if [ $? -ne 0 ]; then
        log_error "Docker build failed"
        exit 1
    fi

    log_info "Docker images built successfully"
}

run_tests() {
    log_info "Running pre-deployment tests..."

    # Build test image
    docker build --target tester -t atlas-test:latest .

    # Run tests
    if docker run --rm atlas-test:latest pytest; then
        log_info "All tests passed"
    else
        log_error "Tests failed"
        if [ "$ROLLBACK_ON_FAILURE" = true ]; then
            rollback
        fi
        exit 1
    fi
}

deploy_local() {
    log_info "Deploying to local environment..."

    # Stop current deployment
    log_info "Stopping current containers..."
    docker-compose down --remove-orphans

    # Update images
    log_info "Starting new containers..."
    docker-compose up -d

    # Wait for services to be ready
    sleep 10

    # Run database migrations if needed
    log_info "Running database migrations..."
    docker-compose exec -T atlas-api python -m atlas.migrations || true
}

deploy_remote() {
    log_info "Deploying to remote host: $DEPLOY_HOST"

    # Copy files to remote host
    log_info "Copying files to remote host..."
    rsync -avz --exclude-from=.dockerignore \
        . "${DEPLOY_USER}@${DEPLOY_HOST}:${DEPLOY_PATH}/"

    # Run deployment on remote host
    log_info "Running deployment on remote host..."
    ssh "${DEPLOY_USER}@${DEPLOY_HOST}" "cd ${DEPLOY_PATH} && ./scripts/deploy.sh local"
}

health_check() {
    log_info "Running health checks..."

    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -f "$HEALTH_CHECK_URL" >/dev/null 2>&1; then
            log_info "Health check passed"
            return 0
        fi
        log_warn "Health check attempt $attempt/$max_attempts failed"
        sleep 5
        attempt=$((attempt + 1))
    done

    log_error "Health check failed after $max_attempts attempts"
    return 1
}

smoke_tests() {
    log_info "Running smoke tests..."

    # Test API endpoints
    local endpoints=("/health" "/api/v1/status" "/metrics")
    for endpoint in "${endpoints[@]}"; do
        if curl -f "http://localhost:8080${endpoint}" >/dev/null 2>&1; then
            log_info "Smoke test passed: $endpoint"
        else
            log_error "Smoke test failed: $endpoint"
            return 1
        fi
    done

    log_info "All smoke tests passed"
}

rollback() {
    log_error "Deployment failed, rolling back..."

    if [ -n "$PREVIOUS_VERSION" ]; then
        # Stop current containers
        docker-compose down

        # Restore previous version
        docker tag "atlas:${PREVIOUS_VERSION}" "atlas:latest"

        # Start containers with previous version
        docker-compose up -d

        # Restore database if backup exists
        if [ -f "backups/latest-db.sql" ]; then
            docker-compose exec -T postgres psql -U atlas atlas < "backups/latest-db.sql"
        fi

        log_info "Rollback completed to version: $PREVIOUS_VERSION"
        send_notification "Deployment failed and rolled back to $PREVIOUS_VERSION" "warning"
    else
        log_error "No previous version found for rollback"
    fi
}

cleanup() {
    log_info "Cleaning up old images and containers..."

    # Remove unused images
    docker image prune -f --filter "until=24h"

    # Remove stopped containers
    docker container prune -f

    # Clean old backups (keep last 10)
    if [ -d backups ]; then
        ls -t backups/*.tar.gz 2>/dev/null | tail -n +11 | xargs rm -f 2>/dev/null || true
    fi

    log_info "Cleanup completed"
}

update_monitoring() {
    log_info "Updating monitoring configuration..."

    # Reload Prometheus configuration
    docker-compose exec -T prometheus kill -HUP 1 2>/dev/null || true

    # Restart Grafana to load new dashboards
    docker-compose restart grafana 2>/dev/null || true

    log_info "Monitoring configuration updated"
}

main() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}MEZAN ATLAS Deployment${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo "Environment: $ENVIRONMENT"
    echo "Version: $VERSION"
    echo "Timestamp: $(date)"
    echo ""

    # Send deployment start notification
    send_notification "Deployment started for version $VERSION to $ENVIRONMENT" "info"

    # Run deployment steps
    check_prerequisites
    backup_current
    build_images
    run_tests

    # Deploy based on target
    if [ -n "$DEPLOY_HOST" ]; then
        deploy_remote
    else
        deploy_local
    fi

    # Post-deployment checks
    if health_check; then
        smoke_tests
        if [ $? -eq 0 ]; then
            update_monitoring
            cleanup

            log_info "Deployment completed successfully!"
            send_notification "Deployment successful! Version $VERSION deployed to $ENVIRONMENT" "good"

            echo ""
            echo -e "${GREEN}========================================${NC}"
            echo -e "${GREEN}Deployment Successful!${NC}"
            echo -e "${GREEN}========================================${NC}"
            echo "Version: $VERSION"
            echo "Environment: $ENVIRONMENT"
            echo "Health Check: PASSED"
            echo "Access URL: $HEALTH_CHECK_URL"
            echo "Logs: $LOG_FILE"
        else
            log_error "Smoke tests failed"
            if [ "$ROLLBACK_ON_FAILURE" = true ]; then
                rollback
            fi
            exit 1
        fi
    else
        log_error "Health check failed"
        if [ "$ROLLBACK_ON_FAILURE" = true ]; then
            rollback
        fi
        exit 1
    fi
}

# Handle script arguments
case "$1" in
    production|staging|development|local)
        ENVIRONMENT=$1
        ;;
    rollback)
        rollback
        exit $?
        ;;
    help|--help|-h)
        echo "Usage: $0 [environment|rollback|help]"
        echo ""
        echo "Environments:"
        echo "  production  - Deploy to production"
        echo "  staging     - Deploy to staging"
        echo "  development - Deploy to development"
        echo "  local       - Deploy locally (default)"
        echo ""
        echo "Commands:"
        echo "  rollback    - Rollback to previous version"
        echo "  help        - Show this help message"
        exit 0
        ;;
    *)
        if [ -n "$1" ]; then
            log_warn "Unknown environment: $1, using 'local'"
        fi
        ENVIRONMENT="local"
        ;;
esac

# Run deployment
main