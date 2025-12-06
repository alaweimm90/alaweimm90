#!/bin/bash
# Health monitoring script for MEZAN ATLAS

set -e

# Configuration
HEALTH_ENDPOINT=${HEALTH_ENDPOINT:-"http://localhost:8080/health"}
METRICS_ENDPOINT=${METRICS_ENDPOINT:-"http://localhost:9090/metrics"}
CHECK_INTERVAL=${CHECK_INTERVAL:-30}
ALERT_THRESHOLD=${ALERT_THRESHOLD:-3}
LOG_FILE=${LOG_FILE:-"/app/logs/health-check.log"}
SLACK_WEBHOOK=${SLACK_WEBHOOK:-""}
EMAIL_TO=${EMAIL_TO:-""}

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Counters
CONSECUTIVE_FAILURES=0
TOTAL_CHECKS=0
FAILED_CHECKS=0

# Service status
declare -A SERVICE_STATUS
SERVICE_STATUS["api"]="unknown"
SERVICE_STATUS["redis"]="unknown"
SERVICE_STATUS["postgres"]="unknown"
SERVICE_STATUS["prometheus"]="unknown"

# Functions
log_message() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" >> "$LOG_FILE"

    # Also output to console with colors
    case "$level" in
        ERROR)
            echo -e "${RED}[${level}]${NC} ${message}"
            ;;
        WARN)
            echo -e "${YELLOW}[${level}]${NC} ${message}"
            ;;
        OK)
            echo -e "${GREEN}[${level}]${NC} ${message}"
            ;;
        *)
            echo "[${level}] ${message}"
            ;;
    esac
}

send_alert() {
    local service="$1"
    local status="$2"
    local message="$3"

    # Log the alert
    log_message "ALERT" "${service} - ${status}: ${message}"

    # Send Slack notification
    if [ -n "$SLACK_WEBHOOK" ]; then
        local color="danger"
        [ "$status" = "recovered" ] && color="good"

        curl -X POST -H 'Content-type: application/json' \
            --data "{
                \"text\":\"ATLAS Health Alert\",
                \"attachments\": [{
                    \"color\": \"${color}\",
                    \"fields\": [
                        {\"title\": \"Service\", \"value\": \"${service}\", \"short\": true},
                        {\"title\": \"Status\", \"value\": \"${status}\", \"short\": true},
                        {\"title\": \"Message\", \"value\": \"${message}\"},
                        {\"title\": \"Time\", \"value\": \"$(date)\", \"short\": true}
                    ]
                }]
            }" \
            "$SLACK_WEBHOOK" 2>/dev/null || log_message "ERROR" "Failed to send Slack notification"
    fi

    # Send email notification (requires mail command)
    if [ -n "$EMAIL_TO" ] && command -v mail &> /dev/null; then
        echo -e "Service: ${service}\nStatus: ${status}\nMessage: ${message}\nTime: $(date)" | \
            mail -s "ATLAS Health Alert: ${service} ${status}" "$EMAIL_TO" 2>/dev/null || \
            log_message "ERROR" "Failed to send email notification"
    fi
}

check_api_health() {
    local service="api"
    local response
    local http_code

    http_code=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_ENDPOINT" 2>/dev/null)

    if [ "$http_code" = "200" ]; then
        if [ "${SERVICE_STATUS[$service]}" = "down" ]; then
            send_alert "$service" "recovered" "API service is back online"
        fi
        SERVICE_STATUS[$service]="up"
        return 0
    else
        if [ "${SERVICE_STATUS[$service]}" != "down" ]; then
            send_alert "$service" "down" "API returned HTTP ${http_code}"
        fi
        SERVICE_STATUS[$service]="down"
        return 1
    fi
}

check_redis_health() {
    local service="redis"

    if docker-compose exec -T redis redis-cli ping 2>/dev/null | grep -q PONG; then
        if [ "${SERVICE_STATUS[$service]}" = "down" ]; then
            send_alert "$service" "recovered" "Redis is back online"
        fi
        SERVICE_STATUS[$service]="up"
        return 0
    else
        if [ "${SERVICE_STATUS[$service]}" != "down" ]; then
            send_alert "$service" "down" "Redis is not responding"
        fi
        SERVICE_STATUS[$service]="down"
        return 1
    fi
}

check_postgres_health() {
    local service="postgres"

    if docker-compose exec -T postgres pg_isready -U atlas 2>/dev/null | grep -q "accepting connections"; then
        if [ "${SERVICE_STATUS[$service]}" = "down" ]; then
            send_alert "$service" "recovered" "PostgreSQL is back online"
        fi
        SERVICE_STATUS[$service]="up"
        return 0
    else
        if [ "${SERVICE_STATUS[$service]}" != "down" ]; then
            send_alert "$service" "down" "PostgreSQL is not accepting connections"
        fi
        SERVICE_STATUS[$service]="down"
        return 1
    fi
}

check_disk_space() {
    local threshold=80
    local usage=$(df /app 2>/dev/null | awk 'NR==2 {print int($5)}')

    if [ "$usage" -gt "$threshold" ]; then
        send_alert "disk" "warning" "Disk usage is at ${usage}% (threshold: ${threshold}%)"
        return 1
    fi
    return 0
}

check_memory_usage() {
    local threshold=80
    local usage=$(free | grep Mem | awk '{print int($3/$2 * 100)}')

    if [ "$usage" -gt "$threshold" ]; then
        send_alert "memory" "warning" "Memory usage is at ${usage}% (threshold: ${threshold}%)"
        return 1
    fi
    return 0
}

check_container_status() {
    local unhealthy_containers=$(docker ps --filter "health=unhealthy" --format "{{.Names}}" 2>/dev/null)

    if [ -n "$unhealthy_containers" ]; then
        send_alert "containers" "unhealthy" "Unhealthy containers: ${unhealthy_containers}"
        return 1
    fi
    return 0
}

check_response_time() {
    local threshold=2000  # 2 seconds in milliseconds
    local start_time=$(date +%s%N)

    curl -s "$HEALTH_ENDPOINT" >/dev/null 2>&1

    local end_time=$(date +%s%N)
    local response_time=$(( (end_time - start_time) / 1000000 ))

    if [ "$response_time" -gt "$threshold" ]; then
        send_alert "performance" "slow" "API response time is ${response_time}ms (threshold: ${threshold}ms)"
        return 1
    fi
    return 0
}

collect_metrics() {
    local timestamp=$(date '+%s')

    # Collect container stats
    docker stats --no-stream --format "json" > "/app/logs/container-stats-${timestamp}.json" 2>/dev/null || true

    # Collect Redis info
    docker-compose exec -T redis redis-cli INFO > "/app/logs/redis-info-${timestamp}.txt" 2>/dev/null || true

    # Collect PostgreSQL stats
    docker-compose exec -T postgres psql -U atlas -c "SELECT * FROM pg_stat_database WHERE datname='atlas';" > "/app/logs/postgres-stats-${timestamp}.txt" 2>/dev/null || true
}

perform_health_check() {
    local all_healthy=true
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

    log_message "INFO" "Starting health check #${TOTAL_CHECKS}"

    # Check each service
    if ! check_api_health; then
        all_healthy=false
    fi

    if ! check_redis_health; then
        all_healthy=false
    fi

    if ! check_postgres_health; then
        all_healthy=false
    fi

    # Check system resources
    check_disk_space || true
    check_memory_usage || true
    check_container_status || true
    check_response_time || true

    # Update failure counter
    if [ "$all_healthy" = false ]; then
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        log_message "ERROR" "Health check failed (consecutive: ${CONSECUTIVE_FAILURES})"

        if [ "$CONSECUTIVE_FAILURES" -ge "$ALERT_THRESHOLD" ]; then
            send_alert "system" "critical" "System has failed ${CONSECUTIVE_FAILURES} consecutive health checks"
        fi
    else
        if [ "$CONSECUTIVE_FAILURES" -gt 0 ]; then
            log_message "OK" "System recovered after ${CONSECUTIVE_FAILURES} failures"
            send_alert "system" "recovered" "All services are healthy again"
        fi
        CONSECUTIVE_FAILURES=0
        log_message "OK" "All services healthy"
    fi

    # Collect metrics periodically
    if [ $((TOTAL_CHECKS % 10)) -eq 0 ]; then
        collect_metrics
    fi
}

print_status() {
    echo ""
    echo "===== ATLAS Health Status ====="
    echo "Time: $(date)"
    echo "Total Checks: ${TOTAL_CHECKS}"
    echo "Failed Checks: ${FAILED_CHECKS}"
    echo "Consecutive Failures: ${CONSECUTIVE_FAILURES}"
    echo ""
    echo "Service Status:"
    for service in "${!SERVICE_STATUS[@]}"; do
        local status="${SERVICE_STATUS[$service]}"
        local color=$NC
        [ "$status" = "up" ] && color=$GREEN
        [ "$status" = "down" ] && color=$RED
        echo -e "  ${service}: ${color}${status}${NC}"
    done
    echo "==============================="
}

cleanup() {
    log_message "INFO" "Health check monitoring stopped"
    print_status
    exit 0
}

# Main execution
main() {
    # Setup signal handlers
    trap cleanup SIGINT SIGTERM

    # Create log directory if needed
    mkdir -p "$(dirname "$LOG_FILE")"

    log_message "INFO" "Starting ATLAS health monitoring (interval: ${CHECK_INTERVAL}s)"

    # Continuous monitoring loop
    if [ "$1" = "--continuous" ] || [ "$1" = "-c" ]; then
        while true; do
            perform_health_check
            print_status
            sleep "$CHECK_INTERVAL"
        done
    else
        # Single check
        perform_health_check
        print_status

        # Exit with appropriate code
        [ "$CONSECUTIVE_FAILURES" -eq 0 ] && exit 0 || exit 1
    fi
}

# Handle arguments
case "$1" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  -c, --continuous  Run continuous monitoring"
        echo "  -h, --help        Show this help message"
        echo ""
        echo "Environment Variables:"
        echo "  HEALTH_ENDPOINT   Health check URL (default: http://localhost:8080/health)"
        echo "  CHECK_INTERVAL    Check interval in seconds (default: 30)"
        echo "  ALERT_THRESHOLD   Consecutive failures before alert (default: 3)"
        echo "  SLACK_WEBHOOK     Slack webhook URL for alerts"
        echo "  EMAIL_TO          Email address for alerts"
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac