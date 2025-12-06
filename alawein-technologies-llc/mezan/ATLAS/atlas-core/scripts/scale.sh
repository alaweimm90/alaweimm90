#!/bin/bash
# Scaling script for MEZAN ATLAS services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SERVICE=${1:-"atlas-api"}
SCALE_TO=${2:-""}
ACTION=${3:-"scale"}
MAX_REPLICAS=${MAX_REPLICAS:-10}
MIN_REPLICAS=${MIN_REPLICAS:-1}
LOAD_BALANCER=${LOAD_BALANCER:-"nginx"}
HEALTH_CHECK_DELAY=${HEALTH_CHECK_DELAY:-30}
AUTO_SCALE_CPU_THRESHOLD=${AUTO_SCALE_CPU_THRESHOLD:-80}
AUTO_SCALE_MEM_THRESHOLD=${AUTO_SCALE_MEM_THRESHOLD:-80}

# Current state
CURRENT_REPLICAS=0
TARGET_REPLICAS=0

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

get_current_replicas() {
    local service="$1"
    CURRENT_REPLICAS=$(docker-compose ps "$service" 2>/dev/null | grep -c "$service" || echo "0")
    log_info "Current replicas for ${service}: ${CURRENT_REPLICAS}"
}

get_resource_usage() {
    local service="$1"

    # Get CPU and memory usage
    local stats=$(docker stats --no-stream --format "json" $(docker-compose ps -q "$service") 2>/dev/null)

    if [ -n "$stats" ]; then
        local cpu_total=0
        local mem_total=0
        local count=0

        echo "$stats" | while read -r line; do
            local cpu=$(echo "$line" | jq -r '.CPUPerc' | tr -d '%')
            local mem=$(echo "$line" | jq -r '.MemPerc' | tr -d '%')
            cpu_total=$(echo "$cpu_total + $cpu" | bc)
            mem_total=$(echo "$mem_total + $mem" | bc)
            count=$((count + 1))
        done

        if [ $count -gt 0 ]; then
            local avg_cpu=$(echo "scale=2; $cpu_total / $count" | bc)
            local avg_mem=$(echo "scale=2; $mem_total / $count" | bc)
            echo "CPU: ${avg_cpu}%, Memory: ${avg_mem}%"
        fi
    fi
}

calculate_auto_scale() {
    local service="$1"

    log_info "Calculating auto-scale requirements for ${service}..."

    # Get current resource usage
    local usage=$(get_resource_usage "$service")
    local cpu_usage=$(echo "$usage" | grep -o 'CPU: [0-9.]*' | cut -d' ' -f2)
    local mem_usage=$(echo "$usage" | grep -o 'Memory: [0-9.]*' | cut -d' ' -f2)

    log_info "Resource usage - CPU: ${cpu_usage}%, Memory: ${mem_usage}%"

    # Calculate target replicas based on usage
    TARGET_REPLICAS=$CURRENT_REPLICAS

    # Scale up if thresholds exceeded
    if (( $(echo "$cpu_usage > $AUTO_SCALE_CPU_THRESHOLD" | bc -l) )); then
        TARGET_REPLICAS=$((CURRENT_REPLICAS + 1))
        log_info "CPU threshold exceeded, scaling up"
    elif (( $(echo "$mem_usage > $AUTO_SCALE_MEM_THRESHOLD" | bc -l) )); then
        TARGET_REPLICAS=$((CURRENT_REPLICAS + 1))
        log_info "Memory threshold exceeded, scaling up"
    # Scale down if usage is low
    elif (( $(echo "$cpu_usage < 30" | bc -l) )) && (( $(echo "$mem_usage < 30" | bc -l) )); then
        if [ $CURRENT_REPLICAS -gt $MIN_REPLICAS ]; then
            TARGET_REPLICAS=$((CURRENT_REPLICAS - 1))
            log_info "Low resource usage, scaling down"
        fi
    fi

    # Enforce limits
    if [ $TARGET_REPLICAS -gt $MAX_REPLICAS ]; then
        TARGET_REPLICAS=$MAX_REPLICAS
        log_warn "Target replicas limited to maximum: ${MAX_REPLICAS}"
    elif [ $TARGET_REPLICAS -lt $MIN_REPLICAS ]; then
        TARGET_REPLICAS=$MIN_REPLICAS
        log_warn "Target replicas limited to minimum: ${MIN_REPLICAS}"
    fi
}

scale_service() {
    local service="$1"
    local replicas="$2"

    log_info "Scaling ${service} to ${replicas} replicas..."

    # Use docker-compose scale command
    docker-compose up -d --scale "${service}=${replicas}" --no-recreate

    if [ $? -eq 0 ]; then
        log_info "Scale command executed successfully"
    else
        log_error "Scale command failed"
        return 1
    fi
}

update_load_balancer() {
    log_info "Updating load balancer configuration..."

    # Generate nginx upstream configuration
    local upstream_conf="/tmp/atlas-upstream.conf"

    cat > "$upstream_conf" <<EOF
upstream atlas_backend {
    least_conn;
EOF

    # Add each replica to upstream
    local containers=$(docker-compose ps -q "$SERVICE")
    for container in $containers; do
        local ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$container")
        if [ -n "$ip" ]; then
            echo "    server ${ip}:8080 max_fails=3 fail_timeout=30s;" >> "$upstream_conf"
        fi
    done

    cat >> "$upstream_conf" <<EOF
    keepalive 32;
}
EOF

    # Copy configuration to nginx container
    docker cp "$upstream_conf" atlas-nginx:/etc/nginx/conf.d/upstream.conf 2>/dev/null || true

    # Reload nginx
    docker-compose exec -T nginx nginx -s reload 2>/dev/null || true

    log_info "Load balancer updated"
}

wait_for_healthy() {
    local service="$1"
    local expected_count="$2"
    local max_wait=300  # 5 minutes
    local elapsed=0

    log_info "Waiting for ${expected_count} healthy instances of ${service}..."

    while [ $elapsed -lt $max_wait ]; do
        local healthy_count=0
        local containers=$(docker-compose ps -q "$service")

        for container in $containers; do
            local health=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "none")
            if [ "$health" = "healthy" ]; then
                healthy_count=$((healthy_count + 1))
            fi
        done

        if [ $healthy_count -eq $expected_count ]; then
            log_info "All ${expected_count} instances are healthy"
            return 0
        fi

        log_info "Healthy instances: ${healthy_count}/${expected_count}, waiting..."
        sleep 10
        elapsed=$((elapsed + 10))
    done

    log_error "Timeout waiting for healthy instances"
    return 1
}

perform_rolling_update() {
    local service="$1"
    local new_image="$2"

    log_info "Performing rolling update for ${service}..."

    get_current_replicas "$service"

    # Update one replica at a time
    for i in $(seq 1 $CURRENT_REPLICAS); do
        log_info "Updating replica ${i}/${CURRENT_REPLICAS}..."

        # Stop old container
        local old_container=$(docker-compose ps -q "$service" | head -n 1)
        docker stop "$old_container"
        docker rm "$old_container"

        # Start new container with updated image
        docker-compose up -d --scale "${service}=${CURRENT_REPLICAS}" --no-recreate

        # Wait for new container to be healthy
        wait_for_healthy "$service" "$CURRENT_REPLICAS"

        # Update load balancer
        update_load_balancer

        log_info "Replica ${i} updated successfully"
        sleep 10
    done

    log_info "Rolling update completed"
}

show_status() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}ATLAS Service Status${NC}"
    echo -e "${BLUE}========================================${NC}"

    # Show all services with replica count
    local services=("atlas-api" "redis" "postgres" "prometheus" "grafana")

    for service in "${services[@]}"; do
        local count=$(docker-compose ps "$service" 2>/dev/null | grep -c "$service" || echo "0")
        local status="DOWN"

        if [ $count -gt 0 ]; then
            status="UP"

            # Get resource usage if running
            local usage=$(get_resource_usage "$service")
            if [ -n "$usage" ]; then
                echo -e "${service}: ${GREEN}${status}${NC} (Replicas: ${count}) - ${usage}"
            else
                echo -e "${service}: ${GREEN}${status}${NC} (Replicas: ${count})"
            fi
        else
            echo -e "${service}: ${RED}${status}${NC}"
        fi
    done

    echo ""
    echo "Load Balancer Status:"
    if docker-compose ps nginx | grep -q "Up"; then
        echo -e "Nginx: ${GREEN}UP${NC}"

        # Show upstream servers
        docker-compose exec -T nginx cat /etc/nginx/conf.d/upstream.conf 2>/dev/null | grep "server " || true
    else
        echo -e "Nginx: ${RED}DOWN${NC}"
    fi
}

monitor_auto_scale() {
    log_info "Starting auto-scale monitoring..."

    while true; do
        get_current_replicas "$SERVICE"
        calculate_auto_scale "$SERVICE"

        if [ $TARGET_REPLICAS -ne $CURRENT_REPLICAS ]; then
            log_info "Auto-scaling from ${CURRENT_REPLICAS} to ${TARGET_REPLICAS} replicas"
            scale_service "$SERVICE" "$TARGET_REPLICAS"
            wait_for_healthy "$SERVICE" "$TARGET_REPLICAS"
            update_load_balancer
        else
            log_info "No scaling needed (current: ${CURRENT_REPLICAS})"
        fi

        sleep 60  # Check every minute
    done
}

main() {
    case "$ACTION" in
        scale)
            if [ -z "$SCALE_TO" ]; then
                log_error "Scale target not specified"
                echo "Usage: $0 <service> <replicas> [scale]"
                exit 1
            fi

            get_current_replicas "$SERVICE"
            TARGET_REPLICAS="$SCALE_TO"

            echo -e "${BLUE}========================================${NC}"
            echo -e "${BLUE}Scaling ${SERVICE}${NC}"
            echo -e "${BLUE}========================================${NC}"
            echo "Current replicas: ${CURRENT_REPLICAS}"
            echo "Target replicas: ${TARGET_REPLICAS}"
            echo ""

            scale_service "$SERVICE" "$TARGET_REPLICAS"
            wait_for_healthy "$SERVICE" "$TARGET_REPLICAS"
            update_load_balancer

            echo -e "${GREEN}Scaling completed successfully!${NC}"
            ;;

        auto)
            calculate_auto_scale "$SERVICE"

            echo -e "${BLUE}========================================${NC}"
            echo -e "${BLUE}Auto-scaling ${SERVICE}${NC}"
            echo -e "${BLUE}========================================${NC}"
            echo "Current replicas: ${CURRENT_REPLICAS}"
            echo "Recommended replicas: ${TARGET_REPLICAS}"
            echo ""

            if [ $TARGET_REPLICAS -ne $CURRENT_REPLICAS ]; then
                scale_service "$SERVICE" "$TARGET_REPLICAS"
                wait_for_healthy "$SERVICE" "$TARGET_REPLICAS"
                update_load_balancer
                echo -e "${GREEN}Auto-scaling completed!${NC}"
            else
                echo "No scaling needed"
            fi
            ;;

        monitor)
            monitor_auto_scale
            ;;

        rolling-update)
            local new_image="$SCALE_TO"
            if [ -z "$new_image" ]; then
                log_error "New image not specified"
                echo "Usage: $0 <service> <new-image> rolling-update"
                exit 1
            fi
            perform_rolling_update "$SERVICE" "$new_image"
            ;;

        status)
            show_status
            ;;

        *)
            log_error "Unknown action: $ACTION"
            exit 1
            ;;
    esac
}

# Handle script arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 <service> <target> [action]"
    echo ""
    echo "Actions:"
    echo "  scale           Scale service to target replicas (default)"
    echo "  auto            Auto-scale based on resource usage"
    echo "  monitor         Continuous auto-scaling monitor"
    echo "  rolling-update  Perform rolling update with new image"
    echo "  status          Show current status"
    echo ""
    echo "Examples:"
    echo "  $0 atlas-api 3 scale           # Scale to 3 replicas"
    echo "  $0 atlas-api auto              # Auto-scale based on load"
    echo "  $0 atlas-api monitor           # Start auto-scale monitor"
    echo "  $0 atlas-api v2.0 rolling-update  # Rolling update"
    echo "  $0 status                      # Show all services status"
    echo ""
    echo "Environment Variables:"
    echo "  MAX_REPLICAS                Maximum number of replicas"
    echo "  MIN_REPLICAS                Minimum number of replicas"
    echo "  AUTO_SCALE_CPU_THRESHOLD    CPU threshold for auto-scale (%)"
    echo "  AUTO_SCALE_MEM_THRESHOLD    Memory threshold for auto-scale (%)"
    exit 0
fi

# Special case for status
if [ "$1" = "status" ]; then
    show_status
    exit 0
fi

# Validate service argument
if [ -z "$SERVICE" ]; then
    log_error "Service not specified"
    exit 1
fi

# Run main function
main