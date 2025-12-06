#!/bin/sh
# Docker entrypoint script for MEZAN ATLAS
set -e

# Function to wait for a service
wait_for_service() {
    local host="$1"
    local port="$2"
    local service="$3"
    local max_attempts=30
    local attempt=1

    echo "Waiting for ${service} at ${host}:${port}..."

    while [ $attempt -le $max_attempts ]; do
        if nc -z "$host" "$port" 2>/dev/null; then
            echo "${service} is available!"
            return 0
        fi
        echo "Attempt $attempt/$max_attempts: ${service} not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done

    echo "ERROR: ${service} failed to become available"
    return 1
}

# Environment validation
validate_environment() {
    echo "Validating environment configuration..."

    # Check required environment variables
    if [ -z "$ANTHROPIC_API_KEY" ] && [ "$ATLAS_ENV" = "production" ]; then
        echo "WARNING: ANTHROPIC_API_KEY not set"
    fi

    if [ -z "$OPENAI_API_KEY" ] && [ "$ATLAS_ENV" = "production" ]; then
        echo "WARNING: OPENAI_API_KEY not set"
    fi

    echo "Environment validation complete"
}

# Initialize application
initialize_app() {
    echo "Initializing ATLAS application..."

    # Create necessary directories
    mkdir -p /app/logs /app/data /app/cache

    # Set proper permissions (if running as root, which shouldn't happen in prod)
    if [ "$(id -u)" = "0" ]; then
        chown -R atlas:atlas /app/logs /app/data /app/cache
    fi

    # Initialize database if needed
    if [ -n "$POSTGRES_HOST" ]; then
        wait_for_service "$POSTGRES_HOST" "${POSTGRES_PORT:-5432}" "PostgreSQL"
    fi

    # Wait for Redis if configured
    if [ -n "$REDIS_HOST" ]; then
        wait_for_service "$REDIS_HOST" "${REDIS_PORT:-6379}" "Redis"
    fi

    echo "Application initialization complete"
}

# Configure logging
setup_logging() {
    echo "Configuring logging..."

    # Set log level
    export LOG_LEVEL=${LOG_LEVEL:-INFO}

    # Create log directory structure
    mkdir -p /app/logs/access /app/logs/error /app/logs/debug

    # Rotate logs if they're too large
    for logfile in /app/logs/*.log; do
        if [ -f "$logfile" ] && [ $(stat -f%z "$logfile" 2>/dev/null || stat -c%s "$logfile") -gt 104857600 ]; then
            mv "$logfile" "$logfile.$(date +%Y%m%d)"
            touch "$logfile"
        fi
    done

    echo "Logging configured"
}

# Health check function
health_check() {
    curl -f http://localhost:${PORT:-8080}/health || exit 1
}

# Graceful shutdown handler
shutdown_handler() {
    echo "Received shutdown signal, gracefully stopping..."

    # Send SIGTERM to all processes
    kill -TERM $(jobs -p) 2>/dev/null || true

    # Wait for processes to finish
    wait

    echo "Shutdown complete"
    exit 0
}

# Trap signals for graceful shutdown
trap shutdown_handler SIGTERM SIGINT SIGQUIT

# Main execution
main() {
    echo "=========================================="
    echo "MEZAN ATLAS Docker Container Starting"
    echo "Environment: ${ATLAS_ENV:-production}"
    echo "Python: $(python --version)"
    echo "=========================================="

    # Run initialization steps
    validate_environment
    setup_logging
    initialize_app

    # Export Flask configuration
    export FLASK_ENV=${FLASK_ENV:-production}
    export FLASK_APP=${FLASK_APP:-atlas.atlas_api_server}

    # Start the application based on environment
    if [ "$ATLAS_ENV" = "development" ]; then
        echo "Starting ATLAS in DEVELOPMENT mode..."
        exec python -m flask run --host=0.0.0.0 --port=${PORT:-8080} --reload
    else
        echo "Starting ATLAS in PRODUCTION mode..."

        # Use gunicorn for production with proper settings
        exec gunicorn \
            --bind 0.0.0.0:${PORT:-8080} \
            --workers ${WORKERS:-4} \
            --threads ${THREADS:-2} \
            --timeout ${TIMEOUT:-120} \
            --keepalive ${KEEPALIVE:-5} \
            --max-requests ${MAX_REQUESTS:-1000} \
            --max-requests-jitter ${MAX_REQUESTS_JITTER:-50} \
            --access-logfile /app/logs/access.log \
            --error-logfile /app/logs/error.log \
            --log-level ${LOG_LEVEL:-info} \
            --capture-output \
            --enable-stdio-inheritance \
            --worker-class ${WORKER_CLASS:-sync} \
            --worker-tmp-dir /dev/shm \
            --graceful-timeout ${GRACEFUL_TIMEOUT:-30} \
            --preload \
            ${GUNICORN_EXTRA_ARGS:-} \
            "atlas.atlas_api_server:app"
    fi
}

# Execute command or run main
if [ "$#" -eq 0 ]; then
    main
else
    exec "$@"
fi