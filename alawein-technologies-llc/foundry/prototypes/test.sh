#!/bin/bash

# Test script for all prototypes
# Usage: ./test.sh [service]

set -e

SERVICE=${1:-all}

echo "🧪 Running tests for: ${SERVICE}"

run_tests() {
    local service=$1
    echo "Testing ${service}..."

    case $service in
        research-prison)
            echo "📚 Testing Research Prison API..."
            # Test paper submission
            curl -X POST http://localhost:8000/api/papers/submit \
                -H "Content-Type: application/json" \
                -d '{
                    "title": "Test Paper",
                    "abstract": "Test abstract",
                    "full_text": "Full text of the paper",
                    "authors": ["Test Author"],
                    "field": "Computer Science"
                }' | jq '.'

            # Test demo interrogation
            curl -X POST http://localhost:8000/api/demo/interrogate | jq '.'
            ;;

        nightmare-mode)
            echo "😈 Testing Nightmare Mode API..."
            # Test project submission
            curl -X POST http://localhost:8001/api/projects/submit \
                -H "Content-Type: application/json" \
                -d '{
                    "name": "Test Project",
                    "code_snippet": "def test(): pass",
                    "language": "python"
                }' | jq '.'

            # Test demo nightmare
            curl -X POST http://localhost:8001/api/demo/nightmare | jq '.'
            ;;

        chaos-engine)
            echo "🌀 Testing Chaos Engine API..."
            # Test random collision
            curl -X POST http://localhost:8002/api/collisions/generate \
                -H "Content-Type: application/json" \
                -d '{
                    "chaos_level": 7
                }' | jq '.'

            # Test demo chaos
            curl -X POST http://localhost:8002/api/demo/chaos | jq '.'
            ;;
    esac
}

# Wait for services to be ready
wait_for_service() {
    local url=$1
    local max_attempts=30
    local attempt=0

    echo "Waiting for $url to be ready..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -o /dev/null -w "%{http_code}" "$url" | grep -q "200"; then
            echo "✅ Service is ready!"
            return 0
        fi
        attempt=$((attempt + 1))
        echo "Attempt $attempt/$max_attempts..."
        sleep 2
    done

    echo "❌ Service failed to start"
    return 1
}

# Run tests based on service parameter
if [ "$SERVICE" == "all" ]; then
    # Wait for all services
    wait_for_service "http://localhost:8000/api/health"
    wait_for_service "http://localhost:8001/api/health"
    wait_for_service "http://localhost:8002/api/health"

    # Run all tests
    run_tests "research-prison"
    echo "---"
    run_tests "nightmare-mode"
    echo "---"
    run_tests "chaos-engine"
else
    # Run specific service test
    case $SERVICE in
        research-prison)
            wait_for_service "http://localhost:8000/api/health"
            run_tests "research-prison"
            ;;
        nightmare-mode)
            wait_for_service "http://localhost:8001/api/health"
            run_tests "nightmare-mode"
            ;;
        chaos-engine)
            wait_for_service "http://localhost:8002/api/health"
            run_tests "chaos-engine"
            ;;
        *)
            echo "❌ Unknown service: $SERVICE"
            echo "Usage: ./test.sh [all|research-prison|nightmare-mode|chaos-engine]"
            exit 1
            ;;
    esac
fi

echo "✅ All tests completed successfully!"

# Optional: Run Python unit tests if they exist
if [ -d "tests" ]; then
    echo "🐍 Running Python unit tests..."
    python -m pytest tests/ -v
fi

echo "🎉 Test suite complete!"