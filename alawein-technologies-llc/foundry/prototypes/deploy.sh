#!/bin/bash

# Deployment script for Crazy Ideas Prototypes
# Usage: ./deploy.sh [local|staging|production]

set -e

ENVIRONMENT=${1:-local}

echo "🚀 Deploying Crazy Ideas Prototypes to ${ENVIRONMENT}"

# Load environment variables
if [ -f ".env.${ENVIRONMENT}" ]; then
    export $(cat .env.${ENVIRONMENT} | xargs)
else
    echo "⚠️  Warning: .env.${ENVIRONMENT} file not found"
fi

case $ENVIRONMENT in
    local)
        echo "📦 Building and starting local services..."
        docker-compose down
        docker-compose build --parallel
        docker-compose up -d

        echo "⏳ Waiting for services to be healthy..."
        sleep 10

        echo "✅ Services running at:"
        echo "   - Research Prison: http://localhost:8000"
        echo "   - Nightmare Mode: http://localhost:8001"
        echo "   - Chaos Engine: http://localhost:8002"
        echo "   - Flower (Celery): http://localhost:5555"
        ;;

    staging)
        echo "🔧 Deploying to staging environment..."

        # Build images with staging tags
        docker build -f Dockerfile.research-prison -t crazyideas/research-prison:staging .
        docker build -f Dockerfile.nightmare-mode -t crazyideas/nightmare-mode:staging .
        docker build -f Dockerfile.chaos-engine -t crazyideas/chaos-engine:staging .

        # Push to registry
        docker push crazyideas/research-prison:staging
        docker push crazyideas/nightmare-mode:staging
        docker push crazyideas/chaos-engine:staging

        # Deploy to staging server (example with SSH)
        ssh staging-server "cd /app && docker-compose pull && docker-compose up -d"
        ;;

    production)
        echo "🚨 Deploying to PRODUCTION..."
        echo "⚠️  This will affect live users. Are you sure? (yes/no)"
        read -r confirmation

        if [ "$confirmation" != "yes" ]; then
            echo "❌ Deployment cancelled"
            exit 1
        fi

        # Build production images
        docker build -f Dockerfile.research-prison -t crazyideas/research-prison:latest .
        docker build -f Dockerfile.nightmare-mode -t crazyideas/nightmare-mode:latest .
        docker build -f Dockerfile.chaos-engine -t crazyideas/chaos-engine:latest .

        # Run tests before deployment
        echo "🧪 Running tests..."
        ./test.sh

        if [ $? -ne 0 ]; then
            echo "❌ Tests failed! Aborting deployment."
            exit 1
        fi

        # Push to production registry
        docker push crazyideas/research-prison:latest
        docker push crazyideas/nightmare-mode:latest
        docker push crazyideas/chaos-engine:latest

        # Rolling deployment (example with Kubernetes)
        kubectl apply -f k8s/production/
        kubectl rollout status deployment/research-prison
        kubectl rollout status deployment/nightmare-mode
        kubectl rollout status deployment/chaos-engine
        ;;

    *)
        echo "❌ Invalid environment: ${ENVIRONMENT}"
        echo "Usage: ./deploy.sh [local|staging|production]"
        exit 1
        ;;
esac

echo "✅ Deployment complete!"

# Show service status
if [ "$ENVIRONMENT" == "local" ]; then
    docker-compose ps

    # Run health checks
    echo "🏥 Running health checks..."
    curl -s http://localhost:8000/api/health | jq '.'
    curl -s http://localhost:8001/api/health | jq '.'
    curl -s http://localhost:8002/api/health | jq '.'
fi

echo "🎉 All systems operational!"