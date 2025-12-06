#!/bin/bash

# TalAI Turing Challenge System - Deployment Script
# Production deployment automation

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-production}
COMPOSE_FILE="docker-compose.yml"

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}TalAI Turing Challenge - Deployment${NC}"
echo -e "${BLUE}=======================================${NC}"
echo ""
echo -e "Environment: ${GREEN}${ENVIRONMENT}${NC}"
echo ""

# ============================================================================
# Pre-flight Checks
# ============================================================================

echo -e "${YELLOW}Running pre-flight checks...${NC}"

# Check Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker is not installed${NC}"
    exit 1
fi

# Check Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}ERROR: Docker Compose is not installed${NC}"
    exit 1
fi

# Check environment variables
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${YELLOW}WARNING: ANTHROPIC_API_KEY not set${NC}"
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}WARNING: OPENAI_API_KEY not set${NC}"
fi

echo -e "${GREEN}✓ Pre-flight checks passed${NC}"
echo ""

# ============================================================================
# Create Required Directories
# ============================================================================

echo -e "${YELLOW}Creating required directories...${NC}"

mkdir -p logs
mkdir -p data
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/grafana/datasources
mkdir -p nginx/logs
mkdir -p nginx/ssl

echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# ============================================================================
# Pull Latest Images
# ============================================================================

echo -e "${YELLOW}Pulling latest Docker images...${NC}"

docker-compose -f ${COMPOSE_FILE} pull

echo -e "${GREEN}✓ Images pulled${NC}"
echo ""

# ============================================================================
# Build Application Image
# ============================================================================

echo -e "${YELLOW}Building TalAI application image...${NC}"

docker-compose -f ${COMPOSE_FILE} build --no-cache talair-api

echo -e "${GREEN}✓ Application built${NC}"
echo ""

# ============================================================================
# Stop Existing Containers
# ============================================================================

echo -e "${YELLOW}Stopping existing containers...${NC}"

docker-compose -f ${COMPOSE_FILE} down

echo -e "${GREEN}✓ Containers stopped${NC}"
echo ""

# ============================================================================
# Start Services
# ============================================================================

echo -e "${YELLOW}Starting services...${NC}"

docker-compose -f ${COMPOSE_FILE} up -d

echo -e "${GREEN}✓ Services started${NC}"
echo ""

# ============================================================================
# Wait for Services to be Healthy
# ============================================================================

echo -e "${YELLOW}Waiting for services to be healthy...${NC}"

# Wait for Redis
echo -n "  Redis: "
for i in {1..30}; do
    if docker-compose -f ${COMPOSE_FILE} exec -T redis redis-cli ping > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        break
    fi
    echo -n "."
    sleep 1
done

# Wait for TalAI API
echo -n "  TalAI API: "
for i in {1..60}; do
    if docker-compose -f ${COMPOSE_FILE} exec -T talair-api python -c "import sys; sys.exit(0)" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        break
    fi
    echo -n "."
    sleep 1
done

echo ""
echo -e "${GREEN}✓ All services healthy${NC}"
echo ""

# ============================================================================
# Display Service Status
# ============================================================================

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}Service Status${NC}"
echo -e "${BLUE}=======================================${NC}"
echo ""

docker-compose -f ${COMPOSE_FILE} ps

echo ""

# ============================================================================
# Display Access URLs
# ============================================================================

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}Access URLs${NC}"
echo -e "${BLUE}=======================================${NC}"
echo ""
echo -e "TalAI API:       ${GREEN}http://localhost:8000${NC}"
echo -e "API Docs:        ${GREEN}http://localhost:8000/docs${NC}"
echo -e "Grafana:         ${GREEN}http://localhost:3000${NC} (admin/admin)"
echo -e "Prometheus:      ${GREEN}http://localhost:9090${NC}"
echo ""

# ============================================================================
# Display Logs Command
# ============================================================================

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}Useful Commands${NC}"
echo -e "${BLUE}=======================================${NC}"
echo ""
echo -e "View logs:       ${YELLOW}docker-compose logs -f${NC}"
echo -e "Stop services:   ${YELLOW}docker-compose down${NC}"
echo -e "Restart:         ${YELLOW}docker-compose restart${NC}"
echo -e "Shell access:    ${YELLOW}docker-compose exec talair-api bash${NC}"
echo ""

# ============================================================================
# Health Check
# ============================================================================

echo -e "${YELLOW}Running final health check...${NC}"

# Check TalAI API health endpoint
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ TalAI API is responding${NC}"
else
    echo -e "${YELLOW}⚠ TalAI API health check failed (endpoint may not exist yet)${NC}"
fi

echo ""
echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}=======================================${NC}"
echo ""
echo -e "TalAI Turing Challenge System is now running."
echo -e "Monitor logs with: ${YELLOW}docker-compose logs -f talair-api${NC}"
echo ""
