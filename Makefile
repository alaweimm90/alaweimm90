# Makefile for Multi-Organization Docker Stack
# Simplifies common Docker operations

.PHONY: help setup up down restart logs ps test build clean validate deploy

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

welcome: ## Show welcome banner
	@cat .metaHub/WELCOME_BANNER.txt

setup: ## Initial setup (copy .env, create networks)
	@cat .metaHub/WELCOME_BANNER.txt
	@echo ""
	@echo "Setting up environment..."
	@cp -n .env.example .env || true
	@echo "✅ Environment configured"
	@docker network create frontend-network 2>/dev/null || true
	@docker network create backend-network 2>/dev/null || true
	@docker network create science-network 2>/dev/null || true
	@echo "✅ Networks created"
	@echo ""
	@echo "🎉 Setup complete! Run 'make up' to start all services"

up: ## Start all services in detached mode
	docker compose up -d
	@echo "✅ All services started"
	@echo "Run 'make logs' to see output"

down: ## Stop all services
	docker compose down
	@echo "✅ All services stopped"

restart: ## Restart all services
	docker compose restart
	@echo "✅ All services restarted"

logs: ## Show logs for all services
	docker compose logs -f

ps: ## Show running containers
	docker compose ps

test: ## Run tests in all containers
	docker compose -f docker-compose.yml -f docker-compose.test.yml up --abort-on-container-exit
	@echo "✅ Tests complete"

build: ## Build all images from scratch
	docker compose build --no-cache
	@echo "✅ All images built"

clean: ## Remove all containers, volumes, and images
	@echo "⚠️  This will remove all data. Press Ctrl+C to cancel, Enter to continue..."
	@read -r
	docker compose down -v --rmi all
	@echo "✅ Cleaned up"

validate: ## Validate docker-compose configuration
	docker compose config --quiet && echo "✅ Configuration valid" || echo "❌ Configuration invalid"

health: ## Check health of all services
	@docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Health}}"

deploy-staging: ## Deploy to staging environment
	@echo "Deploying to staging..."
	docker compose up -d
	@echo "✅ Deployed to staging"

deploy-prod: ## Deploy to production (requires confirmation)
	@echo "⚠️  Deploying to PRODUCTION. Press Ctrl+C to cancel, Enter to continue..."
	@read -r
	docker compose up -d
	@echo "✅ Deployed to production"

update-registry: ## Update projects registry with latest status
	pwsh -ExecutionPolicy Bypass -File .metaHub/scripts/update-containerization-registry.ps1

identify-tier2: ## Identify next Tier 2 projects
	pwsh -ExecutionPolicy Bypass -File .metaHub/scripts/identify-tier2-projects.ps1 -TopN 5

status: ## Show current phase status
	@cat .metaHub/STATUS.txt

stats: ## Show Docker resource usage
	docker stats --no-stream

prune: ## Remove unused Docker resources
	docker system prune -f
	@echo "✅ Pruned unused resources"

monitor: ## Monitor health status of all services
	pwsh -ExecutionPolicy Bypass -File .metaHub/scripts/docker-health-monitor.ps1

monitor-continuous: ## Continuously monitor health (Ctrl+C to stop)
	pwsh -ExecutionPolicy Bypass -File .metaHub/scripts/docker-health-monitor.ps1 -ContinuousMode -IntervalSeconds 30

backup: ## Backup all Docker volumes
	pwsh -ExecutionPolicy Bypass -File .metaHub/scripts/backup-volumes.ps1

backup-stop: ## Backup volumes with containers stopped
	pwsh -ExecutionPolicy Bypass -File .metaHub/scripts/backup-volumes.ps1 -StopContainers

restore: ## Restore volumes from backup (requires TIMESTAMP=yyyyMMdd-HHmmss)
	@if [ -z "$(TIMESTAMP)" ]; then \
		echo "❌ Error: TIMESTAMP required. Usage: make restore TIMESTAMP=20251124-123456"; \
		echo "Available backups:"; \
		ls -1 .metaHub/backups/volumes/ 2>/dev/null || echo "  No backups found"; \
		exit 1; \
	fi
	pwsh -ExecutionPolicy Bypass -File .metaHub/scripts/restore-volumes.ps1 -BackupTimestamp $(TIMESTAMP)

list-backups: ## List all available volume backups
	@echo "Available volume backups:"
	@ls -1 .metaHub/backups/volumes/ 2>/dev/null || echo "  No backups found"
