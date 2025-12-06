.PHONY: help install dev build test deploy clean

help:
	@echo "Marketing Automation Platform - Available Commands"
	@echo "=================================================="
	@echo "install     - Install all dependencies"
	@echo "dev         - Start development environment"
	@echo "build       - Build all services"
	@echo "test        - Run test suite"
	@echo "deploy      - Deploy to production"
	@echo "clean       - Clean build artifacts"
	@echo "docker-up   - Start Docker containers"
	@echo "docker-down - Stop Docker containers"
	@echo "migrate     - Run database migrations"

install:
	@echo "Installing dependencies..."
	npm install
	npm run install:all

dev:
	@echo "Starting development environment..."
	docker-compose up -d postgres mongodb redis rabbitmq
	npm run dev

build:
	@echo "Building all services..."
	npm run build

test:
	@echo "Running tests..."
	npm test

deploy:
	@echo "Deploying to production..."
	npm run deploy

clean:
	@echo "Cleaning build artifacts..."
	find . -name "dist" -type d -exec rm -rf {} +
	find . -name "node_modules" -type d -exec rm -rf {} +

docker-up:
	@echo "Starting Docker containers..."
	docker-compose up -d

docker-down:
	@echo "Stopping Docker containers..."
	docker-compose down

migrate:
	@echo "Running database migrations..."
	npm run migrate

setup: install docker-up migrate
	@echo "Setup complete! Run 'make dev' to start development."
