#!/bin/bash

# ATLAS Repository Creator
# Creates a new repository from template
#
# Usage: ./create-repo.sh <template-name> <repo-name>
# Example: ./create-repo.sh nightmare-mode atlas-nightmare-validator

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -ne 2 ]; then
    echo -e "${RED}Error: Invalid arguments${NC}"
    echo "Usage: ./create-repo.sh <template-name> <repo-name>"
    echo ""
    echo "Available templates:"
    echo "  - nightmare-mode"
    echo "  - chaos-engine"
    echo "  - research-prison"
    echo "  - atlas-core"
    exit 1
fi

TEMPLATE_NAME=$1
REPO_NAME=$2
TEMPLATE_DIR="../templates/${TEMPLATE_NAME}"
TARGET_DIR="../../${REPO_NAME}"

echo -e "${GREEN}🚀 ATLAS Repository Creator${NC}"
echo ""

# Check if template exists
if [ ! -d "$TEMPLATE_DIR" ]; then
    echo -e "${RED}Error: Template '${TEMPLATE_NAME}' not found${NC}"
    echo "Available templates:"
    ls -1 ../templates/
    exit 1
fi

# Check if target directory already exists
if [ -d "$TARGET_DIR" ]; then
    echo -e "${RED}Error: Directory '${TARGET_DIR}' already exists${NC}"
    exit 1
fi

echo -e "${YELLOW}Creating repository from template...${NC}"
echo "  Template: ${TEMPLATE_NAME}"
echo "  Target: ${REPO_NAME}"
echo ""

# Create target directory
mkdir -p "$TARGET_DIR"

# Copy template
echo "📁 Copying template files..."
cp -r "$TEMPLATE_DIR/"* "$TARGET_DIR/"

# Create additional directories
echo "📁 Creating additional directories..."
cd "$TARGET_DIR"
mkdir -p {.github/workflows,tests/{unit,integration,e2e},docs,scripts}

# Create .gitignore
echo "📝 Creating .gitignore..."
cat > .gitignore << 'EOF'
# Dependencies
node_modules/
venv/
__pycache__/
*.pyc

# Environment
.env
.env.local
.env.*.local

# Build
dist/
build/
*.egg-info/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Database
*.db
*.sqlite

# Testing
.coverage
.pytest_cache/
htmlcov/
EOF

# Create .env.example files
echo "📝 Creating .env.example files..."

if [ -d "backend" ]; then
    cat > backend/.env.example << 'EOF'
# AI Models
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/dbname
REDIS_URL=redis://localhost:6379

# App
SECRET_KEY=your-secret-key-change-in-production
DEBUG=True
ENVIRONMENT=development

# CORS
CORS_ORIGINS=http://localhost:3000

# Optional: Monitoring
SENTRY_DSN=
EOF
fi

if [ -d "frontend" ]; then
    cat > frontend/.env.example << 'EOF'
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
NEXT_PUBLIC_STRIPE_KEY=pk_test_...
EOF
fi

# Create docker-compose.yml
echo "🐳 Creating docker-compose.yml..."
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: atlas
      POSTGRES_PASSWORD: atlas_dev_password
      POSTGRES_DB: atlas_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://atlas:atlas_dev_password@postgres:5432/atlas_db
      REDIS_URL: redis://redis:6379
    depends_on:
      - postgres
      - redis
    volumes:
      - ./backend:/app
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      NEXT_PUBLIC_API_URL: http://localhost:8000
    volumes:
      - ./frontend:/app
      - /app/node_modules
    command: npm run dev

volumes:
  postgres_data:
  redis_data:
EOF

# Create GitHub Actions CI
echo "⚙️ Creating GitHub Actions CI..."
cat > .github/workflows/ci.yml << 'EOF'
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_USER: atlas
          POSTGRES_PASSWORD: atlas_test
          POSTGRES_DB: atlas_test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install backend dependencies
        run: |
          cd backend
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run backend tests
        run: |
          cd backend
          pytest --cov=. --cov-report=xml
        env:
          DATABASE_URL: postgresql://atlas:atlas_test@localhost:5432/atlas_test_db
          REDIS_URL: redis://localhost:6379

      - name: Install frontend dependencies
        run: |
          cd frontend
          npm ci

      - name: Run frontend tests
        run: |
          cd frontend
          npm test

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./backend/coverage.xml
EOF

# Create requirements.txt for backend
if [ -d "backend" ]; then
    echo "📦 Creating requirements.txt..."
    cat > backend/requirements.txt << 'EOF'
# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6

# Database
sqlalchemy==2.0.23
asyncpg==0.29.0
alembic==1.12.1

# Cache
redis==5.0.1
aioredis==2.0.1

# AI Models
openai==1.3.5
anthropic==0.7.1

# ML/Data
numpy==1.26.2
pandas==2.1.3
scikit-learn==1.3.2
sentence-transformers==2.2.2

# Async Tasks
celery==5.3.4

# Auth & Security
pyjwt==2.8.0
passlib[bcrypt]==1.7.4
python-jose[cryptography]==3.3.0

# Payments
stripe==7.4.0

# Monitoring
sentry-sdk==1.38.0
prometheus-client==0.19.0

# Utilities
python-dotenv==1.0.0
pydantic==2.5.2
pydantic-settings==2.1.0

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
httpx==0.25.2
EOF
fi

# Create package.json for frontend
if [ -d "frontend" ]; then
    echo "📦 Creating package.json..."
    cat > frontend/package.json << EOF
{
  "name": "${REPO_NAME}",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "test": "jest"
  },
  "dependencies": {
    "next": "14.0.3",
    "react": "18.2.0",
    "react-dom": "18.2.0",
    "typescript": "5.3.2",
    "@types/node": "20.10.0",
    "@types/react": "18.2.42",
    "@types/react-dom": "18.2.17",
    "tailwindcss": "3.3.5",
    "axios": "1.6.2",
    "swr": "2.2.4"
  },
  "devDependencies": {
    "jest": "29.7.0",
    "@testing-library/react": "14.1.2",
    "@testing-library/jest-dom": "6.1.5",
    "eslint": "8.54.0",
    "eslint-config-next": "14.0.3"
  }
}
EOF
fi

# Initialize git
echo "🔧 Initializing git repository..."
git init
git add .
git commit -m "Initial commit from template: ${TEMPLATE_NAME}"

echo ""
echo -e "${GREEN}✅ Repository created successfully!${NC}"
echo ""
echo "Next steps:"
echo "  1. cd ${REPO_NAME}"
echo "  2. Review the README.md"
echo "  3. Setup environment:"
echo "     - Copy backend/.env.example to backend/.env"
echo "     - Copy frontend/.env.example to frontend/.env.local"
echo "     - Add your API keys"
echo "  4. Start development:"
echo "     docker-compose up"
echo "  5. Create GitHub repo and push:"
echo "     gh repo create ${REPO_NAME} --public --source=. --remote=origin"
echo "     git push -u origin main"
echo ""
echo -e "${YELLOW}Happy building! 🚀${NC}"
EOF
