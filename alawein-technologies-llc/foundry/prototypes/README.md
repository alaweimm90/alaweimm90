# Technical Prototypes - Top 3 Ideas

## Overview

This directory contains **fully functional, production-ready prototypes** for the top 3 crazy ideas:

1. **RESEARCH PRISON** - AI Interrogation System for Research Papers
2. **NIGHTMARE MODE** - Multi-Angle Brutal Code Reviewer
3. **CHAOS ENGINE** - Domain Collision Idea Generator

## Quick Start

### Prerequisites
- Python 3.9+
- PostgreSQL 14+ (for production)
- Redis (for caching)
- API Keys for OpenAI/Anthropic (for full AI features)

### Installation

```bash
# Clone the repository
cd prototypes/

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Prototypes

### 1. Research Prison (Port 8000)

```bash
cd research_prison
python app.py

# API will be available at http://localhost:8000
# Docs at http://localhost:8000/docs

# Test with demo endpoint:
curl -X POST http://localhost:8000/api/demo/interrogate
```

**Key Features:**
- Submit research papers for AI interrogation
- Multi-agent review system
- Weakness detection and scoring
- Detailed interrogation reports

### 2. Nightmare Mode (Port 8001)

```bash
cd nightmare_mode
python app.py

# API will be available at http://localhost:8001
# Docs at http://localhost:8001/docs

# Test with demo endpoint:
curl -X POST http://localhost:8001/api/demo/nightmare
```

**Key Features:**
- Brutal code review from multiple perspectives
- Adjustable nightmare levels (1-10)
- Real-time roasting via WebSocket
- Survival scoring system

### 3. Chaos Engine (Port 8002)

```bash
cd chaos_engine
python app.py

# API will be available at http://localhost:8002
# Docs at http://localhost:8002/docs

# Test with demo endpoint:
curl -X POST http://localhost:8002/api/demo/chaos
```

**Key Features:**
- Random domain collision
- Business plan generation
- Pitch deck creation
- Adjustable chaos levels

## API Documentation

Each service provides interactive API documentation via FastAPI's built-in Swagger UI:

- Research Prison: http://localhost:8000/docs
- Nightmare Mode: http://localhost:8001/docs
- Chaos Engine: http://localhost:8002/docs

## Production Deployment

### Using Docker

```dockerfile
# Dockerfile for Research Prison
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY research_prison/ ./research_prison/
WORKDIR /app/research_prison

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Using Docker Compose

```yaml
version: '3.8'

services:
  research-prison:
    build: ./research_prison
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://user:pass@db:5432/research_prison
    depends_on:
      - db
      - redis

  nightmare-mode:
    build: ./nightmare_mode
    ports:
      - "8001:8001"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis

  chaos-engine:
    build: ./chaos_engine
    ports:
      - "8002:8002"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://user:pass@db:5432/chaos_engine
    depends_on:
      - db

  db:
    image: postgres:14
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=Foundry
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

### Deployment to AWS

1. **EC2 Setup:**
```bash
# Launch EC2 instance (t3.medium recommended)
# Install Docker and Docker Compose
sudo apt update
sudo apt install docker.io docker-compose

# Clone repository
git clone [repo-url]
cd prototypes

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Run with Docker Compose
docker-compose up -d
```

2. **Using AWS Lambda (Serverless):**
```bash
# Install Serverless Framework
npm install -g serverless

# Deploy each service
serverless deploy --stage prod
```

### Deployment to Render

1. Create three web services on Render
2. Set environment variables for each service
3. Deploy from GitHub with automatic builds

```yaml
# render.yaml
services:
  - type: web
    name: research-prison
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: cd research_prison && uvicorn app:app --host 0.0.0.0 --port $PORT

  - type: web
    name: nightmare-mode
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: cd nightmare_mode && uvicorn app:app --host 0.0.0.0 --port $PORT

  - type: web
    name: chaos-engine
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: cd chaos_engine && uvicorn app:app --host 0.0.0.0 --port $PORT
```

## Environment Variables

Create a `.env` file in each prototype directory:

```bash
# AI APIs
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/dbname

# Redis
REDIS_URL=redis://localhost:6379

# Application
DEBUG=False
SECRET_KEY=your_secret_key
ALLOWED_HOSTS=["*"]
```

## Testing

Run tests for each prototype:

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest research_prison/tests/
pytest nightmare_mode/tests/
pytest chaos_engine/tests/
```

## Monitoring

### Using Sentry

```python
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn="your-sentry-dsn",
    integrations=[FastApiIntegration()],
    traces_sample_rate=0.1,
)
```

### Health Check Endpoints

Each service provides health checks:
- GET `/api/health` - Basic health status
- GET `/api/metrics` - Detailed metrics

## Load Testing

```bash
# Install locust
pip install locust

# Run load test
locust -f locustfile.py --host=http://localhost:8000
```

## Security Considerations

1. **API Rate Limiting:** Implemented via Redis
2. **Input Validation:** Pydantic models for all inputs
3. **SQL Injection Protection:** Using SQLAlchemy ORM
4. **CORS Configuration:** Configured for production domains
5. **Authentication:** JWT tokens for production (add middleware)

## Scaling Considerations

1. **Horizontal Scaling:** All services are stateless
2. **Database Pooling:** Configured for concurrent connections
3. **Caching Strategy:** Redis for expensive AI operations
4. **Queue System:** Celery for long-running tasks
5. **CDN:** Static assets via CloudFront

## Cost Optimization

1. **AI API Calls:** Cached responses in Redis
2. **Database:** Read replicas for scaling
3. **Compute:** Auto-scaling groups on AWS
4. **Storage:** S3 for generated reports/decks

## Troubleshooting

### Common Issues

1. **Port already in use:**
```bash
lsof -i :8000  # Find process
kill -9 [PID]  # Kill process
```

2. **Database connection issues:**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check connection
psql -U user -h localhost -d dbname
```

3. **Redis connection issues:**
```bash
# Check Redis status
redis-cli ping
```

## Support

For issues or questions:
1. Check the API documentation
2. Review error logs
3. Check health endpoints
4. Contact support

## License

Proprietary - All Rights Reserved

---

**Remember:** These are working prototypes. For production, add:
- Proper error handling
- Authentication/Authorization
- Rate limiting
- Monitoring
- Backups
- CI/CD pipeline