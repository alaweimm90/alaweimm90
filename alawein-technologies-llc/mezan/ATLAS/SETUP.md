# ORCHEX Setup Guide

Quick setup instructions for the ORCHEX Hypothesis Evaluation Platform.

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Docker and Docker Compose (optional, for containerized deployment)

## Quick Start

### 1. Clone and Navigate

```bash
cd ORCHEX
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# For development
make install-dev

# Or manually:
pip install -r requirements-dev.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 5. Run the API Server

```bash
make run
# Or: python -m ORCHEX.atlas_api_server
```

The API will be available at `http://localhost:5000`

## Docker Setup

### Build and Run with Docker Compose

```bash
# Start all services (API, Redis, Prometheus, Grafana)
make docker-up

# Or manually:
docker-compose up -d
```

### Build Docker Image Only

```bash
make docker-build
docker run -p 5000:5000 --env-file .env ORCHEX-api:latest
```

## Verify Installation

1. **Health Check:**
   ```bash
   curl http://localhost:5000/health
   ```

2. **API Status:**
   ```bash
   curl http://localhost:5000/status
   ```

## Development Setup

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed development setup instructions.

## Troubleshooting

### Import Errors

If you see import errors, ensure:
- Virtual environment is activated
- Dependencies are installed: `pip install -r requirements-dev.txt`
- PYTHONPATH includes src: `export PYTHONPATH=$PWD/src:$PYTHONPATH`

### Port Already in Use

Change the port in `.env`:
```
API_PORT=5001
```

### Database Issues

The default SQLite database will be created automatically. For PostgreSQL, update `DATABASE_URL` in `.env`.

## Next Steps

- Read [START_HERE.md](START_HERE.md) for project overview
- Check [README.md](README.md) for feature documentation
- Review [CONTRIBUTING.md](CONTRIBUTING.md) to start contributing
