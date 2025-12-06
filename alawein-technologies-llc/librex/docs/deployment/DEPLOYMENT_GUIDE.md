# Deployment Guide: Librex.QAP-new

**Comprehensive guide for deploying Librex.QAP-new in various environments**

---

## Quick Start Deployment

### Local Development
```bash
git clone https://github.com/AlaweinOS/AlaweinOS.git
cd AlaweinOS/Librex.QAP-new
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
make test
```

### Docker (Production Ready)
```bash
# Build image
docker build -t Librex.QAP-new:latest .

# Run container
docker run -it Librex.QAP-new:latest python -c "from Librex.QAP import *; print('✓ Ready')"

# Run tests
docker run Librex.QAP-new:latest pytest tests/
```

---

## Docker Deployment

### Dockerfile Template

**File:** `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -e .

# Run tests (optional)
RUN pytest tests/ --tb=short

# Default command
CMD ["python"]
```

### Docker Compose

**File:** `docker-compose.yml`

```yaml
version: '3.8'

services:
  Librex.QAP:
    build: .
    image: Librex.QAP-new:latest
    container_name: Librex.QAP-dev
    volumes:
      - .:/app
      - /app/__pycache__
    environment:
      - PYTHONUNBUFFERED=1
    command: python

  # Optional: Jupyter notebook
  jupyter:
    build: .
    image: Librex.QAP-new:latest
    ports:
      - "8888:8888"
    volumes:
      - .:/app
    command: jupyter notebook --ip=0.0.0.0 --allow-root
```

### Build & Run
```bash
docker-compose build
docker-compose run Librex.QAP pytest tests/
docker-compose up jupyter  # Access on localhost:8888
```

---

## CI/CD Pipeline

### GitHub Actions

**File:** `.github/workflows/tests.yml`

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -e ".[dev]"

    - name: Lint
      run: make lint

    - name: Tests
      run: make test

    - name: Coverage
      run: pytest --cov=Librex.QAP --cov=ORCHEX tests/
```

**File:** `.github/workflows/publish.yml`

```yaml
name: Publish

on:
  release:
    types: [created]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: pip install build twine

    - name: Build
      run: python -m build

    - name: Publish
      run: twine upload dist/*
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
```

---

## PyPI Package Release

### Setup

```bash
# Install build tools
pip install build twine

# Create ~/.pypirc with credentials
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-...

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-...
```

### Build & Release

```bash
# Build package
python -m build

# Test on TestPyPI
twine upload --repository testpypi dist/*

# Publish to PyPI
twine upload dist/*
```

---

## Cloud Deployment

### AWS EC2

```bash
# Launch instance
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.medium \
  --key-name my-key

# SSH into instance
ssh -i my-key.pem ec2-user@instance-ip

# Install and run
git clone https://github.com/AlaweinOS/AlaweinOS.git
cd Librex.QAP-new
pip install -e .
python -c "from Librex.QAP import *"
```

### Google Cloud Run

```dockerfile
# Dockerfile for Cloud Run
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e .
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
```

```bash
# Deploy
gcloud run deploy Librex.QAP-new \
  --source . \
  --platform managed \
  --region us-central1
```

### Azure Container Instances

```bash
# Build image
az acr build \
  --registry myregistry \
  --image Librex.QAP-new:latest .

# Run container
az container create \
  --resource-group mygroup \
  --name Librex.QAP-new \
  --image myregistry.azurecr.io/Librex.QAP-new:latest
```

---

## API Server Deployment

### FastAPI Server

**File:** `app.py`

```python
from fastapi import FastAPI
from Librex.QAP.core import OptimizationPipeline
from Librex.QAP.utils import load_qap_instance
import json

app = FastAPI(title="Librex.QAP API", version="0.1.0")

@app.get("/")
def root():
    return {"message": "Librex.QAP API Server"}

@app.post("/solve")
def solve(problem_data: dict):
    """Solve a QAP problem."""
    # Validate and parse
    pipeline = OptimizationPipeline(
        problem_size=problem_data.get('size', 20)
    )

    # Solve
    result = pipeline.solve(
        problem_data,
        method=problem_data.get('method', 'fft_laplace')
    )

    return {
        "solution": result.best_solution.tolist(),
        "objective_value": float(result.objective_value),
        "iterations": result.iterations
    }

@app.get("/methods")
def get_methods():
    """List available methods."""
    from Librex.QAP.methods import Metadata
    return {
        "methods": list(Metadata.all_methods().keys()),
        "count": len(Metadata.all_methods())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Run Server

```bash
# Install dependencies
pip install fastapi uvicorn

# Run locally
python app.py

# Run with production server
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

---

## Performance Tuning

### Database Caching
```python
# Cache benchmark results
import functools
import pickle

@functools.lru_cache(maxsize=128)
def cached_solve(problem_hash, method):
    # Load and solve
    pass
```

### Parallel Processing
```python
from multiprocessing import Pool

def benchmark_parallel(methods, instances):
    with Pool(4) as p:  # 4 workers
        results = p.starmap(solve, [
            (method, instance)
            for method in methods
            for instance in instances
        ])
    return results
```

### Monitoring

```python
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def solve_with_metrics(problem, method):
    start = time.time()
    logger.info(f"Starting solve with {method}")

    result = solve(problem, method)

    elapsed = time.time() - start
    logger.info(f"Completed in {elapsed:.2f}s")

    return result, elapsed
```

---

## Security Considerations

### Input Validation
```python
def validate_problem(problem_data):
    """Validate input problem."""
    if not isinstance(problem_data, dict):
        raise ValueError("Problem must be dict")
    if 'size' not in problem_data:
        raise ValueError("Size is required")
    if problem_data['size'] > 1000:
        raise ValueError("Size > 1000 not supported")
    return True
```

### Rate Limiting
```python
from slowapi import Limiter

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["5 per minute"]
)

@app.post("/solve")
@limiter.limit("5 per minute")
def solve(request: Request, problem_data: dict):
    # Solve...
    pass
```

### Authentication
```python
from fastapi.security import HTTPBearer
from fastapi import HTTPException

security = HTTPBearer()

@app.post("/solve")
def solve(problem_data: dict, credentials = Depends(security)):
    # Verify token
    if not verify_token(credentials.credentials):
        raise HTTPException(status_code=401)
    # Solve...
```

---

## Monitoring & Logging

### Structured Logging
```python
import logging
import json

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

logger.info(json.dumps({
    'event': 'optimization_started',
    'method': 'fft_laplace',
    'problem_size': 20
}))
```

### Metrics Collection
```python
import time
from prometheus_client import Counter, Histogram

solve_counter = Counter(
    'Librex.QAP_solves_total',
    'Total optimization solves',
    ['method']
)

solve_time = Histogram(
    'Librex.QAP_solve_time_seconds',
    'Time to solve',
    ['method']
)

@solve_time.labels(method='fft_laplace').time()
def solve(problem, method):
    result = optimize(problem, method)
    solve_counter.labels(method=method).inc()
    return result
```

---

## Backup & Recovery

### Database Backup
```bash
# Backup results
tar -czf results_backup.tar.gz results/

# Backup code
git tag -a v0.1.0 -m "Release 0.1.0"
git push origin v0.1.0
```

### Recovery Plan
1. Maintain git history (automatic)
2. Regular backups to S3/GCP
3. Test recovery monthly
4. Document recovery procedures

---

## Troubleshooting

### Common Issues

**Memory Error:**
```python
# Reduce problem size or use generators
for chunk in chunked_problems:
    result = solve(chunk)
```

**Slow Startup:**
```python
# Profile startup
python -m cProfile -s cumulative app.py
```

**High CPU:**
```python
# Check for infinite loops
import threading
for thread in threading.enumerate():
    print(thread.name)
```

---

## Deployment Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] Version bumped in setup.py
- [ ] CHANGELOG updated
- [ ] Performance benchmarked
- [ ] Security reviewed
- [ ] Docker image built and tested
- [ ] CI/CD pipeline green
- [ ] Release notes written
- [ ] Git tagged
- [ ] Published to PyPI
- [ ] Deployment monitored

---

**Happy deploying!** 🚀

Last Updated: November 2024
