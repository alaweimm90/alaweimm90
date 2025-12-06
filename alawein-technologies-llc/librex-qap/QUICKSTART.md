# Librex.QAP-new Quick Start Guide

Get Librex.QAP-new running in 5 minutes.

---

## 1. Installation

### Option A: Local Development

```bash
# Clone the repository
cd Librex.QAP-new

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option B: Docker

```bash
# Build image
docker build -t Librex.QAP-new .

# Run
docker run -p 8000:8000 -p 8501:8501 Librex.QAP-new
```

---

## 2. Start the API Server

```bash
# Terminal 1: API Server
python server.py
```

Expected output:
```
==================================================
Librex.QAP-new API Server Starting
==================================================
Available methods: 8
Documentation: http://localhost:8000/docs
Health check: http://localhost:8000/health
==================================================
INFO:     Uvicorn running on http://0.0.0.0:8000
```

✅ Server is ready at `http://localhost:8000`

---

## 3. Test the API (In Another Terminal)

```bash
# Terminal 2: Run tests
python test_integration.py
```

Expected output:
```
============================================================
Librex.QAP-new Integration Test Suite
============================================================

▶ Health Check
  ✓ Server healthy: 2025-01-19T...

▶ Basic Solve Endpoint
  ✓ Solution found: obj=45.32, time=0.012s

▶ List Methods
  ✓ Found 8 methods
  ℹ fft_laplace: FFT-based Laplace preconditioning...
  ...

============================================================
Test Results: ✓ 11 passed, ✗ 0 failed
============================================================
```

✅ All tests passing!

---

## 4. Start the Dashboard (In Third Terminal)

```bash
# Terminal 3: Dashboard
streamlit run dashboard.py
```

Expected output:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
```

✅ Dashboard is ready at `http://localhost:8501`

---

## 5. Use the API

### Option A: Web Interface

1. **API Docs:** http://localhost:8000/docs (interactive Swagger UI)
2. **Dashboard:** http://localhost:8501 (visualization interface)

### Option B: Python Client

```python
import requests

# Solve a problem
response = requests.post("http://localhost:8000/solve", json={
    "problem_size": 10,
    "problem_matrix": [[float(i+j) for j in range(10)] for i in range(10)],
    "method": "fft_laplace",
    "iterations": 500
})

result = response.json()
print(f"Solution quality: {result['objective_value']:.2f}")
print(f"Runtime: {result['runtime_seconds']:.3f}s")

# List available methods
methods = requests.get("http://localhost:8000/methods").json()
for m in methods:
    print(f"- {m['name']}: {m['description']}")

# Get metrics
metrics = requests.get("http://localhost:8000/metrics").json()
print(f"Total optimizations: {metrics['total_optimizations']}")
```

### Option C: cURL

```bash
# Health check
curl http://localhost:8000/health

# Solve problem
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "problem_size": 5,
    "problem_matrix": [[0,1,2,3,4],[1,0,1,2,3],[2,1,0,1,2],[3,2,1,0,1],[4,3,2,1,0]],
    "method": "fft_laplace",
    "iterations": 100
  }'

# List methods
curl http://localhost:8000/methods
```

---

## 6. Try the Dashboard

Open http://localhost:8501 and:

1. **Overview Tab:** View service metrics
2. **Solve Problem Tab:**
   - Choose problem size (5-50)
   - Select optimization method
   - Click "Solve Now"
   - See results in real-time
3. **Benchmarks Tab:**
   - Run comparison across methods
   - Visualize quality vs runtime
4. **Methods Tab:**
   - Explore all available algorithms
   - See complexity and parameters
5. **Analytics Tab:**
   - Track performance metrics
   - View historical trends

---

## 7. Run Benchmarks

```python
import requests

# Run benchmark
response = requests.post("http://localhost:8000/benchmark", json={
    "instance_name": "nug20",
    "methods": ["fft_laplace", "reverse_time", "genetic_algorithm"],
    "num_runs": 5,
    "iterations_per_run": 500
})

results = response.json()
print(f"Benchmark {results['benchmark_id']} completed")
print(f"Total runs: {len(results['results'])}")

# Analyze results
for result in results['results']:
    print(f"  {result['method']}: quality={result['quality']:.2%}, "
          f"time={result['runtime_seconds']:.2f}s")
```

---

## 8. Docker Compose (All-in-One)

```bash
# Start all services
docker-compose up

# In another terminal, run tests
python test_integration.py
```

Services:
- **Librex.QAP**: Main service (http://localhost:8000)
- **jupyter**: Notebook environment (http://localhost:8888)
- **api**: FastAPI server (http://localhost:8000)
- **benchmarks**: Benchmark runner

---

## 9. Troubleshooting

### API not responding?
```bash
# Check if server is running
curl http://localhost:8000/health

# If not, start it
python server.py
```

### Dashboard not loading?
```bash
# Check if streamlit is running
# Start it if needed
streamlit run dashboard.py

# Clear cache if issues persist
streamlit cache clear
```

### Port already in use?
```bash
# Change API port
uvicorn server:app --port 9000

# Change Streamlit port
streamlit run dashboard.py --server.port 9501
```

### ImportError?
```bash
# Install missing dependencies
pip install -r requirements.txt

# Or update pip
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 10. Next Steps

### For Researchers
1. Read [DRAFT_FFT_LAPLACE_ICML2025.md](docs/papers/DRAFT_FFT_LAPLACE_ICML2025.md)
2. Explore [adding-optimization-methods.md](docs/guides/adding-optimization-methods.md)
3. Check [ORCHEX validation system](docs/guides/)
4. Contribute new methods!

### For ML Engineers
1. Review [API_ENDPOINT_TEMPLATE.md](docs/templates/API_ENDPOINT_TEMPLATE.md)
2. Explore [INTEGRATION_EXAMPLES.md](docs/templates/INTEGRATION_EXAMPLES.md)
3. Try dashboard.py with real datasets
4. Deploy with Docker

### For DevOps
1. Check [CLOUD_TEMPLATES.md](docs/deployment/CLOUD_TEMPLATES.md)
2. Review [PRODUCTION_BEST_PRACTICES.md](docs/deployment/PRODUCTION_BEST_PRACTICES.md)
3. Set up CI/CD pipeline
4. Deploy to cloud

### For Community
1. Join our [GitHub Discussions](https://github.com/AlaweinOS/AlaweinOS/discussions)
2. Check [CONTRIBUTING.md](CONTRIBUTING.md)
3. Help with [good-first-issues](https://github.com/AlaweinOS/AlaweinOS/labels/good-first-issue)
4. Share your results!

---

## 11. Project Structure

```
Librex.QAP-new/
├── server.py                 # FastAPI server (start here)
├── dashboard.py              # Streamlit dashboard
├── test_integration.py        # Integration tests
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container configuration
├── docker-compose.yml         # Multi-service setup
│
├── docs/
│   ├── guides/               # How-to guides
│   ├── templates/            # Code templates
│   ├── deployment/           # Cloud deployment
│   ├── community/            # Hiring & partnerships
│   ├── research/             # Publications & grants
│   └── papers/               # Research paper drafts
│
├── Librex.QAP/                # Core optimization library
├── ORCHEX/                    # Autonomous research system
├── tests/                    # Unit tests
└── data/                     # Benchmark instances
```

---

## 12. Documentation

- **[README.md](README.md)** - Project overview
- **[PROJECT.md](PROJECT.md)** - Detailed project description
- **[STRUCTURE.md](STRUCTURE.md)** - Directory guide
- **[GOVERNANCE.md](GOVERNANCE.md)** - Team & leadership
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute
- **[API Docs](http://localhost:8000/docs)** - Interactive API documentation

---

## 13. Support

### Resources
- 📖 Full documentation: [GitHub Wiki](https://github.com/AlaweinOS/AlaweinOS)
- 💬 Discussions: [GitHub Discussions](https://github.com/AlaweinOS/AlaweinOS/discussions)
- 🐛 Bug reports: [GitHub Issues](https://github.com/AlaweinOS/AlaweinOS/issues)
- 📧 Contact: [Email address]

### Getting Help
1. Check [FAQ](docs/FAQ.md)
2. Search [existing issues](https://github.com/AlaweinOS/AlaweinOS/issues)
3. Ask in [discussions](https://github.com/AlaweinOS/AlaweinOS/discussions)
4. Create a new issue with details

---

## 14. Performance Expectations

### API Response Times
- Health check: <50ms
- Basic solve (n=10): 10-50ms
- Solve (n=30): 1-3s
- Benchmark (5 runs): 5-10s

### Quality Metrics
- Typical gap: 0.5-2% from best-known
- Success rate: 98%+
- Consistency: Stable across runs

---

## 15. Example Workflow

```bash
# 1. Start API (Terminal 1)
python server.py

# 2. Start Dashboard (Terminal 2)
streamlit run dashboard.py

# 3. Run tests (Terminal 3)
python test_integration.py

# 4. Open browser
# API Docs: http://localhost:8000/docs
# Dashboard: http://localhost:8501

# 5. Solve a problem via dashboard
# - Set problem size to 20
# - Choose fft_laplace method
# - Click "Solve Now"
# - See real-time results

# 6. Compare methods
# - Go to Benchmarks tab
# - Select nug20 instance
# - Choose 3 methods
# - Run benchmark
# - Analyze results
```

---

**🚀 Ready to go! Start with `python server.py` and you're live.**
