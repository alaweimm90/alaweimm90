# Products - Code Templates

4 production-ready code templates for AI research tools, each ready to be scaffolded into a full repository.

---

## Quick Start

### Option 1: Start with Nightmare Mode (Recommended)

**Best if you want:** Maximum impact, highest feasibility

```bash
cd scripts/
./create-repo.sh nightmare-mode ORCHEX-nightmare-validator
cd ../../ORCHEX-nightmare-validator
docker-compose up
```

### Option 2: Choose by Timeline

**Fastest to MVP:** Research Prison (8 weeks)
```bash
./create-repo.sh research-prison ORCHEX-interrogator
```

**Best Technology:** Nightmare Mode (12 weeks)
```bash
./create-repo.sh nightmare-mode ORCHEX-nightmare-validator
```

**Most Creative:** Chaos Engine (10 weeks)
```bash
./create-repo.sh chaos-engine ORCHEX-chaos-engine
```

### Option 3: Create All Products

```bash
cd scripts/
./create-all-repos.sh
```

---

## Products Overview

### 1. NIGHTMARE MODE ⭐ (Recommended First Product)

**What it does:** AI attacks research papers from 6 different angles

**Key features:**
- 200+ attack vectors (statistical, methodological, logical, ethical, historical, economic)
- Multi-model ensemble (GPT-4, Claude)
- Survival score certification
- Paper weakness report generation

**Use cases:**
- Researchers want to stress-test their papers
- Institutions want to ensure research quality
- Authors want to strengthen papers before publication

**Technology:**
- Backend: Python, FastAPI, Celery
- Frontend: Next.js, React
- DB: PostgreSQL
- Deployment: Docker, Kubernetes

**Timeline:** 12 weeks to MVP

**Feasibility:** 10/10 | **Impact:** 10/10 | **Market Size:** Large

**Financial Projection:** $50-100K MRR at scale

---

### 2. CHAOS ENGINE

**What it does:** Cross-domain hypothesis generator using analogy and collision

**Key features:**
- Generates testable hypotheses across domains
- Analogy-based collision engine
- Hypothesis quality scoring
- Integration with research networks

**Use cases:**
- Research teams want fresh ideas
- Academics seeking cross-domain insights
- Innovation workshops need idea generation

**Technology:**
- Backend: Python, FastAPI
- Frontend: Next.js, React
- ML: GPT-4, Claude
- DB: PostgreSQL

**Timeline:** 10 weeks to MVP

**Feasibility:** 9/10 | **Impact:** 9/10 | **Market Size:** Medium-Large

**Financial Projection:** $30-80K MRR at scale

---

### 3. RESEARCH PRISON

**What it does:** 200-question interrogation system to validate research ideas

**Key features:**
- Good Cop & Bad Cop personas
- 200+ research validation questions
- Weakness identification system
- Auto-generated improvement suggestions

**Use cases:**
- Researchers validate ideas before starting
- Thesis writers strengthen proposals
- PhD students pressure-test research direction

**Technology:**
- Backend: Python, FastAPI
- Frontend: Next.js, React
- ML: Claude, GPT-4
- DB: PostgreSQL

**Timeline:** 8 weeks to MVP (Fastest!)

**Feasibility:** 9/10 | **Impact:** 8/10 | **Market Size:** Medium

**Financial Projection:** $20-60K MRR at scale

---

### 4. ORCHEX CORE

**What it does:** Shared infrastructure for all products

**Key features:**
- Unified authentication system
- Shared AI model zoo (GPT-4, Claude, embeddings)
- Common database schemas
- Shared API gateway
- Analytics and logging

**Use cases:**
- Multi-product infrastructure
- Scaling across products
- Cost optimization via shared resources

**Technology:**
- Backend: Python, FastAPI
- Auth: OAuth2, JWT
- ML: Multiple model APIs
- Infrastructure: Kubernetes

**Timeline:** 4 weeks to setup + ongoing

**Note:** Build NIGHTMARE MODE first, then add ORCHEX CORE for Phase 2

---

## Directory Structure

Each product follows this pattern:

```
product-name/
├── README.md                    # Product-specific overview
├── requirements.txt             # Python dependencies
├── setup.py                     # Python package setup
├── .env.example                 # Environment variables template
├── Dockerfile                   # Docker image
├── docker-compose.yml           # Local development setup
├── pytest.ini                   # Test configuration
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI entry point
│   │   ├── models.py           # Database models
│   │   ├── schemas.py          # Pydantic schemas
│   │   ├── config.py           # Configuration
│   │   ├── database.py         # Database connection
│   │   └── routers/
│   │       ├── __init__.py
│   │       └── api_v1.py       # API routes
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py         # Pytest fixtures
│       └── test_api.py         # API tests
│
├── alembic/                     # Database migrations
│   ├── versions/
│   │   └── 001_initial.py      # Initial schema
│   ├── env.py
│   └── script.py.mako
│
├── k8s/                         # Kubernetes configs
│   ├── deployment.yaml
│   ├── service.yaml
│   └── production.yaml
│
├── .github/
│   └── workflows/
│       ├── ci.yml              # Test on push
│       └── cd.yml              # Deploy on release
│
└── docs/
    ├── API.md                  # API documentation
    ├── SETUP.md                # Development setup
    └── DEPLOYMENT.md           # Production deployment
```

---

## Choosing Which Product to Build

### Build NIGHTMARE MODE if you want:
- ✅ Maximum market impact
- ✅ Highest feasibility
- ✅ Premium pricing potential
- ✅ Best B2B opportunity
- ✅ Technical challenge

### Build CHAOS ENGINE if you want:
- ✅ Pure AI/ML focus
- ✅ Fastest scaling potential
- ✅ B2C opportunity
- ✅ Creative application
- ✅ Easy to demo

### Build RESEARCH PRISON if you want:
- ✅ Fastest time to MVP (8 weeks)
- ✅ Lowest technical complexity
- ✅ Easiest to validate
- ✅ Quick revenue potential
- ✅ Simple to explain

### Build ORCHEX CORE if you want:
- ✅ Multi-product platform
- ✅ Cost optimization
- ✅ Unified infrastructure
- ⚠️ Only after one product succeeds

---

## Development Setup

### Prerequisites
- Python 3.10+
- Node.js 18+ (if using Next.js frontend)
- PostgreSQL 15+
- Docker & Docker Compose
- Git

### Quick Start

```bash
# 1. Clone/scaffold product
cd scripts/
./create-repo.sh nightmare-mode ORCHEX-nightmare-validator

# 2. Setup backend
cd ORCHEX-nightmare-validator/backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# 3. Setup database
alembic upgrade head

# 4. Run development server
uvicorn app.main:app --reload

# 5. Run tests
pytest tests/ -v

# 6. Run everything with Docker
docker-compose up -d
```

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_api.py -v

# Run specific test
pytest tests/test_api.py::test_function_name -v
```

---

## Database Migrations

```bash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

---

## Deployment

### Docker
```bash
# Build image
docker build -t ORCHEX-product:latest .

# Run container
docker run -p 8000:8000 ORCHEX-product:latest
```

### Kubernetes
```bash
# Apply configuration
kubectl apply -f k8s/production.yaml

# Check status
kubectl get pods

# View logs
kubectl logs deployment/ORCHEX-product
```

---

## API Documentation

Each product includes auto-generated API docs:

```
Local: http://localhost:8000/docs (Swagger UI)
Production: https://api.yourproduct.com/docs
```

---

## Product Roadmap

### Phase 1: MVP (Weeks 1-8)
- ✅ Core functionality
- ✅ Basic API
- ✅ Simple frontend
- ✅ 50-100 beta users

### Phase 2: Polish (Weeks 9-16)
- ✅ UI improvements
- ✅ Performance optimization
- ✅ 100+ paying users
- ✅ $3-5K MRR

### Phase 3: Scale (Weeks 17-24)
- ✅ Advanced features
- ✅ Integrations
- ✅ 200-500 users
- ✅ $10-20K MRR

### Phase 4: Platform (Months 7-12)
- ✅ Multi-product support
- ✅ Enterprise features
- ✅ $50-100K MRR

---

## Contributing

### Adding Features
1. Create feature branch: `feature/feature-name`
2. Make changes and test
3. Submit PR with description
4. See `../CONTRIBUTING.md` for full guidelines

### Reporting Issues
1. Include error message
2. Reproduction steps
3. Expected vs actual behavior
4. Environment details (Python version, OS, etc.)

---

## FAQs

### Which product should I start with?
**NIGHTMARE MODE** (highest feasibility, highest impact) or **RESEARCH PRISON** (fastest to MVP)

### Can I combine products?
Yes! Build one to $10K MRR, then add another. Eventually create ORCHEX CORE platform.

### How much does it cost to build?
- Solo: $20-30K (outsource some work)
- Small team: $50-100K
- Full team: $100-200K+

### How long does it take?
- Research Prison: 8 weeks
- Nightmare Mode: 12 weeks
- Chaos Engine: 10 weeks
- Full platform: 6+ months

### What about hiring?
- Weeks 1-4: Solo (validation)
- Weeks 5-16: Hire 1-2 engineers
- Weeks 17+: Hire product, design, marketing

---

## Resources

### External Documentation
- **FastAPI:** https://fastapi.tiangolo.com/
- **PostgreSQL:** https://www.postgresql.org/docs/
- **Kubernetes:** https://kubernetes.io/docs/
- **Docker:** https://docs.docker.com/

### Internal Documentation
- `../START_HERE.md` - Master index
- `../IMPLEMENTATION_GUIDE.md` - Build playbook
- `../docs/guides/GETTING_STARTED.md` - Dev setup
- `../docs/roadmaps/PHASE_1_QUICK_WINS.md` - 16-week plan

---

## Next Steps

1. **Choose a product** (NIGHTMARE MODE recommended)
2. **Run setup script** (`./create-repo.sh`)
3. **Follow GETTING_STARTED.md** in docs/guides/
4. **Read your product's README.md**
5. **Start building!**

---

**Ready to build? Start with NIGHTMARE MODE!** 🚀

*Questions? See `../FAQ.md`* 📚
