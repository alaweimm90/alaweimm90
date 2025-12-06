# ORCHEX Nightmare Validator

**AI-powered adversarial review system that brutally attacks research papers from every angle.**

## 🎯 What is This?

Nightmare Mode is an AI system that:
- Attacks your research paper with 200+ specific critiques
- Uses multiple AI models (GPT-4, Claude) working together
- Tests across 6 dimensions: statistical, methodological, logical, historical, ethical, economic
- Gives you a "survival score" (0-100)
- Issues certification if your paper survives

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- Python 3.10+
- Docker (optional but recommended)
- PostgreSQL 15+
- Redis 7+

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ORCHEX-nightmare-validator.git
cd ORCHEX-nightmare-validator

# Setup backend
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # Add your API keys

# Setup database
docker-compose up -d postgres redis
python manage.py migrate

# Setup frontend
cd ../frontend
npm install
cp .env.example .env.local

# Start development servers
# Terminal 1: Backend
cd backend && uvicorn main:app --reload

# Terminal 2: Frontend
cd frontend && npm run dev

# Terminal 3: Worker (for async tasks)
cd backend && celery -A worker worker --loglevel=info
```

Visit http://localhost:3000

### Docker (Recommended)

```bash
docker-compose up
```

Visit http://localhost:3000

## 📂 Project Structure

```
nightmare-mode/
├── frontend/                # Next.js + React + TypeScript
│   ├── components/
│   │   ├── PaperUpload.tsx
│   │   ├── AttackVisualization.tsx
│   │   ├── SurvivalScore.tsx
│   │   └── CertificateDisplay.tsx
│   ├── pages/
│   │   ├── index.tsx        # Landing page
│   │   ├── upload.tsx       # Upload paper
│   │   ├── attack/[id].tsx  # Live attack view
│   │   └── certificate/[id].tsx
│   ├── lib/
│   │   ├── api.ts           # API client
│   │   └── websocket.ts     # Real-time updates
│   └── styles/
│
├── backend/                 # FastAPI + Python
│   ├── api/
│   │   ├── submissions.py   # POST /api/submissions
│   │   ├── attacks.py       # GET /api/attacks/{id}
│   │   └── certificates.py  # GET /api/certificates/{id}
│   ├── attack_agents/       # Core logic
│   │   ├── statistical_critic.py
│   │   ├── methodological_critic.py
│   │   ├── logical_critic.py
│   │   ├── historical_critic.py
│   │   ├── ethical_critic.py
│   │   └── economic_critic.py
│   ├── orchestrator/
│   │   ├── attack_coordinator.py  # Coordinates all agents
│   │   ├── ensemble.py            # Multi-model for Nightmare Mode
│   │   └── difficulty_manager.py  # Easy/Hard/Nightmare
│   ├── scoring/
│   │   ├── survival_calculator.py
│   │   └── certificate_issuer.py
│   ├── models.py            # Database models
│   ├── database.py          # DB connection
│   ├── worker.py            # Celery worker
│   └── main.py              # FastAPI app
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── API_REFERENCE.md
│   ├── ATTACK_FRAMEWORK.md
│   └── DEPLOYMENT.md
│
├── .github/
│   └── workflows/
│       ├── ci.yml           # Run tests on PR
│       └── deploy.yml       # Deploy to production
│
├── docker-compose.yml
├── Dockerfile
└── README.md
```

## 🎮 How It Works

### 1. Upload Paper
User uploads PDF or pastes text of their research paper.

### 2. Select Difficulty
- **Easy Mode:** Single agent, polite peer review (free)
- **Hard Mode:** 6 agents attack simultaneously ($9)
- **Nightmare Mode:** Multi-model ensemble, brutal ($29)

### 3. Attack Execution
System launches specialized AI critics:
- **Statistical Critic:** Finds p-hacking, multiple comparisons, sample size issues
- **Methodological Critic:** Identifies confounds, bias, poor controls
- **Logical Critic:** Spots non-sequiturs, contradictions, circular reasoning
- **Historical Critic:** Finds prior failed attempts
- **Ethical Critic:** Identifies potential harms
- **Economic Critic:** Questions cost/benefit

### 4. Survival Score
Based on:
- Number of attacks
- Severity of each attack
- Model consensus (if Nightmare Mode)
- Critical vs. minor flaws

### 5. Certification (Optional)
If survival score ≥70:
- Generate PDF certificate
- Unique verification ID
- Shareable badge
- Blockchain timestamp (optional)

## 🔧 Configuration

### Environment Variables

**Backend (.env):**
```bash
# AI Models
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Database
DATABASE_URL=postgresql://user:pass@localhost/nightmare_db
REDIS_URL=redis://localhost:6379

# App
SECRET_KEY=your-secret-key
DEBUG=True
CORS_ORIGINS=http://localhost:3000

# Pricing (cents)
EASY_MODE_PRICE=0
HARD_MODE_PRICE=900
NIGHTMARE_MODE_PRICE=2900
```

**Frontend (.env.local):**
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
NEXT_PUBLIC_STRIPE_KEY=pk_test_...
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Unit tests only
pytest tests/unit

# Integration tests
pytest tests/integration

# E2E tests
pytest tests/e2e

# With coverage
pytest --cov=backend --cov-report=html
```

## 🚢 Deployment

### Production Checklist
- [ ] Set all environment variables
- [ ] Setup PostgreSQL database
- [ ] Setup Redis
- [ ] Configure Stripe webhooks
- [ ] Setup monitoring (Sentry)
- [ ] Configure CDN (Cloudflare)
- [ ] Setup SSL certificates
- [ ] Configure rate limiting
- [ ] Setup backup system

### Deploy to Production

```bash
# Build Docker images
docker build -t nightmare-backend:latest ./backend
docker build -t nightmare-frontend:latest ./frontend

# Push to registry
docker push your-registry/nightmare-backend:latest
docker push your-registry/nightmare-frontend:latest

# Deploy (example with Docker Swarm)
docker stack deploy -c docker-compose.prod.yml nightmare
```

## 📊 Monitoring

- **Health check:** `GET /health`
- **Metrics:** `GET /metrics` (Prometheus format)
- **Logs:** Structured JSON logs
- **Alerts:** Configure in monitoring/alerts.yml

## 💰 Pricing

| Tier | Price | Features |
|------|-------|----------|
| Free | $0/mo | 3 Easy Mode attacks/month |
| Basic | $9/mo | Unlimited Easy, 10 Hard |
| Pro | $29/mo | Unlimited all modes |
| Teams | $99/mo | 5 seats, priority processing |

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 🆘 Support

- **Documentation:** https://docs.orchex-nightmare.com
- **Issues:** https://github.com/your-org/ORCHEX-nightmare-validator/issues
- **Email:** support@ORCHEX-nightmare.com
- **Discord:** https://discord.gg/ORCHEX

## 🗺️ Roadmap

### v1.0 (Week 12) - MVP
- [x] Easy Mode (single agent)
- [x] Hard Mode (6 agents)
- [x] Nightmare Mode (multi-model)
- [x] Survival scoring
- [x] Basic certification

### v1.1 (Week 16) - Social Features
- [ ] Spectator mode
- [ ] Replay system
- [ ] Public leaderboard
- [ ] Share to social media

### v1.2 (Week 20) - Tournament Mode
- [ ] 8 papers enter, 1 survives
- [ ] Bracket visualization
- [ ] Voting/betting

### v2.0 (Week 24) - Enterprise
- [ ] Team accounts
- [ ] Bulk upload
- [ ] Custom attack agents
- [ ] White-label option

## 📚 Learn More

- [Architecture Overview](docs/ARCHITECTURE.md)
- [API Reference](docs/API_REFERENCE.md)
- [Attack Framework](docs/ATTACK_FRAMEWORK.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

---

**Built with ❤️ by the ORCHEX team**

*"If your hypothesis can survive Nightmare Mode, it can survive anything."*
