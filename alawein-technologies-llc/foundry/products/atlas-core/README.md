# ORCHEX Core - Shared Infrastructure

**Centralized services for all ORCHEX products: auth, payments, ML models, API gateway.**

## 🎯 What is This?

ORCHEX Core provides shared infrastructure so all ORCHEX products can:
- Use single sign-on (one account, access all products)
- Share payment processing (one subscription, multiple features)
- Access shared ML model zoo (fine-tuned models, embeddings)
- Route through unified API gateway
- Share monitoring and observability

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     API Gateway                          │
│         (nginx/Kong) - Routes to services                │
└───────────────┬─────────────────────────────────────────┘
                │
        ┌───────┴───────┐
        │               │
┌───────▼──────┐  ┌────▼──────────┐
│ Auth Service │  │ Payment Service│
│ (Supertokens)│  │ (Stripe)       │
└──────────────┘  └────────────────┘
                │
        ┌───────┴───────┐
        │               │
┌───────▼──────────┐  ┌─▼──────────┐
│  ML Model Zoo    │  │ Monitoring  │
│  (HuggingFace)   │  │ (Grafana)   │
└──────────────────┘  └─────────────┘
        │
  ┌─────┴─────┬──────────┬──────────┐
  │           │          │          │
┌─▼────────┐ ┌▼────────┐ ┌▼────────┐
│Nightmare │ │ Chaos   │ │Research │
│  Mode    │ │ Engine  │ │ Prison  │
└──────────┘ └─────────┘ └─────────┘
```

## 📂 Project Structure

```
ORCHEX-core/
├── api-gateway/
│   ├── nginx.conf
│   ├── routes.yml
│   └── rate-limiting.lua
│
├── auth-service/
│   ├── supertokens/          # SuperTokens config
│   ├── user-management/
│   └── session-management/
│
├── payment-service/
│   ├── stripe-integration/
│   ├── subscription-manager/
│   ├── usage-tracking/
│   └── invoicing/
│
├── ml-model-zoo/
│   ├── embeddings/
│   │   └── sentence-transformer/
│   ├── generators/
│   │   ├── gpt4-wrapper/
│   │   └── claude-wrapper/
│   └── custom-models/
│       └── hypothesis-classifier/
│
├── monitoring/
│   ├── prometheus/
│   ├── grafana/
│   │   └── dashboards/
│   ├── alertmanager/
│   └── logs/
│
├── shared-components/         # React components
│   ├── auth/
│   │   ├── LoginForm.tsx
│   │   └── SignupForm.tsx
│   ├── payments/
│   │   ├── PricingTable.tsx
│   │   └── CheckoutForm.tsx
│   └── ui/
│       ├── Button.tsx
│       ├── Card.tsx
│       └── Modal.tsx
│
├── database/
│   ├── migrations/
│   └── shared-schemas/
│       ├── users.sql
│       └── subscriptions.sql
│
└── docker-compose.yml
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Node.js 18+
- Python 3.10+

### Installation

```bash
# Clone
git clone https://github.com/your-org/ORCHEX-core.git
cd ORCHEX-core

# Start all services
docker-compose up -d

# Verify
curl http://localhost:8080/health
```

## 🔐 Authentication Service

### SuperTokens Integration

**Features:**
- Email/password auth
- Social login (Google, GitHub)
- Magic link sign-in
- JWT sessions
- Role-based access control (RBAC)

**Setup:**
```bash
cd auth-service
docker-compose up -d supertokens
```

**Usage in products:**
```typescript
// In any ORCHEX product
import { getSession } from '@ORCHEX/auth'

export async function handler(req, res) {
  const session = await getSession(req)
  if (!session) {
    return res.status(401).json({ error: 'Unauthorized' })
  }

  // User is authenticated
  const userId = session.getUserId()
  // ... your logic
}
```

## 💳 Payment Service

### Stripe Integration

**Features:**
- Subscription management
- Usage-based billing
- Invoice generation
- Webhook handling
- Tax calculation

**Pricing Tiers:**
```typescript
const PRICING = {
  free: {
    priceId: 'price_free',
    features: ['Basic access to all products']
  },
  basic: {
    priceId: 'price_basic',
    amount: 2900,  // $29/mo
    features: ['Unlimited Nightmare Mode', '100 Chaos collisions/mo']
  },
  pro: {
    priceId: 'price_pro',
    amount: 9900,  // $99/mo
    features: ['Everything', 'API access', 'Priority support']
  }
}
```

**Usage:**
```typescript
import { createCheckoutSession } from '@ORCHEX/payments'

const session = await createCheckoutSession({
  userId: 'user_123',
  priceId: 'price_basic',
  successUrl: '/success',
  cancelUrl: '/cancel'
})

// Redirect user to session.url
```

## 🤖 ML Model Zoo

### Shared Models

**1. Sentence Embeddings:**
```python
from atlas_core.ml import get_embeddings

embeddings = get_embeddings([
    "Quantum mechanics",
    "Social media"
])
# Returns: numpy arrays, cached for 24 hours
```

**2. GPT-4 Wrapper (with caching & rate limiting):**
```python
from atlas_core.ml import gpt4_generate

response = gpt4_generate(
    prompt="Generate hypothesis...",
    cache_key="hypothesis_123",  # Cached for 1 hour
    max_tokens=1000
)
```

**3. Custom Models:**
```python
from atlas_core.ml import hypothesis_classifier

score = hypothesis_classifier.predict(
    "Coffee improves cognitive performance"
)
# Returns: feasibility score 0-1
```

## 📊 Monitoring

### Grafana Dashboards

**System Health:**
- Request rate (per product)
- Error rate
- Response time (p50, p95, p99)
- Database connections

**Business Metrics:**
- Active users (DAU, MAU)
- Revenue (MRR, ARR)
- Conversion rate
- Churn rate

**AI Usage:**
- API calls to OpenAI/Anthropic
- Token usage
- Cost per user
- Cache hit rate

### Alerts

```yaml
# alertmanager/alerts.yml
alerts:
  - name: HighErrorRate
    condition: error_rate > 5%
    severity: critical
    notify: pagerduty

  - name: HighAPIUsage
    condition: openai_cost > $100/hour
    severity: warning
    notify: slack
```

## 🔧 Configuration

### Environment Variables

```bash
# Auth
SUPERTOKENS_CONNECTION_URI=...
SUPERTOKENS_API_KEY=...

# Payments
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...

# ML Models
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Database
DATABASE_URL=postgresql://...
REDIS_URL=redis://...

# Monitoring
SENTRY_DSN=...
```

## 🚢 Deployment

### Production Setup

```bash
# 1. Deploy to Kubernetes
kubectl apply -f k8s/

# 2. Or use Docker Swarm
docker stack deploy -c docker-compose.prod.yml ORCHEX-core

# 3. Or managed services
# - Auth: SuperTokens managed cloud
# - Database: AWS RDS
# - Cache: AWS ElastiCache
# - Monitoring: Datadog
```

## 📦 Using in Other Products

### Install Shared Package

```bash
# In any ORCHEX product
npm install @ORCHEX/core
```

### Import Shared Components

```typescript
// Frontend
import { LoginForm, Button, PricingTable } from '@ORCHEX/core/components'

// Backend
import { authenticateUser, processPayment } from '@ORCHEX/core/services'
```

## 💰 Cost Optimization

### Caching Strategy
- Embedding cache: 24 hours (saves $$$)
- GPT-4 cache: 1 hour (identical prompts)
- Redis for session storage

### Rate Limiting
- Free tier: 10 req/min
- Basic tier: 100 req/min
- Pro tier: 1000 req/min

## 🧪 Testing

```bash
# Test all services
docker-compose -f docker-compose.test.yml up

# Test specific service
cd auth-service && pytest
cd payment-service && pytest
```

## 🗺️ Roadmap

**Phase 1 (Week 1-2):**
- [ ] Auth service setup (SuperTokens)
- [ ] Basic API gateway
- [ ] PostgreSQL + Redis

**Phase 2 (Week 3-4):**
- [ ] Payment service (Stripe)
- [ ] Monitoring (Prometheus + Grafana)

**Phase 3 (Week 5-6):**
- [ ] ML model zoo
- [ ] Shared component library

## 📄 License

MIT License

---

**Shared infrastructure for the ORCHEX ecosystem** 🚀
