# TalAI Enterprise System

Production-ready enterprise infrastructure for TalAI platform with comprehensive authentication, billing, multi-tenancy, and administration capabilities.

## Features

### 1. Authentication & Authorization System (550+ LOC)
- **JWT-based authentication** with access and refresh tokens
- **Role-Based Access Control (RBAC)** with permissions
- **OAuth2 integration** (Google, GitHub, Microsoft)
- **Two-factor authentication** (2FA) with TOTP
- **API key management** for programmatic access
- **Team and organization management**
- **Session management** with Redis
- **Password policies** and history tracking
- **Comprehensive audit logging**

### 2. Billing & Subscription System (570+ LOC)
- **Stripe integration** for payments
- **Subscription tiers** (Free, Starter, Pro, Team, Enterprise)
- **Usage-based billing** with overage charges
- **Invoice generation** and management
- **Payment method management**
- **Trial periods** and promotional codes
- **Automated billing cycles**
- **Webhook handling** for Stripe events
- **Cost tracking** and projections

### 3. Multi-tenancy Architecture (590+ LOC)
- **Three isolation strategies**:
  - Row-level security (shared database)
  - Schema isolation (separate schemas)
  - Database isolation (separate databases)
- **Tenant resource management** and quotas
- **Per-tenant configuration** and features
- **Tenant migration** between isolation levels
- **Resource usage tracking** and limits
- **Tenant caching** with Redis
- **Data retention policies** by tier

### 4. Admin Dashboard Backend (580+ LOC)
- **System health monitoring** with alerts
- **User management** (bulk operations)
- **Analytics and reporting** dashboards
- **Feature flags** with gradual rollout
- **System configuration** management
- **Cost tracking** per organization
- **Performance metrics** with Prometheus
- **Audit log analysis**
- **Revenue analytics**

## Architecture

```
TalAI/enterprise/
├── requirements.txt         # Dependencies
├── database.py             # Database configuration (150 LOC)
├── auth_models.py          # Authentication models (450 LOC)
├── auth_service.py         # Authentication service (550 LOC)
├── auth_routes.py          # Auth API endpoints (580 LOC)
├── billing_models.py       # Billing models (480 LOC)
├── billing_service.py      # Billing service (570 LOC)
├── billing_routes.py       # Billing API endpoints (430 LOC)
├── tenancy_service.py      # Multi-tenancy service (590 LOC)
├── tenant_routes.py        # Tenant API endpoints (90 LOC)
├── admin_service.py        # Admin dashboard service (580 LOC)
├── admin_routes.py         # Admin API endpoints (80 LOC)
├── api_routes.py           # Core API endpoints (200 LOC)
├── main.py                 # FastAPI application (240 LOC)
└── README.md              # Documentation
```

**Total Lines of Code: ~5,000+**

## Installation

1. **Install dependencies:**
```bash
cd /home/user/AlaweinOS/TalAI/enterprise
pip install -r requirements.txt
```

2. **Set environment variables:**
```bash
# Database
export TALAI_DB_DATABASE_URL="postgresql+asyncpg://user:pass@localhost/talai"

# Authentication
export TALAI_SECRET_KEY="your-secret-key-here"

# OAuth2 Providers
export GOOGLE_CLIENT_ID="your-google-client-id"
export GOOGLE_CLIENT_SECRET="your-google-client-secret"
export GITHUB_CLIENT_ID="your-github-client-id"
export GITHUB_CLIENT_SECRET="your-github-client-secret"

# Stripe
export STRIPE_SECRET_KEY="your-stripe-secret-key"
export STRIPE_WEBHOOK_SECRET="your-webhook-secret"
export STRIPE_PUBLISHABLE_KEY="your-publishable-key"

# Redis
export REDIS_URL="redis://localhost:6379"

# Multi-tenancy
export TALAI_TENANT_ISOLATION="row_level"  # or "schema" or "database"
```

3. **Initialize database:**
```python
from database import init_database
import asyncio

asyncio.run(init_database())
```

4. **Run the application:**
```bash
python main.py
# Or with uvicorn:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Documentation

Once running, access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Key Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login with credentials
- `POST /api/v1/auth/oauth2/login` - OAuth2 login
- `POST /api/v1/auth/logout` - Logout
- `GET /api/v1/auth/me` - Get current user
- `POST /api/v1/auth/2fa/enable` - Enable 2FA

### Billing
- `GET /api/v1/billing/plans` - List subscription plans
- `POST /api/v1/billing/subscription` - Create subscription
- `GET /api/v1/billing/invoices` - List invoices
- `POST /api/v1/billing/usage` - Record usage

### Admin (System Admin Only)
- `GET /api/v1/admin/health` - System health
- `GET /api/v1/admin/stats` - System statistics
- `GET /api/v1/admin/analytics` - Analytics dashboard
- `PUT /api/v1/admin/feature-flags` - Update feature flags

### Tenants
- `POST /api/v1/tenants/` - Create tenant
- `GET /api/v1/tenants/context` - Get tenant context
- `GET /api/v1/tenants/{id}/usage` - Get resource usage

### Core API
- `POST /api/v1/execute` - Execute TalAI module
- `GET /api/v1/modules` - List available modules
- `GET /api/v1/quota` - Get API quota

## Security Features

- **Password Requirements:**
  - Minimum 8 characters
  - Uppercase, lowercase, digits, special characters
  - Password history (last 5 passwords)
  - Maximum age (90 days for enterprise)

- **Session Security:**
  - JWT tokens with short expiry (30 minutes)
  - Refresh tokens for extended sessions
  - Session invalidation on logout
  - IP-based session tracking

- **Rate Limiting:**
  - Per-user/organization limits
  - Tier-based quotas
  - Automatic blocking on excessive failures

- **Audit Logging:**
  - All authentication events
  - Administrative actions
  - API usage tracking
  - Security-relevant operations

## Monitoring

The system includes Prometheus metrics:
- Request counts and durations
- Active users gauge
- System resource usage
- Custom business metrics

Access metrics at: http://localhost:8000/metrics

## Testing

Run tests with:
```bash
pytest tests/ -v --cov=.
```

## Production Deployment

1. **Database:** Use managed PostgreSQL (AWS RDS, Google Cloud SQL)
2. **Redis:** Use managed Redis (AWS ElastiCache, Redis Cloud)
3. **Application:** Deploy with Docker/Kubernetes
4. **Load Balancer:** Use AWS ALB, Google Cloud Load Balancer
5. **Monitoring:** Integrate with Datadog, New Relic, or Grafana
6. **Logging:** Use centralized logging (ELK stack, CloudWatch)

## License

Copyright (c) 2024 AlaweinOS. All rights reserved.