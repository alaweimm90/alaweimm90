# Rapid Deployment Playbook

> Streamlined procedures for deploying projects across all LLCs.

## Quick Start Matrix

| Project Type | Template | Deploy Target | Time |
|--------------|----------|---------------|------|
| Python Library | `templates/python-library/` | PyPI | 15 min |
| React SaaS | `templates/landing-page/` | Vercel | 10 min |
| Full-Stack App | Lovable → GitHub | Vercel + Supabase | 30 min |
| API Service | FastAPI template | Railway/Render | 20 min |

---

## Pre-Deployment Checklist

### Required Files
- [ ] `README.md` - Project description
- [ ] `LICENSE` - MIT/Apache-2.0
- [ ] `.meta/repo.yaml` - Project metadata
- [ ] `CONTRIBUTING.md` - Contribution guide
- [ ] `SECURITY.md` - Security policy

### CI/CD Requirements
- [ ] `.github/workflows/ci.yml` - Lint + Test
- [ ] `.github/workflows/deploy.yml` - Auto-deploy
- [ ] `CODEOWNERS` - Review assignments

---

## Deployment Procedures

### 1. Python Library → PyPI

```bash
# From project root
cd alawein-technologies-llc/<project>

# Build and test
pip install -e ".[dev]"
pytest

# Build package
python -m build

# Upload to PyPI
twine upload dist/*
```

### 2. React App → Vercel

```bash
# From project root
cd <llc>/<project>

# Install and build
npm install
npm run build

# Deploy
vercel --prod
```

### 3. Full-Stack → Supabase + Vercel

```bash
# 1. Setup Supabase
supabase init
supabase db push

# 2. Configure environment
cp .env.example .env.local
# Edit SUPABASE_URL, SUPABASE_ANON_KEY

# 3. Deploy frontend
vercel --prod
```

---

## Environment Variables

### Required for All Projects

| Variable | Source | Description |
|----------|--------|-------------|
| `SUPABASE_URL` | Supabase Dashboard | Project URL |
| `SUPABASE_ANON_KEY` | Supabase Dashboard | Public anon key |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase Dashboard | Server-side only |

### Project-Specific

| Project | Variables |
|---------|-----------|
| REPZ | `STRIPE_KEY`, `RESEND_API_KEY` |
| TalAI | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` |
| Librex | `PYPI_TOKEN` |

---

## Domain Configuration

| Project | Domain | Status |
|---------|--------|--------|
| Librex | `librex.dev` | Owned |
| TalAI | `talai.dev` | Owned |
| REPZ | `repz.app` | TBD |
| LiveItIconic | `liveiticonic.com` | TBD |

---

## Monitoring & Alerts

### Vercel
- Automatic deployment previews on PR
- Production deployment on merge to `main`
- Error tracking via Sentry integration

### Supabase
- Database health in dashboard
- Edge function logs
- Auth analytics

---

## Rollback Procedures

### Vercel
```bash
# List deployments
vercel ls

# Rollback to previous
vercel rollback <deployment-url>
```

### Supabase
```bash
# Rollback migration
supabase db reset --db-url <url>
```

---

## Post-Deployment

1. **Verify** - Check all endpoints respond
2. **Monitor** - Watch error rates for 30 min
3. **Announce** - Update changelog
4. **Tag** - Create git tag for release

```bash
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

---

## Related Documentation

- [LLC Registry](../LLC_PROJECT_REGISTRY.md) - Project ownership
- [Master AI Spec](../MASTER_AI_SPECIFICATION.md) - AI configuration
- [Architecture](../ARCHITECTURE.md) - System design

