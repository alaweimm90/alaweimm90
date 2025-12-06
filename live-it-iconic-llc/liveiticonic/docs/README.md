# Live It Iconic - Documentation Hub

**Complete documentation for the Live It Iconic wellness platform.**

---

## 📚 Quick Navigation

| I want to... | Go to |
|--------------|-------|
| **Get started quickly** | [Getting Started](#-getting-started) |
| **Learn about features** | [Guides](#-guides) |
| **Understand the architecture** | [Architecture](#-architecture) |
| **Deploy the platform** | [Deployment](#-deployment) |
| **Use the API** | [API Reference](#-api-reference) |
| **Find quick answers** | [Reference](#-reference) |
| **Understand AI features** | [AI & Automation](#-ai--automation) |

---

## 🚀 Getting Started

**New to Live It Iconic? Start here!**

### Installation & Setup
- [Installation Guide](./getting-started/installation.md) - Install dependencies and set up environment
- [Configuration Guide](./getting-started/configuration.md) - Configure environment variables and services
- [First Steps](./getting-started/first-steps.md) - Your first 15 minutes with the platform

### Quick Links
- 📖 [Root README](../README.md) - Project overview
- 🎯 [PROJECT.md](../PROJECT.md) - Comprehensive project details
- ⚡ [QUICK_START.md](../QUICK_START.md) - 5-minute quick start

---

## 📖 Guides

**Feature-specific guides and tutorials**

### Core Features
- [Wellness Features Guide](./guides/wellness-features.md) - Complete wellness platform guide
  - Wearable Integration
  - Mental Health Tracking
  - Habit Tracking & Streaks
  - Gamification System
  - Social Features
  - Nutrition Planning
  - AI Recommendations
  - Offline PWA
  - Privacy & Security

### Integration Guides
- [i18n Guide](./guides/i18n-guide.md) - Internationalization setup and usage
- [Stripe Integration](./guides/stripe-integration.md) - Payment processing setup
- [Social Media Integration](./guides/social-media.md) - Twitch, YouTube integration

### Development Guides
- [Testing Guide](./guides/testing-guide.md) - Unit, integration, and E2E testing
- [Storybook Guide](./guides/storybook.md) - Component development and documentation
- [Admin Dashboard](./guides/admin-dashboard.md) - Admin functionality guide

---

## 🏗️ Architecture

**System design and technical architecture**

### System Documentation
- [System Design](./architecture/system-design.md) - Overall architecture and design decisions
- [Database Schema](./architecture/database-schema.md) - PostgreSQL schema and relationships
- [Security Architecture](./architecture/security.md) - Security measures and best practices
- [Performance](./architecture/performance.md) - Performance optimization strategies

### Design Patterns
- **Services:** Singleton pattern for business logic
- **Components:** Functional components with hooks
- **State:** React Query + Context API
- **Types:** Comprehensive TypeScript definitions

---

## 📡 API Reference

**API documentation and integration**

- [API Overview](./api/README.md) - API introduction and authentication
- [Endpoints Reference](./api/endpoints.md) - Complete endpoint documentation
- [Authentication](./api/authentication.md) - Auth flows and token management
- [Webhooks](./api/webhooks.md) - Webhook integration guide

### Key APIs
- `/api/wellness/*` - Wellness data endpoints
- `/api/users/*` - User management
- `/api/social/*` - Social features
- `/api/nutrition/*` - Nutrition tracking

---

## 🚀 Deployment

**Deploy Live It Iconic to production**

### Deployment Guides
- [Infrastructure Guide](./deployment/infrastructure.md) - Cloud infrastructure setup
- [Monitoring Guide](./deployment/monitoring.md) - Monitoring and observability
- [Launch Checklist](./deployment/launch-checklist.md) - Pre-launch checklist

### Quick Deploy
See [DEPLOYMENT_INFRASTRUCTURE_PLAN.md](../DEPLOYMENT_INFRASTRUCTURE_PLAN.md) for complete infrastructure planning.

### Platforms
- **Frontend:** Vercel
- **Database:** Supabase
- **Cache:** Upstash Redis (optional)
- **Email:** Resend or SendGrid

---

## 📚 Reference

**Quick reference guides and cheat sheets**

- [Reference Card](./reference/reference-card.md) - Quick command reference
- [Environment Variables](./reference/environment-variables.md) - All environment variables explained
- [Commit Messages](./reference/commit-messages.md) - Commit message guidelines
- [Pull Requests](./reference/pull-requests.md) - PR process and templates
- [Governance](./reference/governance.md) - Project governance

---

## 🤖 AI & Automation

**AI features and automation guides**

- [Claude AI Prompts](./ai/claude-prompts.md) - Claude AI integration patterns
- [Superprompt](./ai/superprompt.md) - AI assistant superprompt

### AI Features
- Personalized recommendations
- Natural language meal logging
- Smart habit reminders
- Wellness insights

---

## 📋 Planning

**Business and strategic planning**

- [Business Plan](./planning/business-plan.md) - Business model and strategy

---

## 📦 Archive

**Historical documents and implementation reports**

See [archive/](./archive/) for:
- Implementation reports
- Accessibility audits
- Performance optimization reports
- Security implementation summaries

---

## 🗺️ Documentation Map

```
docs/
├── README.md (this file)           # Documentation hub
│
├── getting-started/                # New user guides
│   ├── installation.md
│   ├── configuration.md
│   └── first-steps.md
│
├── guides/                         # Feature guides
│   ├── wellness-features.md        # Main wellness guide
│   ├── i18n-guide.md               # Internationalization
│   ├── testing-guide.md            # Testing strategy
│   ├── storybook.md                # Component docs
│   ├── stripe-integration.md       # Payments
│   ├── social-media.md             # Social integrations
│   └── admin-dashboard.md          # Admin features
│
├── api/                            # API documentation
│   ├── README.md
│   ├── endpoints.md
│   ├── authentication.md
│   └── webhooks.md
│
├── architecture/                   # System design
│   ├── system-design.md
│   ├── database-schema.md
│   ├── security.md
│   └── performance.md
│
├── deployment/                     # Deployment guides
│   ├── infrastructure.md
│   ├── monitoring.md
│   └── launch-checklist.md
│
├── reference/                      # Quick reference
│   ├── reference-card.md
│   ├── environment-variables.md
│   ├── commit-messages.md
│   ├── pull-requests.md
│   └── governance.md
│
├── ai/                             # AI & automation
│   ├── claude-prompts.md
│   └── superprompt.md
│
├── planning/                       # Business planning
│   └── business-plan.md
│
└── archive/                        # Historical docs
    └── implementation-reports/
```

---

## 🔍 Search Tips

### By Topic

| Topic | Search Keywords |
|-------|----------------|
| Setup | installation, configuration, environment |
| Wellness | health, habits, nutrition, mental health |
| Social | community, posts, friends, groups |
| Technical | architecture, database, API |
| Deployment | infrastructure, monitoring, launch |
| Development | testing, storybook, contributing |

### By Role

| Role | Start Here |
|------|-----------|
| **New Developer** | getting-started/ |
| **Frontend Dev** | guides/ + architecture/ |
| **Backend Dev** | api/ + architecture/ |
| **DevOps** | deployment/ |
| **Designer** | guides/storybook.md |
| **PM/Stakeholder** | ../PROJECT.md + planning/ |

---

## 📝 Documentation Standards

### Writing Guidelines
1. **Clear & Concise** - Get to the point quickly
2. **Examples** - Show, don't just tell
3. **Up-to-date** - Keep in sync with code
4. **Accessible** - Write for all skill levels
5. **Searchable** - Use clear headings and keywords

### File Naming
- Use kebab-case: `feature-guide.md`
- Be descriptive: `stripe-integration.md` not `payments.md`
- Group related: `api/*.md`, `guides/*.md`

### Structure
```markdown
# Title

Brief introduction (1-2 sentences)

## Overview
What is this document about?

## Content Sections
Detailed information

## Examples
Code examples and use cases

## Related
Links to related documentation
```

---

## 🆘 Need Help?

### Can't Find What You're Looking For?

1. **Check Root Docs:**
   - [README.md](../README.md)
   - [PROJECT.md](../PROJECT.md)
   - [WELLNESS_PLATFORM_DOCUMENTATION.md](../WELLNESS_PLATFORM_DOCUMENTATION.md)

2. **Search the Codebase:**
   ```bash
   grep -r "your search term" docs/
   ```

3. **Ask the Community:**
   - [GitHub Discussions](https://github.com/alaweimm90-business/alaweimm90-business/discussions)
   - [GitHub Issues](https://github.com/alaweimm90-business/alaweimm90-business/issues)

4. **Check Archive:**
   - [archive/implementation-reports/](./archive/implementation-reports/)

---

## 🔄 Recently Updated

| Document | Last Updated | Changes |
|----------|--------------|---------|
| Documentation Hub | 2025-11-19 | Initial organization |
| Wellness Features | 2025-11-19 | Comprehensive platform docs |
| Deployment Plan | 2025-11-19 | Infrastructure options |

---

## ✅ Documentation Checklist

Maintaining good documentation:

- [ ] All guides have examples
- [ ] No broken links
- [ ] Code examples are tested
- [ ] Screenshots are up-to-date
- [ ] API docs match implementation
- [ ] Environment variables documented
- [ ] Architecture diagrams current
- [ ] Getting started is < 15 min

---

## 🎯 Next Steps

### New Users
1. Read [Installation Guide](./getting-started/installation.md)
2. Follow [First Steps](./getting-started/first-steps.md)
3. Explore [Wellness Features](./guides/wellness-features.md)

### Developers
1. Read [System Design](./architecture/system-design.md)
2. Review [API Reference](./api/endpoints.md)
3. Check [Testing Guide](./guides/testing-guide.md)

### Contributors
1. Read [CONTRIBUTING.md](../CONTRIBUTING.md)
2. Review [Commit Messages](./reference/commit-messages.md)
3. Check [Pull Request Guide](./reference/pull-requests.md)

---

**Questions?** Open a [Discussion](https://github.com/alaweimm90-business/alaweimm90-business/discussions) or [Issue](https://github.com/alaweimm90-business/alaweimm90-business/issues)

---

**Last Updated:** 2025-11-19
**Maintained By:** alaweimm90-business organization
**Version:** 1.0.0
