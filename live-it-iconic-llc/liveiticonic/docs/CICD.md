# Comprehensive CI/CD Pipeline Documentation

## Overview

This project implements a complete, production-grade CI/CD pipeline using GitHub Actions. The pipeline automates testing, building, security scanning, performance monitoring, and deployments across development, staging, and production environments.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Actions Workflows                 │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   CI.yml     │  │ Security.yml │  │Performance.yml       │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤      │
│  │ - Lint       │  │ - npm audit  │  │ - Lighthouse │      │
│  │ - TypeCheck  │  │ - CodeQL     │  │ - Performance│      │
│  │ - Unit Tests │  │              │  │   Monitoring │      │
│  │ - E2E Tests  │  │              │  │              │      │
│  │ - Build      │  │              │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌────────────────────────────────────────────────┐         │
│  │         Deployment Workflows                    │         │
│  ├────────────────────────────────────────────────┤         │
│  │  Deploy-Production.yml     Deploy-Staging.yml  │         │
│  │  (Vercel)                  (Netlify)          │         │
│  └────────────────────────────────────────────────┘         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Workflows

### 1. Main CI Pipeline (.github/workflows/ci.yml)

**Triggers:**
- Push to `main`, `develop`, or `claude/**` branches
- Pull requests to `main` or `develop` branches

**Jobs:**

#### lint-and-typecheck
- **Purpose:** Validate code quality and type safety
- **Steps:**
  - Checkout code
  - Setup Node.js 20
  - Install dependencies (with npm cache)
  - Run ESLint for code style and quality
  - Run TypeScript compiler for type checking

**Configuration:**
```bash
npm run lint          # ESLint check
npx tsc --noEmit    # TypeScript check without emitting files
```

#### unit-tests
- **Purpose:** Run unit tests with code coverage
- **Steps:**
  - Checkout code
  - Setup Node.js 20
  - Install dependencies
  - Run tests with coverage reporting
  - Upload coverage to Codecov

**Configuration:**
```bash
npm run test:coverage  # Run tests with coverage
```

#### e2e-tests
- **Purpose:** Run end-to-end tests using Playwright
- **Steps:**
  - Checkout code
  - Setup Node.js 20
  - Install dependencies
  - Install Playwright browsers
  - Run E2E tests
  - Upload test reports as artifacts

**Configuration:**
```bash
npm run test:e2e       # Run Playwright E2E tests
```

#### build
- **Purpose:** Build the application (depends on lint-and-typecheck and unit-tests)
- **Steps:**
  - Checkout code
  - Setup Node.js 20
  - Install dependencies
  - Build application with Vite
  - Upload dist folder as artifact (7-day retention)

**Configuration:**
```bash
npm run build  # Vite build process
```

#### lighthouse
- **Purpose:** Performance and quality auditing (depends on build)
- **Steps:**
  - Checkout code
  - Setup Node.js 20
  - Install dependencies
  - Download build artifacts
  - Run Lighthouse CI

**Configuration:**
- Requires LHCI_GITHUB_APP_TOKEN secret
- Uses .lighthouserc.json configuration

### 2. Production Deployment (.github/workflows/deploy-production.yml)

**Triggers:**
- Push to `main` branch
- Manual trigger (workflow_dispatch)

**Environment:** Production (liveiconic.com)

**Jobs:**

#### deploy
- **Purpose:** Deploy to production on Vercel
- **Steps:**
  - Checkout code
  - Setup Node.js 20
  - Install dependencies
  - Build with environment variables
  - Deploy to Vercel with --prod flag
  - Notify Slack on completion

**Environment Variables Required:**
```
VITE_SUPABASE_URL: Supabase project URL
VITE_SUPABASE_ANON_KEY: Supabase anonymous key
VITE_GA_MEASUREMENT_ID: Google Analytics measurement ID
```

**Secrets Required:**
```
VERCEL_TOKEN: Vercel authentication token
VERCEL_ORG_ID: Vercel organization ID
VERCEL_PROJECT_ID: Vercel project ID
SLACK_WEBHOOK: Slack webhook for notifications
```

### 3. Staging Deployment (.github/workflows/deploy-staging.yml)

**Triggers:**
- Push to `develop` branch

**Environment:** Staging (staging.liveiconic.com)

**Jobs:**

#### deploy
- **Purpose:** Deploy to staging on Netlify
- **Steps:**
  - Checkout code
  - Setup Node.js 20
  - Install dependencies
  - Build with staging environment variables
  - Deploy to Netlify

**Environment Variables Required:**
```
STAGING_SUPABASE_URL: Staging Supabase URL
STAGING_SUPABASE_ANON_KEY: Staging Supabase key
```

**Secrets Required:**
```
NETLIFY_AUTH_TOKEN: Netlify authentication token
NETLIFY_SITE_ID: Netlify site ID
GITHUB_TOKEN: GitHub authentication token (automatic)
```

### 4. Security Scanning (.github/workflows/security.yml)

**Triggers:**
- Push to `main` or `develop`
- Weekly schedule (Sunday 00:00 UTC)

**Jobs:**

#### dependency-check
- **Purpose:** Check for npm vulnerabilities
- **Steps:**
  - Checkout code
  - Run npm audit at moderate level
  - Auto-fix minor vulnerabilities

**Configuration:**
```bash
npm audit --audit-level=moderate  # Check for moderate and high vulnerabilities
```

#### codeql
- **Purpose:** Code security analysis
- **Steps:**
  - Checkout code
  - Initialize CodeQL for JavaScript/TypeScript
  - Perform static code analysis
  - Report security issues

### 5. Performance Monitoring (.github/workflows/performance.yml)

**Triggers:**
- Push to `main` branch
- Every 6 hours schedule

**Jobs:**

#### lighthouse
- **Purpose:** Monitor performance metrics
- **URLs Audited:**
  - https://liveiconic.com
  - https://liveiconic.com/shop
  - https://liveiconic.com/about

## Configuration Files

### .lighthouserc.json

Lighthouse CI configuration with performance thresholds:

```json
{
  "ci": {
    "collect": {
      "startServerCommand": "npm run preview",
      "url": [
        "http://localhost:4173/",
        "http://localhost:4173/shop",
        "http://localhost:4173/about"
      ],
      "numberOfRuns": 3
    },
    "assert": {
      "preset": "lighthouse:recommended",
      "assertions": {
        "categories:performance": ["error", { "minScore": 0.9 }],
        "categories:accessibility": ["error", { "minScore": 0.95 }],
        "categories:best-practices": ["error", { "minScore": 0.95 }],
        "categories:seo": ["error", { "minScore": 1 }],
        "first-contentful-paint": ["error", { "maxNumericValue": 2000 }],
        "largest-contentful-paint": ["error", { "maxNumericValue": 2500 }],
        "cumulative-layout-shift": ["error", { "maxNumericValue": 0.1 }],
        "max-potential-fid": ["error", { "maxNumericValue": 200 }]
      }
    },
    "upload": {
      "target": "temporary-public-storage"
    }
  }
}
```

**Performance Thresholds:**
- Performance Score: 90% minimum
- Accessibility Score: 95% minimum
- Best Practices Score: 95% minimum
- SEO Score: 100% (perfect)
- First Contentful Paint: ≤ 2000ms
- Largest Contentful Paint: ≤ 2500ms
- Cumulative Layout Shift: ≤ 0.1
- Max Potential FID: ≤ 200ms

### .github/dependabot.yml

Automated dependency updates configuration:

```yaml
version: 2
updates:
  # npm dependencies - weekly on Monday
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    open-pull-requests-limit: 10
    reviewers:
      - "maintainers"
    commit-message:
      prefix: "chore"
      include: "scope"

  # GitHub Actions - weekly on Monday
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    open-pull-requests-limit: 5
    commit-message:
      prefix: "ci"
      include: "scope"
```

## Docker Configuration

### Dockerfile

Multi-stage Docker build for production:

```dockerfile
# Build stage
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

**Features:**
- Multi-stage build for smaller final image
- Node.js build stage
- Nginx serving layer
- Optimized for production deployment

### nginx.conf

Nginx configuration for serving SPA:

```nginx
events {
  worker_connections 1024;
}

http {
  include /etc/nginx/mime.types;
  default_type application/octet-stream;

  server {
    listen 80;
    server_name _;
    root /usr/share/nginx/html;
    index index.html;

    # SPA routing - all requests go to index.html
    location / {
      try_files $uri $uri/ /index.html;
    }

    # Long-term cache for static assets
    location /assets {
      expires 1y;
      add_header Cache-Control "public, immutable";
    }
  }
}
```

**Key Features:**
- SPA routing support (try_files)
- Asset caching (1 year for /assets)
- Optimized for production serving

### docker-compose.yml

Local development setup:

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "3000:80"
    environment:
      - NODE_ENV=production
    restart: unless-stopped

  dev:
    image: node:20-alpine
    working_dir: /app
    volumes:
      - .:/app
    ports:
      - "5173:5173"
    command: npm run dev
    environment:
      - NODE_ENV=development
```

**Services:**
- `app`: Production build served on port 3000
- `dev`: Development server on port 5173 with hot reload

### .dockerignore

Files excluded from Docker build:

```
node_modules
dist
.git
.github
*.md
.env*
!.env.example
coverage
playwright-report
test-results
archive
reports
.claude
.gitignore
desktop.ini
```

## Setup Instructions

### 1. Initial GitHub Configuration

#### Create Environments
In GitHub Repository Settings > Environments:

1. **Production**
   - Deployment branch: main
   - Protection rules: Require reviewers

2. **Staging**
   - Deployment branch: develop
   - Protection rules: Require reviewers (optional)

#### Configure Secrets

In GitHub Repository Settings > Secrets and variables > Actions:

**Production Secrets:**
```
VERCEL_TOKEN: <your-vercel-token>
VERCEL_ORG_ID: <your-vercel-org-id>
VERCEL_PROJECT_ID: <your-vercel-project-id>
VITE_SUPABASE_URL: <production-supabase-url>
VITE_SUPABASE_ANON_KEY: <production-anon-key>
VITE_GA_MEASUREMENT_ID: <google-analytics-id>
SLACK_WEBHOOK: <your-slack-webhook-url>
LHCI_GITHUB_APP_TOKEN: <lighthouse-ci-token>
```

**Staging Secrets:**
```
NETLIFY_AUTH_TOKEN: <your-netlify-token>
NETLIFY_SITE_ID: <your-netlify-site-id>
STAGING_SUPABASE_URL: <staging-supabase-url>
STAGING_SUPABASE_ANON_KEY: <staging-anon-key>
```

### 2. Local Development

#### Prerequisites
- Node.js 20+
- Docker and Docker Compose (optional)
- npm 10+

#### Setup
```bash
# Install dependencies
npm ci

# Run development server
npm run dev

# Run tests
npm run test

# Run linting
npm run lint

# Build for production
npm run build
```

#### Docker Development
```bash
# Start development environment
docker-compose up dev

# Access development server
# http://localhost:5173

# Start production build
docker-compose up app

# Access production build
# http://localhost:3000
```

### 3. Configure Vercel (Production)

1. Connect GitHub repository to Vercel
2. Set environment variables:
   ```
   VITE_SUPABASE_URL
   VITE_SUPABASE_ANON_KEY
   VITE_GA_MEASUREMENT_ID
   ```
3. Disable auto-deployments (use GitHub Actions)
4. Set build command: `npm run build`
5. Set output directory: `dist`

### 4. Configure Netlify (Staging)

1. Connect GitHub repository to Netlify
2. Set environment variables:
   ```
   STAGING_SUPABASE_URL
   STAGING_SUPABASE_ANON_KEY
   ```
3. Disable auto-deployments (use GitHub Actions)
4. Set build command: `npm run build`
5. Set publish directory: `dist`

### 5. Configure Lighthouse CI

1. Install Lighthouse CI: `npm install -g @lhci/cli@latest`
2. Run: `lhci wizard`
3. Choose "npm script" as start server command
4. Get GitHub App token from Lighthouse CI dashboard
5. Add to GitHub Secrets as `LHCI_GITHUB_APP_TOKEN`

## GitHub Actions Status and Badges

Add these badges to your README.md:

```markdown
![CI](https://github.com/yourusername/live-it-iconic/actions/workflows/ci.yml/badge.svg)
![Deploy Production](https://github.com/yourusername/live-it-iconic/actions/workflows/deploy-production.yml/badge.svg)
![Deploy Staging](https://github.com/yourusername/live-it-iconic/actions/workflows/deploy-staging.yml/badge.svg)
![Security Scan](https://github.com/yourusername/live-it-iconic/actions/workflows/security.yml/badge.svg)
```

## Workflow Execution Flow

### Development Workflow (Feature Branch)
```
Feature Branch Push
    ↓
    ├─→ CI Job (Lint, Type Check, Tests)
    │   ├─ If fails: Notify developer
    │   └─ If passes: Continue
    └─→ Build Job (depends on CI)
        └─→ Lighthouse (depends on Build)
```

### Merge to Develop
```
Push to develop
    ↓
    ├─→ CI Job
    ├─→ Build Job
    ├─→ Security Scan
    └─→ Auto Deploy to Staging (Netlify)
```

### Merge to Main
```
Push to main
    ↓
    ├─→ CI Job
    ├─→ Build Job
    ├─→ Security Scan
    ├─→ Performance Monitoring
    └─→ Auto Deploy to Production (Vercel)
```

## Monitoring and Maintenance

### Weekly Tasks
- Review Dependabot PRs
- Check security scan results
- Monitor performance trends

### Monthly Tasks
- Review workflow performance
- Update Node.js version if needed
- Update GitHub Actions versions

### Quarterly Tasks
- Review and update performance thresholds
- Audit CI/CD costs
- Update security policies

## Troubleshooting

### Build Failures

1. **Node cache issues:**
   ```bash
   npm cache clean --force
   rm -rf node_modules package-lock.json
   npm ci
   ```

2. **Type errors:**
   ```bash
   npx tsc --noEmit
   npm run lint:fix
   ```

3. **Test failures:**
   ```bash
   npm run test -- --reporter=verbose
   npm run test:coverage
   ```

### Deployment Issues

1. **Check workflow logs:** GitHub > Actions > Workflow > Latest Run
2. **Verify secrets:** Settings > Secrets and variables > Actions
3. **Test locally:** Run build and preview commands locally
4. **Check deployment targets:**
   - Vercel: vercel.com/dashboard
   - Netlify: app.netlify.com/sites

### Performance Monitoring

1. **Lighthouse CI issues:**
   - Check .lighthouserc.json thresholds
   - Review Lighthouse CI reports
   - Test locally: `lhci autorun`

2. **Performance degradation:**
   - Check bundle size
   - Profile with Chrome DevTools
   - Review recent changes

## Best Practices

1. **Always run tests locally before pushing:**
   ```bash
   npm run lint && npm run test && npm run build
   ```

2. **Use descriptive commit messages:**
   ```
   feat(feature): Description
   fix(bug): Description
   chore(deps): Update dependency
   ```

3. **Create detailed PRs** with:
   - Clear description
   - Testing steps
   - Screenshots/videos if UI changes
   - Link to related issues

4. **Monitor workflow costs:**
   - GitHub Actions: 3000 minutes/month (free)
   - Artifact storage: 500MB (free)
   - External services: Check Vercel/Netlify quotas

5. **Keep dependencies updated:**
   - Review Dependabot PRs weekly
   - Test thoroughly before merging
   - Update Node.js annually

## File Structure

```
live-it-iconic/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                 # Main CI pipeline
│   │   ├── deploy-production.yml   # Production deployment
│   │   ├── deploy-staging.yml      # Staging deployment
│   │   ├── security.yml            # Security scanning
│   │   └── performance.yml         # Performance monitoring
│   ├── dependabot.yml             # Automated dependency updates
│   └── CODEOWNERS                 # Code ownership
├── .lighthouserc.json             # Lighthouse CI configuration
├── Dockerfile                      # Production Docker image
├── docker-compose.yml             # Local Docker development
├── nginx.conf                     # Nginx web server config
├── .dockerignore                  # Docker ignore patterns
├── docs/
│   └── CICD.md                    # This file
├── src/                           # Source code
├── package.json                   # Project dependencies
├── tsconfig.json                  # TypeScript configuration
└── vite.config.ts                # Vite configuration
```

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Lighthouse CI Documentation](https://github.com/GoogleChromeLabs/lighthouse-ci)
- [Docker Documentation](https://docs.docker.com/)
- [Vercel Deployment Guide](https://vercel.com/docs)
- [Netlify Deployment Guide](https://docs.netlify.com/)
