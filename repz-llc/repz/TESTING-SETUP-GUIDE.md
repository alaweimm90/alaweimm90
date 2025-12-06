# 🚀 REPZ Testing Framework Setup Guide

## Quick Start

### 1. Install Dependencies
```bash
npm install
```

This will install all testing dependencies including:
- **@axe-core/playwright** - Accessibility testing
- **@playwright/test** - End-to-end testing
- **vitest** - Unit and integration testing
- **wait-on** - Server readiness checking

### 2. Install Playwright Browsers
```bash
npx playwright install
```

### 3. Install k6 (Performance Testing)
```bash
# macOS
brew install k6

# Ubuntu/Debian
sudo gpg -k
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update
sudo apt-get install k6

# Windows
choco install k6
```

### 4. Configure Environment Variables
Copy and configure test environment:
```bash
cp .env.test .env.local
```

Update the following variables in `.env.local`:
```bash
VITE_SUPABASE_URL=your_test_supabase_url
VITE_SUPABASE_ANON_KEY=your_test_supabase_anon_key
VITE_STRIPE_PUBLIC_KEY=pk_test_your_stripe_test_key
```

### 5. Set Up Test Database (if using Supabase locally)
```bash
# Start Supabase locally
npx supabase start

# Run migrations
npm run db:migrate:test

# Seed test data
npm run db:seed:test
```

## Running Tests

### All Tests
```bash
npm run test:all
```

### Individual Test Suites
```bash
# Unit tests
npm run test:unit

# Component tests
npm run test:component

# Integration tests
npm run test:integration

# End-to-end tests
npm run test:e2e

# Accessibility tests
npm run test:a11y

# Performance tests
npm run test:performance
```

### Development & Debugging
```bash
# Run E2E tests with browser UI
npm run test:e2e:headed

# Debug E2E tests
npm run test:e2e:debug

# Watch mode for unit tests
npm test

# Test coverage
npm run test:coverage
```

## GitHub Actions Setup

### Required Secrets
Add these secrets to your GitHub repository:

**Supabase:**
- `SUPABASE_TEST_URL`
- `SUPABASE_TEST_ANON_KEY`
- `SUPABASE_STAGING_URL`
- `SUPABASE_STAGING_ANON_KEY`

**Stripe:**
- `STRIPE_TEST_PUBLIC_KEY`
- `STRIPE_TEST_SECRET_KEY`

**Deployment:**
- `VERCEL_TOKEN`
- `VERCEL_ORG_ID`
- `VERCEL_PROJECT_ID`

**Monitoring:**
- `CODECOV_TOKEN`
- `SLACK_WEBHOOK_URL`

### Workflow Triggers
The test suite runs on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Daily schedule at 2 AM UTC
- Manual workflow dispatch

## Test Structure

```
tests/
├── unit/                 # Jest/Vitest unit tests
├── integration/          # API and database integration tests
├── e2e/                 # Playwright end-to-end tests
│   ├── 01-client-onboarding.spec.ts
│   ├── 02-advanced-protocols.spec.ts
│   ├── 03-nutrition-management.spec.ts
│   ├── 04-medical-oversight.spec.ts
│   └── 05-ai-assistant.spec.ts
├── accessibility/       # axe-core accessibility tests
├── performance/        # k6 load testing scripts
└── fixtures/           # Test data and utilities
    ├── auth/           # Authentication states
    ├── sample-files/   # Test documents
    ├── global-setup.ts
    ├── global-teardown.ts
    └── test-users.ts
```

## Test Data & Users

### Test User Accounts
The framework creates authenticated sessions for:

- **Core Tier Client**: `test-core@repz.com`
- **Adaptive Tier Client**: `test-adaptive@repz.com`
- **Performance Tier Client**: `test-performance@repz.com`
- **Longevity Tier Client**: `test-longevity@repz.com`
- **Coach**: `test-coach@repz.com`
- **Admin**: `test-admin@repz.com`
- **Medical Professional**: `test-medical@repz.com`

All test accounts use password: `TestPassword123!`

### Test Data
- **Medical professionals** with verified credentials
- **Food database** with 1000+ sample items
- **Protocols** for each tier level
- **Sample recipes** with calculated nutrition
- **Mock conversations** for AI testing

## Quality Standards

### Performance Thresholds
- **LCP**: ≤ 2.5s (75th percentile)
- **CLS**: ≤ 0.1
- **INP**: ≤ 200ms
- **Lighthouse Performance**: ≥ 90 for critical paths

### Accessibility Standards
- **WCAG 2.2 AA** compliance across all pages
- **Keyboard navigation** functional
- **Screen reader** compatible
- **Color contrast** ratios met

### Security Requirements
- Zero **Critical/High** vulnerabilities
- **Authentication** & session management secure
- **Payment data** protected (PCI compliance)
- **AI endpoints** protected from prompt injection

### Browser Support
- **Chrome** 100+
- **Firefox** 100+
- **Safari** 15+
- **Mobile** iOS Safari 15+, Android Chrome 100+

## Performance Testing

### Test Types
```bash
# Quick validation (30 seconds, 5 users)
npm run test:performance:smoke

# Realistic load (24 minutes, up to 100 users)
npm run test:performance:load

# Stress testing (45 minutes, up to 500 users)
npm run test:performance:stress
```

### Performance Metrics
- **Response Time**: P50, P95, P99 percentiles
- **Error Rate**: HTTP failures and custom errors
- **Throughput**: Requests per second
- **Breaking Point**: Maximum sustainable load

## Troubleshooting

### Common Issues

#### Playwright Browser Installation
```bash
# If browsers fail to install
npx playwright install --with-deps
```

#### Port Conflicts
```bash
# If port 8080 is in use
lsof -ti:8080 | xargs kill -9
```

#### Database Connection Issues
```bash
# Reset local Supabase
npx supabase db reset
npx supabase start
```

#### Test Flakiness
- Increase timeouts in `playwright.config.ts`
- Use `page.waitForLoadState('networkidle')`
- Add explicit waits for dynamic content

#### Performance Test Failures
- Check server resources during load testing
- Verify k6 installation and version
- Review test environment configuration

### Debug Commands
```bash
# View test artifacts
ls -la test-results/

# Check Playwright trace
npx playwright show-trace test-results/trace.zip

# View coverage report
open coverage/index.html

# Debug test server
npm run dev -- --debug
```

## Integration with Development Workflow

### Pre-commit Hooks
Add to `.husky/pre-commit`:
```bash
npm run lint
npm run type-check
npm run test:unit
```

### Pre-push Hooks
Add to `.husky/pre-push`:
```bash
npm run test:e2e
npm run test:a11y
```

### Pull Request Checklist
- [ ] All unit tests pass
- [ ] Component tests pass
- [ ] Critical E2E tests pass
- [ ] No new accessibility violations
- [ ] Security scan passes
- [ ] Performance budget maintained

## Monitoring & Reporting

### Test Results
- **HTML Reports**: `playwright-report/index.html`
- **Coverage Reports**: `coverage/index.html`
- **Performance Results**: `test-results/performance/`
- **Accessibility Reports**: `test-results/accessibility/`

### CI/CD Artifacts
- Test execution videos
- Screenshots on failure
- Performance metrics
- Security scan results
- Bundle analysis reports

### Notifications
- **Slack alerts** on test failures
- **GitHub status checks** for PRs
- **Email reports** for scheduled runs

## Support & Resources

### Documentation
- [Playwright Documentation](https://playwright.dev/)
- [Vitest Documentation](https://vitest.dev/)
- [axe-core Documentation](https://github.com/dequelabs/axe-core)
- [k6 Documentation](https://k6.io/docs/)

### Getting Help
1. Check the test logs and artifacts
2. Review existing GitHub issues
3. Use debugging tools and traces
4. Consult team documentation
5. Reach out to QA team

---

**Setup Complete!** 🎉

Your REPZ testing framework is now ready for enterprise-grade quality assurance. The framework provides comprehensive coverage of functionality, accessibility, performance, and security testing.