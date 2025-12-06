# REPZ Testing Guide

## Overview
Comprehensive testing setup for REPZ platform using modern tools and best practices.

## Quick Start

### Local Development
```bash
# Install dependencies
npm install

# Run all tests
npm run test

# Run specific test suites
npm run test:unit          # Unit tests
npm run test:component     # Component tests  
npm run test:integration   # Integration tests
npm run test:e2e          # E2E tests
npm run test:a11y         # Accessibility tests
npm run test:performance  # Performance tests
npm run test:security     # Security scans
```

### CI/CD
Tests run automatically on PR and main branch pushes via GitHub Actions.

## Test Structure

```
tests/
├── unit/                 # Jest/Vitest unit tests
├── component/           # React Testing Library component tests
├── integration/         # API and database integration tests
├── e2e/                # Playwright end-to-end tests
│   ├── auth/           # Authentication flows
│   ├── tiers/          # Tier system testing
│   ├── protocols/      # Advanced protocol testing
│   ├── nutrition/      # Food database and meal planning
│   ├── medical/        # Medical oversight workflows
│   └── fixtures/       # Test data and utilities
├── performance/        # k6 load testing scripts
├── security/          # OWASP ZAP security tests
└── accessibility/     # axe-core accessibility tests
```

## Test Categories

### Unit Tests (Jest/Vitest)
- Pure functions and utilities
- React hooks
- State management logic
- Business rule validation

### Component Tests (React Testing Library)
- Component rendering
- User interactions
- Props and state changes
- Integration with hooks

### Integration Tests
- API endpoint testing
- Database operations
- Supabase integration
- Stripe payment flows

### E2E Tests (Playwright)
- Complete user journeys
- Cross-browser compatibility
- Real-world scenarios
- Critical business flows

### Performance Tests (k6)
- Load testing
- Stress testing
- API performance
- Database performance

### Security Tests (OWASP ZAP)
- Vulnerability scanning
- Authentication security
- Authorization testing
- Input validation

### Accessibility Tests (axe-core)
- WCAG 2.2 AA compliance
- Keyboard navigation
- Screen reader compatibility
- Color contrast validation

## Test Data Management

### Fixtures
Test data is managed through fixtures for consistent, repeatable tests:

```typescript
// Tier system test data
export const tierFixtures = {
  coreUser: { tier: 'core', features: [...] },
  adaptiveUser: { tier: 'adaptive', features: [...] },
  performanceUser: { tier: 'performance', features: [...] },
  longevityUser: { tier: 'longevity', features: [...] }
};

// Medical professional test data
export const medicalFixtures = {
  verifiedDoctor: { license: 'MD123456', verified: true },
  pendingDoctor: { license: 'MD789012', verified: false }
};
```

### Database Seeding
```bash
# Seed test database
npm run db:seed:test

# Reset test database
npm run db:reset:test
```

## Environment Configuration

### Test Environments
- **Local**: `npm run test:local`
- **CI**: Automated in GitHub Actions
- **Staging**: `npm run test:staging`

### Environment Variables
```bash
# .env.test
VITE_SUPABASE_URL=your_test_supabase_url
VITE_SUPABASE_ANON_KEY=your_test_supabase_key
VITE_STRIPE_PUBLIC_KEY=pk_test_your_stripe_test_key
PLAYWRIGHT_TEST_BASE_URL=http://localhost:8080
```

## Debugging Tests

### Playwright Debug Mode
```bash
# Run with browser UI
npm run test:e2e -- --headed

# Debug specific test
npm run test:e2e -- --debug auth.spec.ts

# Generate trace files
PLAYWRIGHT_TRACE=on npm run test:e2e
```

### Test Artifacts
- Screenshots on failure
- Video recordings
- Network logs
- Console outputs
- Lighthouse reports

## Quality Gates

### PR Requirements
- [ ] All unit tests pass
- [ ] Component tests pass
- [ ] Integration tests pass
- [ ] Critical E2E tests pass
- [ ] No new accessibility violations
- [ ] Security scan passes
- [ ] Performance budget maintained

### Release Requirements
- [ ] Full E2E test suite passes
- [ ] Load testing completed
- [ ] Security penetration test passes
- [ ] Medical safety protocols validated
- [ ] Cross-browser compatibility verified

## Best Practices

### Writing Tests
1. **Arrange-Act-Assert** pattern
2. **Descriptive test names** that explain the scenario
3. **Independent tests** that don't rely on each other
4. **Realistic test data** that matches production scenarios
5. **Proper cleanup** to avoid test pollution

### E2E Test Guidelines
1. **Test user journeys**, not individual features
2. **Use data-testid attributes** for reliable element selection
3. **Wait for elements** properly with Playwright locators
4. **Mock external services** when appropriate
5. **Keep tests maintainable** with page object models

### Performance Testing
1. **Realistic load patterns** based on production traffic
2. **Gradual ramp-up** to identify breaking points
3. **Monitor key metrics** (response time, throughput, errors)
4. **Test different scenarios** (normal, peak, stress)

### Security Testing
1. **Authentication bypass** attempts
2. **Authorization escalation** testing
3. **Input validation** with malicious payloads
4. **Session management** security
5. **Medical data protection** validation

## Troubleshooting

### Common Issues
1. **Flaky tests**: Use proper waits and stable selectors
2. **Slow tests**: Optimize test data and reduce unnecessary operations
3. **Environment issues**: Verify configuration and dependencies
4. **Browser compatibility**: Test across supported browsers

### Getting Help
- Check existing GitHub issues
- Review test logs and artifacts
- Use debugging tools and traces
- Consult team documentation

## Contributing

### Adding New Tests
1. Choose appropriate test type based on scope
2. Follow existing patterns and conventions
3. Add proper documentation and comments
4. Ensure tests are reliable and maintainable
5. Update this guide if needed

### Modifying Existing Tests
1. Understand the test's purpose and scope
2. Maintain backward compatibility when possible
3. Update related documentation
4. Run full test suite to verify changes
5. Consider impact on CI/CD pipeline

---

For specific implementation examples, see the test files in each directory.