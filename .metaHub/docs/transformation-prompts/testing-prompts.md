# TRAE Testing Prompts

## Validate Routing Implementation & Quality Assurance

**Purpose**: Generate comprehensive testing strategies and scripts to validate TRAE routing implementation, ensuring 7-10x cost reduction while maintaining >95% quality.

**Prerequisites**:

- TRAE routing components implemented
- Configuration files deployed
- Basic integration testing completed
- Performance baselines established

---

## TESTING FRAMEWORK OVERVIEW

You are a QA automation specialist. Your task is to create comprehensive testing strategies that validate TRAE routing implementation across functional, performance, compliance, and cost optimization dimensions.

### Testing Pyramid Structure

```
Unit Tests (70%)          → Component-level validation
Integration Tests (20%)   → End-to-end routing validation
System Tests (10%)        → Production scenario validation
```

### Test Categories

1. **Functional Testing**: Routing logic and component integration
2. **Performance Testing**: Latency, throughput, scalability
3. **Cost Optimization Testing**: Budget controls and savings validation
4. **Compliance Testing**: Framework adherence and audit validation
5. **Reliability Testing**: Failover, error handling, recovery

---

## UNIT TESTING PROMPTS

### IntelligentRouter Unit Tests

```
You are a unit testing specialist. Generate comprehensive unit tests for IntelligentRouter that:

TEST COVERAGE REQUIREMENTS:
- 7-step routing pipeline validation
- Cost evaluation and budget checking
- Model selection integration
- Error handling and recovery
- Configuration loading and validation

MOCKING STRATEGY:
- Mock AIEngineIntegration for provider abstraction
- Mock CostController for budget operations
- Mock ModelSelector for selection logic
- Mock FallbackManager for geographic operations

TEST SCENARIOS:
- Successful routing with optimal model selection
- Budget exceeded with automatic downgrade
- Geographic failover on provider failure
- Compliance violation blocking
- Emergency mode activation

Generate test suite with:
1. Test class structure with proper setup/teardown
2. Mock implementations for all dependencies
3. Test cases for all routing scenarios
4. Edge case and error condition testing
5. Performance benchmarking utilities
6. Code coverage reporting integration
```

### Generated Test Example

```typescript
describe('IntelligentRouter', () => {
  let router: IntelligentRouter;
  let mockAiEngine: jest.Mocked<AIEngineIntegration>;
  let mockCostController: jest.Mocked<CostController>;
  let mockModelSelector: jest.Mocked<ModelSelector>;

  beforeEach(() => {
    mockAiEngine = {
      generateText: jest.fn(),
      getAvailableModels: jest.fn(),
      estimateCost: jest.fn(),
      validateCompliance: jest.fn(),
    };

    mockCostController = {
      evaluateCost: jest.fn(),
      recordCost: jest.fn(),
      getCostAnalysis: jest.fn(),
    };

    mockModelSelector = {
      analyzeTask: jest.fn(),
      selectModel: jest.fn(),
    };

    router = new IntelligentRouter(config, mockAiEngine, mockOrchestration);
  });

  describe('routeRequest', () => {
    it('should successfully route simple request', async () => {
      // Test setup
      const prompt = 'Hello world';
      const context = createTestContext();

      // Mock responses
      mockModelSelector.analyzeTask.mockResolvedValue(mockTaskAnalysis);
      mockModelSelector.selectModel.mockResolvedValue(mockRoutingDecision);
      mockCostController.evaluateCost.mockResolvedValue({ approved: true });
      mockAiEngine.generateText.mockResolvedValue('Hello!');

      // Execute test
      const result = await router.routeRequest(prompt, context);

      // Assertions
      expect(result.success).toBe(true);
      expect(result.routingDecision).toBeDefined();
      expect(mockAiEngine.generateText).toHaveBeenCalledWith(
        expect.any(String),
        expect.any(Object)
      );
    });

    it('should handle budget exceeded with downgrade', async () => {
      // Test budget constraint handling
      mockCostController.evaluateCost.mockResolvedValue({
        approved: false,
        adjustedDecision: mockDowngradedDecision,
        reason: 'Budget exceeded',
      });

      const result = await router.routeRequest(prompt, context);

      expect(result.routingDecision.selectedModel.tier).toBe('tier_2');
      expect(result.success).toBe(true);
    });
  });
});
```

### ModelSelector Unit Tests

```
You are a testing specialist for AI model selection. Generate unit tests for ModelSelector that:

COMPLEXITY ANALYSIS TESTING:
- Test 8-indicator complexity detection
- Validate task classification accuracy
- Test capability requirement identification
- Verify domain detection logic

MODEL SELECTION TESTING:
- Test multi-criteria ranking algorithm
- Validate fallback chain generation
- Test geographic preference handling
- Verify compliance filtering

EDGE CASE TESTING:
- Empty or invalid prompts
- Missing model capabilities
- Geographic restrictions
- Compliance violations

Generate test suite with:
1. Complexity analysis test cases
2. Model ranking validation
3. Geographic routing tests
4. Compliance filtering tests
5. Performance benchmarks
6. Error handling validation
```

### CostController Unit Tests

```
You are a financial testing specialist. Generate unit tests for CostController that:

BUDGET MANAGEMENT TESTING:
- Test multi-layer budget enforcement
- Validate cost evaluation logic
- Test automatic optimization triggers
- Verify emergency mode activation

COST OPTIMIZATION TESTING:
- Test model downgrade logic
- Validate geographic cost optimization
- Test predictive cost analysis
- Verify savings calculations

ALERT & MONITORING TESTING:
- Test budget threshold alerts
- Validate cost anomaly detection
- Test notification system integration
- Verify audit trail generation

Generate test suite with:
1. Budget enforcement test cases
2. Cost optimization validation
3. Alert system testing
4. Emergency control testing
5. Performance and scalability tests
6. Integration with external systems
```

---

## INTEGRATION TESTING PROMPTS

### End-to-End Routing Pipeline Tests

```
You are an integration testing specialist. Generate end-to-end tests for the complete routing pipeline that:

FULL PIPELINE VALIDATION:
- Test complete request flow from prompt to response
- Validate all 7 routing pipeline steps
- Test component integration and data flow
- Verify cross-component communication

MULTI-PROVIDER TESTING:
- Test routing across different AI providers
- Validate provider failover scenarios
- Test model compatibility and switching
- Verify cost optimization across providers

GEOGRAPHIC FAILOVER TESTING:
- Test geographic region switching
- Validate latency-based routing decisions
- Test compliance-aware geographic selection
- Verify health monitoring integration

Generate integration tests with:
1. Complete pipeline test scenarios
2. Multi-provider integration tests
3. Geographic failover validation
4. Performance and load testing
5. Error recovery and resilience testing
6. Monitoring and observability validation
```

### Generated Integration Test Example

```typescript
describe('Routing Pipeline Integration', () => {
  let testEnvironment: TestEnvironment;

  beforeAll(async () => {
    testEnvironment = await setupTestEnvironment({
      providers: ['openai', 'anthropic'],
      regions: ['north_america', 'europe'],
      budget: { dailyLimit: 100 },
    });
  });

  describe('Complete Request Flow', () => {
    it('should route request through full pipeline successfully', async () => {
      const request = {
        prompt: 'Analyze this complex algorithm',
        context: {
          userId: 'test-user',
          priority: 'high',
          complianceContext: {
            dataClassification: 'confidential',
          },
        },
      };

      const result = await testEnvironment.router.routeRequest(request.prompt, request.context);

      // Validate complete pipeline execution
      expect(result.success).toBe(true);
      expect(result.routingDecision).toBeDefined();
      expect(result.actualCost).toBeGreaterThan(0);
      expect(result.metadata.taskAnalysis.complexity).toBeDefined();

      // Verify cost optimization
      expect(result.routingDecision.costSavings).toBeGreaterThan(0);

      // Check compliance validation
      expect(result.metadata.complianceAudit.validated).toBe(true);
    });

    it('should handle provider failover gracefully', async () => {
      // Simulate OpenAI failure
      testEnvironment.mockProviderFailure('openai');

      const result = await testEnvironment.router.routeRequest('Simple question', {
        priority: 'normal',
      });

      // Should failover to Anthropic
      expect(result.success).toBe(true);
      expect(result.routingDecision.selectedModel.provider).toBe('anthropic');
      expect(result.attempts).toBeGreaterThan(1);
    });
  });
});
```

### Compliance Integration Tests

```
You are a compliance testing specialist. Generate integration tests for compliance framework validation that:

GDPR COMPLIANCE TESTING:
- Test data minimization implementation
- Validate consent management integration
- Test right to erasure functionality
- Verify audit logging completeness

HIPAA COMPLIANCE TESTING:
- Test PHI data handling
- Validate encryption requirements
- Test breach notification workflows
- Verify access control enforcement

SOC2 COMPLIANCE TESTING:
- Test security control effectiveness
- Validate audit trail integrity
- Test availability monitoring
- Verify incident response procedures

Generate compliance tests with:
1. Framework-specific test suites
2. Data handling validation
3. Audit and logging verification
4. Security control testing
5. Incident response validation
6. Compliance reporting accuracy
```

---

## PERFORMANCE TESTING PROMPTS

### Load and Scalability Testing

```
You are a performance testing specialist. Generate load testing scripts for TRAE routing that:

LOAD TESTING SCENARIOS:
- Gradual load increase to identify breaking points
- Sustained load testing for stability validation
- Spike testing for emergency control activation
- Concurrent user simulation

PERFORMANCE METRICS:
- Response latency under various loads
- Throughput measurement (requests/second)
- Error rates under stress
- Resource utilization monitoring

SCALABILITY VALIDATION:
- Horizontal scaling capability testing
- Database connection pool validation
- Cache effectiveness under load
- Network bandwidth utilization

Generate performance tests with:
1. Load testing scripts and scenarios
2. Performance monitoring integration
3. Scalability validation procedures
4. Resource utilization analysis
5. Performance regression detection
6. Capacity planning recommendations
```

### Generated Load Test Script

```typescript
class LoadTester {
  private router: IntelligentRouter;
  private metrics: PerformanceMetrics;

  async runLoadTest(scenario: LoadScenario): Promise<LoadTestResult> {
    const results: RequestResult[] = [];

    // Warm-up phase
    await this.warmUp(scenario.warmUpRequests);

    // Load testing phase
    const startTime = Date.now();
    const promises = Array.from({ length: scenario.concurrentUsers }, (_, i) =>
      this.simulateUser(scenario, i)
    );

    const userResults = await Promise.all(promises);
    const endTime = Date.now();

    // Aggregate results
    return this.analyzeResults(userResults, endTime - startTime);
  }

  private async simulateUser(scenario: LoadScenario, userId: number): Promise<RequestResult[]> {
    const results: RequestResult[] = [];

    for (let i = 0; i < scenario.requestsPerUser; i++) {
      const startTime = Date.now();

      try {
        const result = await this.router.routeRequest(
          this.generateTestPrompt(scenario.promptType),
          this.generateTestContext(userId)
        );

        results.push({
          success: result.success,
          latency: Date.now() - startTime,
          cost: result.actualCost,
          model: result.routingDecision.selectedModel.name,
        });
      } catch (error) {
        results.push({
          success: false,
          latency: Date.now() - startTime,
          error: error.message,
        });
      }

      // Rate limiting simulation
      await delay(scenario.delayBetweenRequests);
    }

    return results;
  }
}
```

### Cost Optimization Testing

```
You are a cost optimization testing specialist. Generate tests to validate 7-10x cost reduction that:

COST REDUCTION VALIDATION:
- Compare costs before/after TRAE implementation
- Validate optimization algorithm effectiveness
- Test budget control enforcement
- Verify savings calculation accuracy

OPTIMIZATION SCENARIO TESTING:
- Test automatic model downgrades
- Validate geographic cost optimization
- Test time-based routing savings
- Verify emergency cost controls

BUDGET COMPLIANCE TESTING:
- Test budget threshold enforcement
- Validate alert system accuracy
- Test emergency mode activation
- Verify cost prediction reliability

Generate cost tests with:
1. Cost reduction validation scripts
2. Budget compliance test cases
3. Optimization scenario testing
4. Predictive cost analysis validation
5. Cost anomaly detection testing
6. ROI calculation and reporting
```

---

## COMPLIANCE TESTING PROMPTS

### GDPR Compliance Test Suite

```
You are a GDPR compliance tester. Generate comprehensive GDPR validation tests that:

DATA PROTECTION PRINCIPLES:
- Lawfulness, fairness, and transparency testing
- Purpose limitation validation
- Data minimization verification
- Accuracy testing procedures

INDIVIDUAL RIGHTS TESTING:
- Right to access implementation testing
- Right to rectification validation
- Right to erasure (deletion) testing
- Right to data portability verification

TECHNICAL MEASURES:
- Encryption implementation validation
- Anonymization technique testing
- Pseudonymization verification
- Data breach notification testing

Generate GDPR tests with:
1. Data protection principle validation
2. Individual rights implementation testing
3. Technical safeguard verification
4. Audit and accountability testing
5. Data breach simulation and response
6. Compliance reporting validation
```

### HIPAA Compliance Test Suite

```
You are a HIPAA compliance tester. Generate HIPAA validation tests for healthcare data handling that:

PRIVACY RULE TESTING:
- Protected health information handling
- Minimum necessary standard validation
- Individual rights implementation
- Notice of privacy practices verification

SECURITY RULE TESTING:
- Administrative safeguard validation
- Physical safeguard implementation
- Technical safeguard verification
- Security incident procedures

BREACH NOTIFICATION TESTING:
- Breach detection and analysis
- Notification timeline validation
- Content requirement verification
- Log and documentation testing

Generate HIPAA tests with:
1. Privacy rule compliance validation
2. Security rule implementation testing
3. Breach notification procedure testing
4. Risk analysis and management validation
5. Audit control and monitoring verification
6. Business associate agreement testing
```

---

## RELIABILITY TESTING PROMPTS

### Failover and Recovery Testing

```
You are a reliability testing specialist. Generate failover and recovery tests that:

GEOGRAPHIC FAILOVER TESTING:
- Regional outage simulation
- Latency threshold triggering
- Automatic failover validation
- Recovery procedure testing

PROVIDER FAILOVER TESTING:
- Provider API outage simulation
- Rate limit handling validation
- Service degradation testing
- Multi-provider redundancy verification

NETWORK RESILIENCE TESTING:
- Network partition simulation
- Connection timeout handling
- Retry logic validation
- Circuit breaker testing

Generate reliability tests with:
1. Geographic failover scenario testing
2. Provider redundancy validation
3. Network resilience testing
4. Recovery procedure verification
5. Chaos engineering scenarios
6. Disaster recovery testing
```

### Chaos Engineering Tests

```
You are a chaos engineering specialist. Generate chaos tests to validate system resilience that:

CONTROLLED FAILURE INJECTION:
- Random component failure simulation
- Network latency injection
- Resource exhaustion testing
- Configuration corruption simulation

SYSTEM RECOVERY TESTING:
- Automatic recovery validation
- Manual intervention procedures
- Graceful degradation verification
- Service restoration testing

MONITORING VALIDATION:
- Alert system effectiveness testing
- Monitoring blind spot identification
- Dashboard accuracy validation
- Incident response procedure testing

Generate chaos tests with:
1. Failure injection scenarios
2. Recovery validation procedures
3. Monitoring and alerting testing
4. Incident response validation
5. System resilience measurement
6. Continuous improvement recommendations
```

---

## TESTING AUTOMATION FRAMEWORK

### CI/CD Integration

```yaml
# .github/workflows/test.yml
name: TRAE Testing Suite

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install dependencies
        run: npm ci
      - name: Run unit tests
        run: npm run test:unit
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  integration-tests:
    runs-on: ubuntu-latest
    services:
      redis: redis:6-alpine
      postgres: postgres:13-alpine
    steps:
      - uses: actions/checkout@v3
      - name: Run integration tests
        run: npm run test:integration
        env:
          DATABASE_URL: postgres://test:test@localhost:5432/test

  performance-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run performance tests
        run: npm run test:performance
      - name: Upload performance results
        uses: actions/upload-artifact@v3
        with:
          name: performance-results
          path: performance-results/
```

### Test Data Management

```typescript
class TestDataGenerator {
  // Generate realistic test prompts
  generateTestPrompts(count: number, complexity: TaskComplexity): string[] {
    // Implementation for generating test prompts
  }

  // Generate compliance test scenarios
  generateComplianceScenarios(framework: ComplianceFramework): ComplianceTestCase[] {
    // Implementation for compliance test data
  }

  // Generate performance test loads
  generateLoadScenarios(): LoadScenario[] {
    // Implementation for load testing data
  }
}
```

---

## QUALITY ASSURANCE METRICS

### Test Coverage Requirements

```
Unit Tests: >80% code coverage
Integration Tests: >90% API coverage
Performance Tests: All critical paths
Compliance Tests: 100% framework requirements
Reliability Tests: All failure scenarios
```

### Success Criteria

```
Functional: 100% existing functionality preserved
Performance: <10% degradation vs baseline
Cost: 7-10x reduction achieved
Quality: >95% success rate maintained
Compliance: Zero violations in production
Reliability: >99.9% uptime
```

---

**These testing prompts create comprehensive validation strategies for TRAE routing implementation, ensuring quality maintenance while achieving cost optimization goals. The testing framework covers all aspects from unit testing to production validation.**</content>
