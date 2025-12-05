# Component Specifications

Detailed specifications for all major ORCHEX components, including their responsibilities, interfaces, algorithms, and implementation details.

---

## 1. Agent Registry

### Purpose

Maintains a comprehensive catalog of available AI agents with their capabilities, constraints, health status, and performance metrics.

### Responsibilities

- Register and deregister agents dynamically
- Track agent capabilities and limitations
- Monitor agent health and availability
- Store and analyze performance history
- Provide intelligent agent discovery and selection

### Interface

```typescript
interface AgentRegistry {
  // Agent Management
  registerAgent(agent: AgentMetadata): Promise<string>;
  deregisterAgent(agentId: string): Promise<void>;
  updateAgent(agentId: string, updates: Partial<AgentMetadata>): Promise<void>;

  // Discovery and Selection
  getAgent(agentId: string): Promise<Agent | null>;
  queryAgents(criteria: AgentQuery): Promise<Agent[]>;
  findBestAgent(task: Task): Promise<Agent | null>;

  // Health and Performance
  updateHealth(agentId: string, health: HealthStatus): Promise<void>;
  getPerformanceHistory(agentId: string, window: TimeWindow): Promise<PerformanceMetrics[]>;
  getHealthStatus(agentId: string): Promise<HealthStatus>;
}
```

### Data Structures

```typescript
interface AgentMetadata {
  agentId: string;
  name: string;
  provider: 'anthropic' | 'openai' | 'google' | 'local';
  model: string;
  version: string;
  capabilities: Capability[];
  constraints: AgentConstraints;
  health: HealthStatus;
  performance: PerformanceMetrics;
  metadata: Record<string, any>;
}

interface AgentConstraints {
  maxTokens: number;
  maxConcurrentTasks: number;
  rateLimitPerMinute: number;
  costPer1kTokens: number;
  supportedLanguages: string[];
  maxFileSize: number;
  timeoutSeconds: number;
}

interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy' | 'offline';
  lastCheck: Date;
  uptimePercentage: number;
  avgResponseTimeMs: number;
  errorRate: number;
  consecutiveFailures: number;
}
```

### Storage Implementation

```typescript
interface AgentStorage {
  // Agent Registry
  agents: Map<string, AgentMetadata>;

  // Health History (time-series)
  healthHistory: TimeSeriesStore<HealthStatus>;

  // Performance History (time-series)
  performanceHistory: TimeSeriesStore<PerformanceMetrics>;

  // Capabilities Index
  capabilitiesIndex: InvertedIndex<Capability, string>;

  // Health Index (for fast queries)
  healthIndex: SpatialIndex<HealthStatus>;
}
```

---

## 2. Task Router

### Purpose

Intelligently routes tasks to the most appropriate AI agent based on task requirements, agent capabilities, performance history, and current system load.

### Responsibilities

- Analyze task requirements and constraints
- Score and rank available agents
- Select optimal agent for task execution
- Manage task lifecycle and state transitions
- Handle routing failures and fallbacks

### Routing Algorithm

```typescript
class TaskRouter {
  async routeTask(task: Task): Promise<RoutingDecision> {
    // 1. Query eligible agents
    const eligibleAgents = await this.registry.queryAgents({
      requiredCapabilities: task.requirements.capabilities,
      maxCost: task.constraints.maxCost,
      supportedLanguages: [task.context.language],
    });

    if (eligibleAgents.length === 0) {
      throw new NoEligibleAgentsError(task);
    }

    // 2. Score each agent
    const scoredAgents = await Promise.all(
      eligibleAgents.map((agent) => this.scoreAgent(agent, task))
    );

    // 3. Sort by score (highest first)
    scoredAgents.sort((a, b) => b.score - a.score);

    // 4. Select best agent
    const bestAgent = scoredAgents[0];

    // 5. Check capacity
    const hasCapacity = await this.loadBalancer.checkCapacity(bestAgent.agent);

    if (!hasCapacity) {
      // Try next best agent
      const nextBest = scoredAgents.find((sa) => sa !== bestAgent);
      if (nextBest) {
        return this.createRoutingDecision(nextBest.agent, nextBest.score, task);
      }
      throw new NoCapacityError(task);
    }

    return this.createRoutingDecision(bestAgent.agent, bestAgent.score, task);
  }

  private async scoreAgent(agent: Agent, task: Task): Promise<ScoredAgent> {
    const capabilityScore = this.calculateCapabilityScore(agent, task);
    const performanceScore = await this.calculatePerformanceScore(agent);
    const availabilityScore = await this.calculateAvailabilityScore(agent);
    const costScore = this.calculateCostScore(agent, task);

    // Weighted scoring (0-100 scale)
    const totalScore =
      0.4 * capabilityScore + 0.3 * performanceScore + 0.2 * availabilityScore + 0.1 * costScore;

    return {
      agent,
      score: totalScore,
      components: {
        capability: capabilityScore,
        performance: performanceScore,
        availability: availabilityScore,
        cost: costScore,
      },
    };
  }
}
```

### Scoring Components

#### Capability Score (40% weight)

```typescript
calculateCapabilityScore(agent: Agent, task: Task): number {
  const required = new Set(task.requirements.capabilities)
  const available = new Set(agent.capabilities)

  const matched = [...required].filter(cap => available.has(cap))
  const matchRatio = matched.length / required.size

  // Bonus for exact matches and additional capabilities
  const bonus = (available.size - matched.length) * 0.1

  return Math.min((matchRatio * 100) + bonus, 100)
}
```

#### Performance Score (30% weight)

```typescript
async calculatePerformanceScore(agent: Agent): Promise<number> {
  const history = await this.registry.getPerformanceHistory(
    agent.agentId,
    { hours: 24 }
  )

  if (history.length === 0) return 50 // Neutral score for new agents

  const avgQuality = history.reduce((sum, h) => sum + h.qualityScore, 0) / history.length
  const successRate = history.filter(h => h.success).length / history.length

  return (avgQuality * 0.6) + (successRate * 100 * 0.4)
}
```

#### Availability Score (20% weight)

```typescript
async calculateAvailabilityScore(agent: Agent): Promise<number> {
  const health = await this.registry.getHealthStatus(agent.agentId)
  const loadRatio = agent.currentLoad / agent.maxCapacity

  // Health factor (0-1)
  const healthFactor = {
    healthy: 1.0,
    degraded: 0.7,
    unhealthy: 0.3,
    offline: 0.0
  }[health.status]

  // Load factor (higher load = lower score)
  const loadFactor = Math.max(0, 1 - loadRatio)

  // Response time factor (faster = higher score)
  const responseFactor = Math.max(0, 1 - (health.avgResponseTimeMs / 10000))

  return (healthFactor * 0.5 + loadFactor * 0.3 + responseFactor * 0.2) * 100
}
```

#### Cost Score (10% weight)

```typescript
calculateCostScore(agent: Agent, task: Task): number {
  if (!task.constraints.maxCost) return 100 // No cost constraint

  const estimatedCost = agent.costPer1kTokens * (task.estimatedTokens / 1000)

  if (estimatedCost > task.constraints.maxCost) {
    return 0 // Over budget
  }

  // Score based on budget utilization (lower utilization = higher score)
  const utilization = estimatedCost / task.constraints.maxCost
  return (1 - utilization) * 100
}
```

---

## 3. Load Balancer

### Purpose

Distributes tasks across multiple agents to optimize throughput, prevent overload, and ensure fair resource utilization.

### Responsibilities

- Monitor agent load and capacity
- Distribute tasks using intelligent algorithms
- Prevent agent overload and cascading failures
- Manage task queues during high load
- Provide load balancing metrics and insights

### Distribution Strategies

#### Weighted Round-Robin

```typescript
class WeightedRoundRobinBalancer {
  private currentIndex = 0;

  distribute(task: Task, candidates: Agent[]): Agent {
    const weightedAgents = candidates.map((agent) => ({
      agent,
      weight: this.calculateWeight(agent),
    }));

    // Sort by weight (highest first)
    weightedAgents.sort((a, b) => b.weight - a.weight);

    // Select agent using weighted round-robin
    const selected = weightedAgents[this.currentIndex % weightedAgents.length];
    this.currentIndex++;

    return selected.agent;
  }

  private calculateWeight(agent: Agent): number {
    const baseWeight = agent.baseWeight || 1;
    const loadRatio = agent.currentLoad / agent.maxCapacity;
    const healthFactor = this.getHealthFactor(agent);

    // Reduce weight as load increases
    return baseWeight * (1 - loadRatio) * healthFactor;
  }
}
```

#### Least Loaded

```typescript
class LeastLoadedBalancer {
  distribute(task: Task, candidates: Agent[]): Agent {
    let bestAgent = candidates[0];
    let lowestLoad = this.calculateLoad(bestAgent);

    for (const agent of candidates.slice(1)) {
      const load = this.calculateLoad(agent);
      if (load < lowestLoad) {
        lowestLoad = load;
        bestAgent = agent;
      }
    }

    return bestAgent;
  }

  private calculateLoad(agent: Agent): number {
    return agent.currentLoad / agent.maxCapacity;
  }
}
```

#### Performance-Based

```typescript
class PerformanceBasedBalancer {
  async distribute(task: Task, candidates: Agent[]): Promise<Agent> {
    const scoredAgents = await Promise.all(
      candidates.map(async (agent) => ({
        agent,
        score: await this.calculatePerformanceScore(agent, task),
      }))
    );

    scoredAgents.sort((a, b) => b.score - a.score);
    return scoredAgents[0].agent;
  }

  private async calculatePerformanceScore(agent: Agent, task: Task): Promise<number> {
    const recentPerformance = await this.registry.getPerformanceHistory(agent.agentId, {
      hours: 1,
    });

    const avgResponseTime =
      recentPerformance.reduce((sum, p) => sum + p.responseTimeMs, 0) / recentPerformance.length;

    const successRate =
      recentPerformance.filter((p) => p.success).length / recentPerformance.length;

    // Prefer faster, more reliable agents
    return successRate * 100 - avgResponseTime / 100;
  }
}
```

### Queue Management

```typescript
class TaskQueue {
  private queues = new Map<string, Task[]>();

  enqueue(task: Task, agentId: string): void {
    if (!this.queues.has(agentId)) {
      this.queues.set(agentId, []);
    }
    this.queues.get(agentId)!.push(task);
  }

  dequeue(agentId: string): Task | null {
    const queue = this.queues.get(agentId);
    return queue && queue.length > 0 ? queue.shift()! : null;
  }

  getQueueLength(agentId: string): number {
    return this.queues.get(agentId)?.length || 0;
  }

  peek(agentId: string): Task | null {
    const queue = this.queues.get(agentId);
    return queue && queue.length > 0 ? queue[0] : null;
  }
}
```

---

## 4. Fallback Manager

### Purpose

Ensures task completion through intelligent fallback chains when primary agents fail, with automatic retry logic and escalation.

### Responsibilities

- Execute tasks with multi-tier fallback chains
- Implement exponential backoff retry logic
- Track failure patterns and adapt strategies
- Escalate to human intervention when needed
- Provide comprehensive failure analysis

### Fallback Chain Execution

```typescript
class FallbackManager {
  async executeWithFallback(task: Task): Promise<TaskResult> {
    const fallbackChain = await this.router.getFallbackChain(task);

    for (const [tier, agent] of fallbackChain.entries()) {
      const maxRetries = this.getMaxRetriesForTier(tier);

      for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
          const result = await this.executeWithTimeout(agent, task);

          // Record success
          await this.telemetry.recordSuccess(agent, task, tier, attempt);

          return result;
        } catch (error) {
          const backoffDelay = this.calculateBackoffDelay(attempt);

          // Record failure
          await this.telemetry.recordFailure(agent, task, tier, attempt, error);

          // Wait before retry (except on last attempt)
          if (!(tier === fallbackChain.length - 1 && attempt === maxRetries - 1)) {
            await this.delay(backoffDelay);
          }
        }
      }
    }

    // All fallbacks exhausted
    return this.escalateToHuman(task);
  }

  private getMaxRetriesForTier(tier: number): number {
    // Decreasing retry count per tier
    return Math.max(1, 4 - tier);
  }

  private calculateBackoffDelay(attempt: number): number {
    // Exponential backoff: 1s, 2s, 4s, 8s...
    return Math.min(1000 * Math.pow(2, attempt), 30000); // Max 30 seconds
  }

  private async executeWithTimeout(agent: Agent, task: Task): Promise<TaskResult> {
    return Promise.race([
      agent.execute(task),
      new Promise((_, reject) =>
        setTimeout(() => reject(new TimeoutError()), task.timeoutSeconds * 1000)
      ),
    ]);
  }
}
```

### Failure Pattern Analysis

```typescript
class FailureAnalyzer {
  analyzeFailure(agent: Agent, task: Task, error: Error): FailurePattern {
    const pattern = {
      agentId: agent.agentId,
      taskType: task.type,
      errorType: this.categorizeError(error),
      timestamp: new Date(),
      context: {
        load: agent.currentLoad / agent.maxCapacity,
        recentFailures: this.getRecentFailures(agent.agentId),
        taskComplexity: this.assessTaskComplexity(task),
      },
    };

    // Update failure patterns
    this.updatePatterns(pattern);

    // Trigger adaptations if needed
    this.checkForAdaptations(pattern);

    return pattern;
  }

  private categorizeError(error: Error): ErrorCategory {
    if (error.message.includes('rate limit')) return 'rate_limit';
    if (error.message.includes('timeout')) return 'timeout';
    if (error.message.includes('network')) return 'network';
    if (error.message.includes('authentication')) return 'auth';
    return 'unknown';
  }

  private checkForAdaptations(pattern: FailurePattern): void {
    const recentPatterns = this.getRecentPatterns(pattern.agentId);

    // Check for rate limiting
    if (this.detectRateLimiting(recentPatterns)) {
      this.adaptForRateLimiting(pattern.agentId);
    }

    // Check for capacity issues
    if (this.detectCapacityIssues(recentPatterns)) {
      this.adaptForCapacity(pattern.agentId);
    }

    // Check for model-specific issues
    if (this.detectModelIssues(recentPatterns)) {
      this.adaptForModel(pattern.agentId);
    }
  }
}
```

### Human Escalation

```typescript
class HumanEscalationHandler {
  async escalateToHuman(task: Task, failureHistory: FailureRecord[]): Promise<TaskResult> {
    // Create escalation record
    const escalation = await this.createEscalationRecord(task, failureHistory);

    // Notify team
    await this.notifyTeam(escalation);

    // Create tracking issue/ticket
    const ticket = await this.createTrackingTicket(escalation);

    // Set up monitoring
    await this.setupMonitoring(escalation);

    return {
      success: false,
      escalation: {
        ticketId: ticket.id,
        ticketUrl: ticket.url,
        priority: this.calculatePriority(task, failureHistory),
        estimatedResolutionTime: this.estimateResolutionTime(task),
      },
    };
  }

  private async createEscalationRecord(task: Task, failures: FailureRecord[]) {
    return {
      id: generateId(),
      taskId: task.taskId,
      taskType: task.type,
      failureHistory: failures,
      createdAt: new Date(),
      priority: this.calculateEscalationPriority(task, failures),
      context: {
        repository: task.context.repository,
        files: task.context.files,
        requirements: task.requirements,
      },
    };
  }
}
```

---

## 5. Repository Analyzer

### Purpose

Analyzes codebases to identify technical debt, code quality issues, and refactoring opportunities using AST-based parsing and chaos metrics.

### Responsibilities

- Parse source code using Abstract Syntax Trees (AST)
- Calculate comprehensive chaos metrics
- Identify code smells and anti-patterns
- Generate prioritized refactoring opportunities
- Provide actionable improvement recommendations

### Analysis Pipeline

```typescript
class RepositoryAnalyzer {
  async analyze(repository: Repository): Promise<AnalysisReport> {
    // 1. Discover source files
    const sourceFiles = await this.discoverSourceFiles(repository);

    // 2. Parse files to AST
    const asts = await this.parseFilesToAST(sourceFiles);

    // 3. Calculate chaos metrics
    const chaosMetrics = await this.calculateChaosMetrics(asts);

    // 4. Identify opportunities
    const opportunities = await this.identifyOpportunities(asts, chaosMetrics);

    // 5. Prioritize opportunities
    const prioritized = this.prioritizeOpportunities(opportunities);

    // 6. Generate summary
    const summary = this.generateSummary(chaosMetrics, prioritized);

    return {
      repository: repository.path,
      timestamp: new Date(),
      filesAnalyzed: sourceFiles.length,
      chaosMetrics,
      opportunities: prioritized,
      summary,
    };
  }
}
```

### Chaos Metrics Calculation

```typescript
class ChaosMetricsCalculator {
  calculateMetrics(ast: AST, filePath: string): ChaosMetrics {
    const complexity = this.calculateComplexity(ast);
    const duplication = this.detectDuplication(ast);
    const coupling = this.measureCoupling(ast);
    const size = this.measureSize(ast);
    const documentation = this.calculateDocumentationCoverage(ast);

    // Normalize to 0-100 scale
    const normalized = {
      complexity: this.normalizeComplexity(complexity),
      duplication: duplication.percentage * 100,
      coupling: this.normalizeCoupling(coupling),
      size: this.normalizeSize(size),
      documentation: (1 - documentation.coverage) * 100, // Gap, not coverage
    };

    // Weighted total score
    const totalScore =
      0.3 * normalized.complexity +
      0.25 * normalized.duplication +
      0.2 * normalized.coupling +
      0.15 * normalized.size +
      0.1 * normalized.documentation;

    return {
      filePath,
      totalScore,
      complexity,
      duplication,
      coupling,
      size,
      documentation,
      normalized,
      timestamp: new Date(),
    };
  }

  private normalizeComplexity(complexity: ComplexityMetrics): number {
    // Cyclomatic complexity thresholds
    const cycloScore = Math.min((complexity.cyclomatic / 20) * 100, 100);
    const cognitiveScore = Math.min((complexity.cognitive / 30) * 100, 100);
    const nestingScore = Math.min((complexity.nestingDepth / 5) * 100, 100);

    return cycloScore * 0.5 + cognitiveScore * 0.3 + nestingScore * 0.2;
  }

  private normalizeCoupling(coupling: CouplingMetrics): number {
    // Instability = Ce / (Ca + Ce)
    return coupling.instability * 100;
  }

  private normalizeSize(size: SizeMetrics): number {
    // Lines of code thresholds
    const locScore = Math.min((size.linesOfCode / 500) * 100, 100);
    const funcScore = size.functions > 10 ? Math.min(((size.functions - 10) / 20) * 100, 100) : 0;

    return locScore * 0.7 + funcScore * 0.3;
  }
}
```

### Opportunity Identification

```typescript
class OpportunityIdentifier {
  identify(ast: AST, metrics: ChaosMetrics): RefactoringOpportunity[] {
    const opportunities: RefactoringOpportunity[] = [];

    // Long functions
    if (metrics.complexity.cyclomatic > 10) {
      opportunities.push({
        type: 'extract_function',
        filePath: metrics.filePath,
        location: this.findComplexFunction(ast),
        description: 'Function is too complex and should be broken down',
        impact: { complexityReduction: 40, maintainabilityImprovement: 30 },
        risk: { level: 'medium', breakingChangeProbability: 0.1 },
      });
    }

    // Code duplication
    if (metrics.duplication.percentage > 0.3) {
      opportunities.push({
        type: 'extract_common_code',
        filePath: metrics.filePath,
        location: metrics.duplication.duplicatedBlocks[0],
        description: 'Significant code duplication detected',
        impact: { maintainabilityImprovement: 50, sizeReduction: 25 },
        risk: { level: 'low', breakingChangeProbability: 0.05 },
      });
    }

    // Missing documentation
    if (metrics.documentation.coverage < 0.5) {
      opportunities.push({
        type: 'add_documentation',
        filePath: metrics.filePath,
        description: 'Functions missing documentation',
        impact: { readabilityImprovement: 60 },
        risk: { level: 'low', breakingChangeProbability: 0.0 },
      });
    }

    return opportunities;
  }
}
```

---

## 6. Refactoring Engine

### Purpose

Generates and applies safe code refactorings to reduce technical debt while ensuring correctness through validation and testing.

### Responsibilities

- Generate refactoring transformations
- Validate safety of proposed changes
- Apply transformations with rollback capability
- Integrate with testing frameworks
- Create pull requests for review

### Refactoring Operations

```typescript
abstract class RefactoringOperation {
  abstract type: RefactoringType;
  abstract description: string;

  abstract generate(ast: AST, location: Location): RefactoringPlan;
  abstract validate(plan: RefactoringPlan): ValidationResult;
  abstract apply(plan: RefactoringPlan): TransformationResult;
  abstract rollback(result: TransformationResult): void;
}

class ExtractFunctionRefactoring extends RefactoringOperation {
  type = 'extract_function';

  generate(ast: AST, location: Location): RefactoringPlan {
    const selectedCode = this.extractCodeBlock(ast, location);
    const functionName = this.generateFunctionName(selectedCode);
    const parameters = this.identifyParameters(selectedCode);

    return {
      type: this.type,
      originalLocation: location,
      newFunction: {
        name: functionName,
        parameters,
        body: selectedCode,
        returnType: this.inferReturnType(selectedCode),
      },
      replacement: {
        type: 'function_call',
        name: functionName,
        arguments: parameters.map((p) => p.name),
      },
    };
  }

  validate(plan: RefactoringPlan): ValidationResult {
    const issues: ValidationIssue[] = [];

    // Check for variable scoping issues
    if (this.hasVariableConflicts(plan)) {
      issues.push({
        severity: 'error',
        message: 'Variable scoping conflict detected',
      });
    }

    // Check for side effects
    if (this.hasSideEffects(plan.newFunction.body)) {
      issues.push({
        severity: 'warning',
        message: 'Extracted code has side effects',
      });
    }

    return {
      valid: issues.filter((i) => i.severity === 'error').length === 0,
      issues,
    };
  }
}
```

### Safety Validation

```typescript
class SafetyValidator {
  async validateRefactoring(plan: RefactoringPlan): Promise<SafetyReport> {
    const checks = await Promise.all([
      this.checkSyntax(plan),
      this.checkTypes(plan),
      this.checkTests(plan),
      this.checkBreakingChanges(plan),
      this.checkPerformance(plan),
    ]);

    const safe = checks.every((check) => check.passed);
    const riskLevel = this.calculateRiskLevel(checks);

    return { safe, riskLevel, checks };
  }

  private async checkSyntax(plan: RefactoringPlan): Promise<CheckResult> {
    try {
      const transformedCode = this.applyTransformation(plan);
      const ast = this.parseCode(transformedCode);
      return { passed: true, message: 'Syntax is valid' };
    } catch (error) {
      return { passed: false, message: `Syntax error: ${error.message}` };
    }
  }

  private async checkTypes(plan: RefactoringPlan): Promise<CheckResult> {
    if (!this.hasTypeChecker()) {
      return { passed: true, message: 'No type checker available' };
    }

    const transformedCode = this.applyTransformation(plan);
    const typeErrors = await this.runTypeChecker(transformedCode);

    return {
      passed: typeErrors.length === 0,
      message: typeErrors.length > 0 ? `Type errors: ${typeErrors.join(', ')}` : 'Types are valid',
    };
  }

  private async checkTests(plan: RefactoringPlan): Promise<CheckResult> {
    const testResult = await this.runTests();

    return {
      passed: testResult.passed,
      message: testResult.passed
        ? 'All tests pass'
        : `Tests failed: ${testResult.failures.join(', ')}`,
    };
  }
}
```

### Application with Rollback

```typescript
class RefactoringApplier {
  async apply(plan: RefactoringPlan): Promise<RefactoringResult> {
    // 1. Create backup
    const backup = await this.createBackup(plan.filePath);

    try {
      // 2. Validate safety
      const safety = await this.validator.validateRefactoring(plan);
      if (!safety.safe) {
        throw new SafetyValidationError(safety);
      }

      // 3. Apply transformation
      const transformedCode = this.applyTransformation(plan);
      await this.writeFile(plan.filePath, transformedCode);

      // 4. Run tests
      const testResult = await this.runTests();
      if (!testResult.passed) {
        throw new TestFailureError(testResult);
      }

      // 5. Record successful application
      await this.recordSuccess(plan, safety, testResult);

      return {
        success: true,
        plan,
        safety,
        testResult,
        backup,
      };
    } catch (error) {
      // Rollback on failure
      await this.rollback(backup);
      throw error;
    }
  }

  private async createBackup(filePath: string): Promise<Backup> {
    const content = await this.readFile(filePath);
    const backupPath = `${filePath}.backup.${Date.now()}`;

    await this.writeFile(backupPath, content);

    return {
      originalPath: filePath,
      backupPath,
      timestamp: new Date(),
      content,
    };
  }

  private async rollback(backup: Backup): Promise<void> {
    await this.writeFile(backup.originalPath, backup.content);
    await this.deleteFile(backup.backupPath);
  }
}
```

---

## 7. Optimization Service

### Purpose

Continuously monitors repositories and applies automated optimizations on a scheduled basis to maintain code quality.

### Responsibilities

- Schedule and execute optimization runs
- Coordinate analysis and refactoring operations
- Track optimization impact and effectiveness
- Generate reports and pull requests
- Learn from feedback to improve future runs

### Continuous Optimization Loop

```typescript
class OptimizationService {
  async startContinuousOptimization(config: OptimizationConfig): Promise<void> {
    this.running = true;

    while (this.running) {
      try {
        // Check if it's time for next optimization run
        if (this.shouldRunOptimization()) {
          await this.runOptimizationCycle();
        }

        // Wait before next check
        await this.delay(config.checkIntervalSeconds * 1000);
      } catch (error) {
        await this.handleOptimizationError(error);
      }
    }
  }

  private async runOptimizationCycle(): Promise<void> {
    const cycleId = generateId();

    try {
      // 1. Analyze repository
      const analysis = await this.analyzer.analyze(this.repository);

      // 2. Filter opportunities based on configuration
      const eligibleOpportunities = this.filterOpportunities(analysis.opportunities, this.config);

      // 3. Apply safe refactorings
      const appliedRefactorings = [];
      for (const opportunity of eligibleOpportunities) {
        if (this.shouldApplyRefactoring(opportunity)) {
          try {
            const result = await this.engine.applyRefactoring(opportunity);
            appliedRefactorings.push(result);
          } catch (error) {
            await this.handleRefactoringError(opportunity, error);
          }
        }
      }

      // 4. Generate report
      const report = await this.generateOptimizationReport(cycleId, analysis, appliedRefactorings);

      // 5. Create pull request if configured
      if (this.config.createPullRequests && appliedRefactorings.length > 0) {
        await this.createOptimizationPR(report);
      }

      // 6. Record cycle completion
      await this.recordCycleCompletion(cycleId, report);
    } catch (error) {
      await this.recordCycleFailure(cycleId, error);
    }
  }

  private shouldRunOptimization(): boolean {
    const now = new Date();
    const lastRun = this.lastOptimizationRun;

    // Check daily schedule
    if (this.config.schedule.daily) {
      const hoursSinceLastRun = (now.getTime() - lastRun.getTime()) / (1000 * 60 * 60);
      if (hoursSinceLastRun >= 24) return true;
    }

    // Check weekly schedule
    if (this.config.schedule.weekly) {
      const daysSinceLastRun = (now.getTime() - lastRun.getTime()) / (1000 * 60 * 60 * 24);
      if (daysSinceLastRun >= 7) return true;
    }

    // Check monthly schedule
    if (this.config.schedule.monthly) {
      const monthsSinceLastRun = this.getMonthsDifference(now, lastRun);
      if (monthsSinceLastRun >= 1) return true;
    }

    return false;
  }

  private filterOpportunities(
    opportunities: RefactoringOpportunity[],
    config: OptimizationConfig
  ): RefactoringOpportunity[] {
    return opportunities.filter((opp) => {
      // Filter by risk level
      if (config.maxRiskLevel === 'low' && opp.risk.level !== 'low') {
        return false;
      }

      // Filter by minimum impact
      const impactScore = this.calculateImpactScore(opp.impact);
      if (impactScore < config.minImpactScore) {
        return false;
      }

      // Filter by file types
      if (config.includeFileTypes && config.includeFileTypes.length > 0) {
        const fileExt = path.extname(opp.filePath);
        if (!config.includeFileTypes.includes(fileExt)) {
          return false;
        }
      }

      return true;
    });
  }

  private shouldApplyRefactoring(opportunity: RefactoringOpportunity): boolean {
    // Check if refactoring is enabled for this type
    if (!this.config.enabledRefactoringTypes.includes(opportunity.type)) {
      return false;
    }

    // Check risk threshold
    if (this.config.maxRiskLevel === 'low' && opportunity.risk.level !== 'low') {
      return false;
    }

    // Check test coverage requirement
    if (
      this.config.requireTestCoverage &&
      opportunity.risk.testCoverage < this.config.minTestCoverage
    ) {
      return false;
    }

    // Check if we've exceeded daily limits
    if (this.dailyRefactoringCount >= this.config.maxDailyRefactorings) {
      return false;
    }

    return true;
  }
}
```

### Learning and Adaptation

```typescript
class OptimizationLearner {
  async learnFromCycle(cycle: OptimizationCycle): Promise<void> {
    // Analyze successful refactorings
    for (const refactoring of cycle.appliedRefactorings) {
      await this.learnFromSuccess(refactoring);
    }

    // Analyze failed refactorings
    for (const failure of cycle.failedRefactorings) {
      await this.learnFromFailure(failure);
    }

    // Update models
    await this.updatePredictionModels();
    await this.updateRiskModels();
  }

  private async learnFromSuccess(refactoring: AppliedRefactoring): Promise<void> {
    // Update success patterns
    await this.patternStore.recordSuccess({
      type: refactoring.type,
      fileType: path.extname(refactoring.filePath),
      complexity: refactoring.metrics.complexityReduction,
      risk: refactoring.risk,
      context: refactoring.context,
    });

    // Update impact predictions
    await this.impactModel.train({
      input: this.extractFeatures(refactoring),
      output: refactoring.actualImpact,
    });
  }

  private async learnFromFailure(failure: FailedRefactoring): Promise<void> {
    // Update failure patterns
    await this.patternStore.recordFailure({
      type: failure.opportunity.type,
      reason: failure.reason,
      context: failure.context,
    });

    // Update risk assessments
    await this.riskModel.adjustRisk(
      failure.opportunity.type,
      failure.reason,
      +0.1 // Increase risk assessment
    );
  }
}
```

This comprehensive component specification provides the foundation for understanding how ORCHEX orchestrates AI agents, manages tasks, analyzes code, and continuously optimizes repositories. Each component is designed with resilience, scalability, and safety in mind, ensuring reliable operation in production environments.</instructions>
