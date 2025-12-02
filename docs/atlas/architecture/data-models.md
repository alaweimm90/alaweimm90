# Data Models and Schemas

Complete specification of all data models, schemas, and data structures used in the ATLAS system for tasks, agents, metrics, and configuration.

---

## Core Data Models

### Task Model

```typescript
interface Task {
  // Unique identifier
  taskId: string;

  // Task metadata
  type: TaskType;
  description: string;
  priority: TaskPriority;
  status: TaskStatus;
  createdAt: Date;
  updatedAt: Date;

  // Execution context
  context: TaskContext;

  // Requirements and constraints
  requirements: TaskRequirements;
  constraints: TaskConstraints;

  // Execution results
  result?: TaskResult;
  error?: TaskError;

  // Metadata
  metadata: Record<string, any>;
  tags: string[];
}

enum TaskType {
  CODE_GENERATION = 'code_generation',
  CODE_REVIEW = 'code_review',
  DEBUGGING = 'debugging',
  REFACTORING = 'refactoring',
  DOCUMENTATION = 'documentation',
  TESTING = 'testing',
  ARCHITECTURE = 'architecture',
  SECURITY_ANALYSIS = 'security_analysis',
}

enum TaskPriority {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical',
}

enum TaskStatus {
  QUEUED = 'queued',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
  ESCALATED = 'escalated',
}

interface TaskContext {
  repository?: string;
  branch?: string;
  commit?: string;
  files?: string[];
  language?: string;
  framework?: string;
  environment?: Record<string, any>;
}

interface TaskRequirements {
  requiredCapabilities: Capability[];
  estimatedTokens?: number;
  timeoutSeconds?: number;
  qualityThreshold?: number;
}

interface TaskConstraints {
  maxCost?: number;
  maxTokens?: number;
  preferredProviders?: Provider[];
  excludedAgents?: string[];
  deadline?: Date;
}

interface TaskResult {
  success: boolean;
  output?: string;
  code?: string;
  explanation?: string;
  suggestions?: string[];
  metrics: ExecutionMetrics;
  agentId: string;
  tier: number;
  attempts: number;
  durationMs: number;
  cost: number;
}

interface TaskError {
  code: string;
  message: string;
  details?: any;
  retryable: boolean;
  escalationRequired: boolean;
}

interface ExecutionMetrics {
  tokensUsed: number;
  tokensPrompt: number;
  tokensCompletion: number;
  responseTimeMs: number;
  cost: number;
}
```

### Agent Model

```typescript
interface Agent {
  // Identity
  agentId: string;
  name: string;
  provider: Provider;
  model: string;
  version: string;

  // Capabilities and constraints
  capabilities: Capability[];
  constraints: AgentConstraints;

  // Health and status
  health: HealthStatus;
  status: AgentStatus;

  // Performance metrics
  performance: PerformanceMetrics;

  // Configuration
  config: AgentConfig;

  // Metadata
  registeredAt: Date;
  lastSeen: Date;
  metadata: Record<string, any>;
}

enum Provider {
  ANTHROPIC = 'anthropic',
  OPENAI = 'openai',
  GOOGLE = 'google',
  LOCAL = 'local',
}

enum Capability {
  CODE_GENERATION = 'code_generation',
  CODE_REVIEW = 'code_review',
  DEBUGGING = 'debugging',
  REFACTORING = 'refactoring',
  DOCUMENTATION = 'documentation',
  TESTING = 'testing',
  ARCHITECTURE = 'architecture',
  SECURITY_ANALYSIS = 'security_analysis',
}

enum AgentStatus {
  ACTIVE = 'active',
  INACTIVE = 'inactive',
  MAINTENANCE = 'maintenance',
  DEPRECATED = 'deprecated',
}

interface AgentConstraints {
  maxTokens: number;
  maxConcurrentTasks: number;
  rateLimitPerMinute: number;
  costPer1kTokens: number;
  supportedLanguages: string[];
  maxFileSizeBytes: number;
  timeoutSeconds: number;
}

interface HealthStatus {
  status: HealthState;
  lastCheck: Date;
  uptimePercentage: number;
  avgResponseTimeMs: number;
  errorRate: number;
  consecutiveFailures: number;
  totalRequests: number;
  totalErrors: number;
}

enum HealthState {
  HEALTHY = 'healthy',
  DEGRADED = 'degraded',
  UNHEALTHY = 'unhealthy',
  OFFLINE = 'offline',
}

interface PerformanceMetrics {
  successRate: number;
  avgQualityScore: number;
  avgResponseTimeMs: number;
  totalTasksCompleted: number;
  totalTasksFailed: number;
  avgCostPerTask: number;
  avgTokensPerTask: number;
}

interface AgentConfig {
  baseWeight: number;
  retryPolicy: RetryPolicy;
  fallbackBehavior: FallbackBehavior;
  costOptimization: boolean;
}

interface RetryPolicy {
  maxRetries: number;
  backoffMultiplier: number;
  maxBackoffSeconds: number;
}

interface FallbackBehavior {
  enabled: boolean;
  maxFallbackTiers: number;
  escalateAfterFailures: number;
}
```

### Repository Analysis Models

```typescript
interface AnalysisReport {
  analysisId: string;
  repository: string;
  branch: string;
  commit: string;
  timestamp: Date;

  // Summary
  summary: AnalysisSummary;

  // Detailed metrics
  chaosMetrics: ChaosMetrics[];
  opportunities: RefactoringOpportunity[];

  // Metadata
  config: AnalysisConfig;
  durationMs: number;
  status: AnalysisStatus;
}

interface AnalysisSummary {
  totalFiles: number;
  filesAnalyzed: number;
  avgChaosScore: number;
  highChaosFiles: number;
  totalOpportunities: number;
  estimatedDebtHours: number;
  languages: LanguageStats[];
}

interface LanguageStats {
  language: string;
  files: number;
  linesOfCode: number;
  avgChaosScore: number;
}

enum AnalysisStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

interface ChaosMetrics {
  filePath: string;
  timestamp: Date;
  totalScore: number;

  // Component scores (0-100)
  complexity: ComplexityMetrics;
  duplication: DuplicationMetrics;
  coupling: CouplingMetrics;
  size: SizeMetrics;
  documentation: DocumentationMetrics;
}

interface ComplexityMetrics {
  cyclomatic: number;
  cognitive: number;
  nestingDepth: number;
  halsteadVolume: number;
  maintainabilityIndex: number;
}

interface DuplicationMetrics {
  percentage: number;
  duplicatedBlocks: CodeBlock[];
  duplicatedLines: number;
}

interface CouplingMetrics {
  afferent: number; // Incoming dependencies
  efferent: number; // Outgoing dependencies
  instability: number; // I = Ce / (Ca + Ce)
  abstractness: number;
  distance: number; // D = |A + I - 1|
}

interface SizeMetrics {
  linesOfCode: number;
  linesOfComments: number;
  blankLines: number;
  functions: number;
  classes: number;
  files: number;
}

interface DocumentationMetrics {
  coverage: number;
  documentedFunctions: number;
  totalFunctions: number;
  missingDocstrings: string[];
}

interface CodeBlock {
  startLine: number;
  endLine: number;
  content: string;
  hash: string;
}

interface RefactoringOpportunity {
  opportunityId: string;
  type: RefactoringType;
  filePath: string;
  location: CodeLocation;
  description: string;

  // Impact assessment
  impact: RefactoringImpact;
  risk: RefactoringRisk;

  // Context
  context: RefactoringContext;

  // Metadata
  priority: number;
  estimatedEffort: number;
  tags: string[];
}

enum RefactoringType {
  EXTRACT_FUNCTION = 'extract_function',
  EXTRACT_VARIABLE = 'extract_variable',
  RENAME_VARIABLE = 'rename_variable',
  SIMPLIFY_CONDITIONAL = 'simplify_conditional',
  REMOVE_DUPLICATION = 'remove_duplication',
  REDUCE_COMPLEXITY = 'reduce_complexity',
  IMPROVE_NAMING = 'improve_naming',
  ADD_TYPE_HINTS = 'add_type_hints',
  EXTRACT_CONSTANT = 'extract_constant',
  SPLIT_CLASS = 'split_class',
  MOVE_METHOD = 'move_method',
}

interface CodeLocation {
  startLine: number;
  endLine: number;
  startColumn: number;
  endColumn: number;
}

interface RefactoringImpact {
  complexityReduction: number;
  maintainabilityImprovement: number;
  readabilityImprovement: number;
  performanceImprovement: number;
  estimatedTimeSavedHours: number;
}

interface RefactoringRisk {
  level: RiskLevel;
  breakingChangeProbability: number;
  testCoverage: number;
  regressionProbability: number;
}

enum RiskLevel {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
}

interface RefactoringContext {
  surroundingCode: string;
  dependencies: string[];
  testCoverage: number;
  lastModified: Date;
  author: string;
}
```

### Refactoring Models

```typescript
interface RefactoringPlan {
  planId: string;
  opportunityId: string;
  type: RefactoringType;

  // Original code
  originalLocation: CodeLocation;
  originalCode: string;

  // Transformation
  transformation: RefactoringTransformation;

  // Validation
  safetyChecks: SafetyCheck[];
  testRequirements: TestRequirement[];

  // Metadata
  createdAt: Date;
  estimatedDuration: number;
}

interface RefactoringTransformation {
  type: TransformationType;
  changes: CodeChange[];
  newElements: NewCodeElement[];
  dependencies: DependencyChange[];
}

enum TransformationType {
  REPLACE = 'replace',
  INSERT = 'insert',
  DELETE = 'delete',
  MOVE = 'move',
  RENAME = 'rename',
}

interface CodeChange {
  location: CodeLocation;
  oldCode: string;
  newCode: string;
  description: string;
}

interface NewCodeElement {
  type: 'function' | 'class' | 'variable' | 'constant';
  name: string;
  location: CodeLocation;
  code: string;
}

interface DependencyChange {
  type: 'add' | 'remove' | 'update';
  dependency: string;
  version?: string;
}

interface SafetyCheck {
  checkId: string;
  type: SafetyCheckType;
  description: string;
  required: boolean;
}

enum SafetyCheckType {
  SYNTAX_VALIDATION = 'syntax_validation',
  TYPE_CHECKING = 'type_checking',
  TEST_EXECUTION = 'test_execution',
  BREAKING_CHANGE_ANALYSIS = 'breaking_change_analysis',
  PERFORMANCE_IMPACT = 'performance_impact',
}

interface TestRequirement {
  type: TestType;
  scope: TestScope;
  description: string;
}

enum TestType {
  UNIT = 'unit',
  INTEGRATION = 'integration',
  REGRESSION = 'regression',
}

enum TestScope {
  FUNCTION = 'function',
  MODULE = 'module',
  COMPONENT = 'component',
  SYSTEM = 'system',
}

interface RefactoringResult {
  resultId: string;
  planId: string;
  success: boolean;

  // Execution details
  appliedAt: Date;
  durationMs: number;

  // Changes made
  changes: AppliedChange[];
  newFiles: NewFile[];
  modifiedFiles: ModifiedFile[];

  // Validation results
  safetyReport: SafetyReport;
  testResults: TestResults;

  // Rollback information
  rollbackAvailable: boolean;
  rollbackData?: RollbackData;

  // Metadata
  agentId: string;
  tier: number;
  cost: number;
}

interface AppliedChange {
  filePath: string;
  changeType: ChangeType;
  location: CodeLocation;
  oldCode: string;
  newCode: string;
}

enum ChangeType {
  MODIFIED = 'modified',
  ADDED = 'added',
  DELETED = 'deleted',
}

interface NewFile {
  path: string;
  content: string;
  type: FileType;
}

interface ModifiedFile {
  path: string;
  changes: AppliedChange[];
  checksum: string;
}

enum FileType {
  SOURCE = 'source',
  TEST = 'test',
  CONFIG = 'config',
  DOCUMENTATION = 'documentation',
}

interface SafetyReport {
  overallSafety: SafetyLevel;
  checks: SafetyCheckResult[];
  riskAssessment: RiskAssessment;
}

enum SafetyLevel {
  SAFE = 'safe',
  WARNING = 'warning',
  UNSAFE = 'unsafe',
}

interface SafetyCheckResult {
  checkId: string;
  passed: boolean;
  message: string;
  details?: any;
  severity: 'info' | 'warning' | 'error';
}

interface RiskAssessment {
  overallRisk: RiskLevel;
  breakingChanges: BreakingChange[];
  performanceImpact: PerformanceImpact;
  regressionRisk: number;
}

interface BreakingChange {
  type: BreakingChangeType;
  description: string;
  severity: 'low' | 'medium' | 'high';
  mitigation?: string;
}

enum BreakingChangeType {
  API_CHANGE = 'api_change',
  BEHAVIOR_CHANGE = 'behavior_change',
  DEPENDENCY_CHANGE = 'dependency_change',
}

interface PerformanceImpact {
  expectedChange: number; // percentage
  confidence: number; // 0-1
  benchmarks: BenchmarkResult[];
}

interface BenchmarkResult {
  test: string;
  baseline: number;
  result: number;
  change: number;
}

interface TestResults {
  passed: boolean;
  totalTests: number;
  passedTests: number;
  failedTests: number;
  skippedTests: number;
  durationMs: number;
  failures: TestFailure[];
}

interface TestFailure {
  testName: string;
  error: string;
  stackTrace?: string;
}

interface RollbackData {
  backupFiles: BackupFile[];
  originalState: SystemState;
  rollbackSteps: RollbackStep[];
}

interface BackupFile {
  originalPath: string;
  backupPath: string;
  checksum: string;
}

interface SystemState {
  gitCommit: string;
  fileChecksums: Record<string, string>;
  databaseState?: any;
}

interface RollbackStep {
  stepId: string;
  type: 'restore_file' | 'run_command' | 'revert_commit';
  target: string;
  data?: any;
}
```

### Metrics and Telemetry Models

```typescript
interface MetricsSnapshot {
  snapshotId: string;
  timestamp: Date;
  period: MetricsPeriod;

  // System metrics
  system: SystemMetrics;

  // Agent metrics
  agents: AgentMetrics[];

  // Task metrics
  tasks: TaskMetrics;

  // Repository metrics
  repositories: RepositoryMetrics[];

  // Cost metrics
  costs: CostMetrics;
}

enum MetricsPeriod {
  HOUR = 'hour',
  DAY = 'day',
  WEEK = 'week',
  MONTH = 'month',
}

interface SystemMetrics {
  uptimeSeconds: number;
  totalRequests: number;
  activeConnections: number;
  memoryUsage: MemoryMetrics;
  cpuUsage: CpuMetrics;
  diskUsage: DiskMetrics;
  networkUsage: NetworkMetrics;
}

interface MemoryMetrics {
  used: number;
  total: number;
  percentage: number;
  swapUsed: number;
  swapTotal: number;
}

interface CpuMetrics {
  usage: number;
  loadAverage: number[];
  cores: number;
}

interface DiskMetrics {
  used: number;
  total: number;
  percentage: number;
  readBytesPerSecond: number;
  writeBytesPerSecond: number;
}

interface NetworkMetrics {
  bytesReceivedPerSecond: number;
  bytesSentPerSecond: number;
  packetsReceivedPerSecond: number;
  packetsSentPerSecond: number;
}

interface AgentMetrics {
  agentId: string;
  status: AgentStatus;
  health: HealthStatus;
  performance: PerformanceMetrics;
  load: AgentLoad;
  costs: AgentCostMetrics;
}

interface AgentLoad {
  activeTasks: number;
  queuedTasks: number;
  capacity: number;
  utilization: number;
}

interface AgentCostMetrics {
  totalCost: number;
  costPerHour: number;
  costPerTask: number;
  budgetRemaining: number;
}

interface TaskMetrics {
  total: number;
  completed: number;
  failed: number;
  cancelled: number;
  escalated: number;
  successRate: number;
  avgDurationMs: number;
  avgCost: number;

  // Breakdown by type
  byType: Record<TaskType, TaskTypeMetrics>;

  // Breakdown by priority
  byPriority: Record<TaskPriority, TaskPriorityMetrics>;

  // Time-based metrics
  responseTime: PercentileMetrics;
  queueTime: PercentileMetrics;
}

interface TaskTypeMetrics {
  count: number;
  successRate: number;
  avgDurationMs: number;
  avgCost: number;
}

interface TaskPriorityMetrics {
  count: number;
  avgWaitTimeMs: number;
  successRate: number;
}

interface PercentileMetrics {
  p50: number;
  p90: number;
  p95: number;
  p99: number;
}

interface RepositoryMetrics {
  repository: string;
  branch: string;
  lastAnalysis: Date;
  chaosScore: number;
  opportunities: number;
  appliedRefactorings: number;
  technicalDebtHours: number;
  trends: RepositoryTrends;
}

interface RepositoryTrends {
  chaosScoreChange: number;
  opportunitiesChange: number;
  debtHoursChange: number;
  filesImproved: number;
  filesDegraded: number;
}

interface CostMetrics {
  totalCost: number;
  costByProvider: Record<Provider, number>;
  costByTaskType: Record<TaskType, number>;
  costByAgent: Record<string, number>;
  budgetUtilization: number;
  projectedMonthlyCost: number;
}

interface TelemetryEvent {
  eventId: string;
  timestamp: Date;
  type: TelemetryEventType;
  source: string;
  level: TelemetryLevel;

  // Event data
  data: Record<string, any>;

  // Context
  context: TelemetryContext;

  // Metadata
  tags: string[];
  userId?: string;
  sessionId?: string;
}

enum TelemetryEventType {
  TASK_STARTED = 'task_started',
  TASK_COMPLETED = 'task_completed',
  TASK_FAILED = 'task_failed',
  AGENT_HEALTH_CHECK = 'agent_health_check',
  REFACTORING_APPLIED = 'refactoring_applied',
  ANALYSIS_COMPLETED = 'analysis_completed',
  SYSTEM_ERROR = 'system_error',
  USER_ACTION = 'user_action',
}

enum TelemetryLevel {
  DEBUG = 'debug',
  INFO = 'info',
  WARN = 'warn',
  ERROR = 'error',
}

interface TelemetryContext {
  requestId?: string;
  userAgent?: string;
  ipAddress?: string;
  userId?: string;
  sessionId?: string;
  repository?: string;
  branch?: string;
  agentId?: string;
  taskId?: string;
}
```

### Configuration Models

```typescript
interface AtlasConfig {
  version: string;
  environment: string;

  // Project information
  project: ProjectConfig;

  // Component configurations
  agents: AgentConfig;
  tasks: TaskConfig;
  routing: RoutingConfig;
  analysis: AnalysisConfig;
  refactoring: RefactoringConfig;
  optimization: OptimizationConfig;

  // External integrations
  integrations: IntegrationConfig;

  // Security settings
  security: SecurityConfig;

  // Logging and monitoring
  logging: LoggingConfig;
  monitoring: MonitoringConfig;

  // Cost management
  cost: CostConfig;
}

interface ProjectConfig {
  name: string;
  version: string;
  description?: string;
  repository?: string;
  language?: string;
  framework?: string;
  team?: string[];
  tags?: string[];
}

interface AgentConfig {
  defaultProvider: Provider;
  fallbackEnabled: boolean;
  healthCheckInterval: number;
  registrationTimeout: number;
  deregistrationGracePeriod: number;
}

interface TaskConfig {
  defaultTimeout: number;
  maxRetries: number;
  defaultPriority: TaskPriority;
  queueSize: number;
  cleanup: {
    enabled: boolean;
    maxAgeDays: number;
    batchSize: number;
  };
}

interface RoutingConfig {
  algorithm: RoutingAlgorithm;
  scoring: ScoringConfig;
  fallback: FallbackConfig;
  loadBalancing: LoadBalancingConfig;
}

enum RoutingAlgorithm {
  WEIGHTED_SCORING = 'weighted_scoring',
  PERFORMANCE_BASED = 'performance_based',
  COST_OPTIMIZED = 'cost_optimized',
  LOAD_BALANCED = 'load_balanced',
}

interface ScoringConfig {
  weights: {
    capability: number;
    performance: number;
    availability: number;
    cost: number;
  };
  normalization: boolean;
}

interface FallbackConfig {
  enabled: boolean;
  maxTiers: number;
  escalationEnabled: boolean;
  escalationTimeout: number;
}

interface LoadBalancingConfig {
  algorithm: LoadBalancingAlgorithm;
  queueEnabled: boolean;
  maxQueueSize: number;
}

enum LoadBalancingAlgorithm {
  ROUND_ROBIN = 'round_robin',
  LEAST_LOADED = 'least_loaded',
  WEIGHTED = 'weighted',
  PERFORMANCE = 'performance',
}

interface AnalysisConfig {
  enabled: boolean;
  schedule: AnalysisSchedule;
  scope: AnalysisScope;
  chaos: ChaosConfig;
  reporting: AnalysisReportingConfig;
}

interface AnalysisSchedule {
  daily: boolean;
  weekly: boolean;
  monthly: boolean;
  custom?: string;
}

interface AnalysisScope {
  includePatterns: string[];
  excludePatterns: string[];
  maxFileSize: number;
  maxFiles: number;
}

interface ChaosConfig {
  enabled: boolean;
  threshold: number;
  weights: {
    complexity: number;
    duplication: number;
    coupling: number;
    size: number;
    documentation: number;
  };
}

interface AnalysisReportingConfig {
  format: 'json' | 'markdown' | 'html';
  outputDir: string;
  includeMetrics: boolean;
  includeRecommendations: boolean;
}

interface RefactoringConfig {
  enabled: boolean;
  autoApply: boolean;
  safety: SafetyConfig;
  types: RefactoringTypeConfig[];
  reporting: RefactoringReportingConfig;
}

interface SafetyConfig {
  syntaxValidation: boolean;
  typeChecking: boolean;
  testExecution: boolean;
  breakingChangeAnalysis: boolean;
  performanceImpact: boolean;
  minTestCoverage: number;
}

interface RefactoringTypeConfig {
  type: RefactoringType;
  enabled: boolean;
  maxRisk: RiskLevel;
  autoApply: boolean;
}

interface RefactoringReportingConfig {
  createPRs: boolean;
  prTemplate: string;
  notify: boolean;
  notificationChannels: string[];
}

interface OptimizationConfig {
  enabled: boolean;
  schedule: OptimizationSchedule;
  scope: OptimizationScope;
  rules: OptimizationRule[];
  reporting: OptimizationReportingConfig;
}

interface OptimizationSchedule {
  intervalHours: number;
  maxDailyRuns: number;
  businessHoursOnly: boolean;
}

interface OptimizationScope {
  repositories: string[];
  branches: string[];
  fileTypes: string[];
}

interface OptimizationRule {
  name: string;
  condition: string;
  action: string;
  priority: number;
}

interface OptimizationReportingConfig {
  enabled: boolean;
  format: string;
  outputDir: string;
  notifyOnChanges: boolean;
}

interface IntegrationConfig {
  kilo: KiloIntegrationConfig;
  github: GitHubIntegrationConfig;
  slack: SlackIntegrationConfig;
  jira: JiraIntegrationConfig;
}

interface KiloIntegrationConfig {
  enabled: boolean;
  endpoint: string;
  apiKey: string;
  syncInterval: number;
  policies: string[];
}

interface GitHubIntegrationConfig {
  enabled: boolean;
  token: string;
  repositories: string[];
  autoPR: boolean;
  labels: string[];
}

interface SlackIntegrationConfig {
  enabled: boolean;
  webhook: string;
  channels: string[];
  events: string[];
}

interface JiraIntegrationConfig {
  enabled: boolean;
  endpoint: string;
  username: string;
  apiToken: string;
  project: string;
}

interface SecurityConfig {
  apiKeys: APIKeyConfig;
  authentication: AuthConfig;
  encryption: EncryptionConfig;
  audit: AuditConfig;
}

interface APIKeyConfig {
  required: boolean;
  rotationDays: number;
  allowedIPs: string[];
}

interface AuthConfig {
  method: 'api_key' | 'jwt' | 'oauth';
  jwtSecret?: string;
  oauthConfig?: OAuthConfig;
}

interface EncryptionConfig {
  enabled: boolean;
  algorithm: string;
  keyRotationDays: number;
}

interface AuditConfig {
  enabled: boolean;
  logLevel: string;
  retentionDays: number;
  sensitiveFields: string[];
}

interface LoggingConfig {
  level: 'debug' | 'info' | 'warn' | 'error';
  format: 'json' | 'text';
  file: string;
  maxSize: string;
  maxFiles: number;
  console: boolean;
}

interface MonitoringConfig {
  enabled: boolean;
  metrics: MetricsConfig;
  alerting: AlertingConfig;
  dashboards: DashboardConfig;
}

interface MetricsConfig {
  collectionInterval: number;
  retentionDays: number;
  exporters: string[];
}

interface AlertingConfig {
  enabled: boolean;
  rules: AlertRule[];
  channels: NotificationChannel[];
}

interface AlertRule {
  name: string;
  condition: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  channels: string[];
}

interface NotificationChannel {
  name: string;
  type: 'email' | 'slack' | 'webhook';
  config: Record<string, any>;
}

interface DashboardConfig {
  enabled: boolean;
  provider: 'grafana' | 'datadog' | 'custom';
  endpoint?: string;
  apiKey?: string;
}

interface CostConfig {
  maxPerTask: number;
  maxPerDay: number;
  maxPerMonth: number;
  alertThreshold: number;
  notifications: boolean;
  budgetAlerts: BudgetAlert[];
}

interface BudgetAlert {
  threshold: number;
  channels: string[];
  message: string;
}
```

These comprehensive data models provide the foundation for all ATLAS operations, ensuring type safety, consistency, and extensibility across the entire system. Each model includes detailed specifications for validation, serialization, and integration with other components.</instructions>
