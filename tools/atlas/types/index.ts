/**
 * ATLAS Core Type Definitions
 * Multiagent LLM Orchestration System
 */

// ============================================================================
// Agent Types
// ============================================================================

export type AgentProvider = 'anthropic' | 'openai' | 'google' | 'local' | 'custom';

export type AgentCapability =
  | 'code_generation'
  | 'code_review'
  | 'refactoring'
  | 'documentation'
  | 'testing'
  | 'debugging'
  | 'analysis'
  | 'explanation'
  | 'chat';

export type AgentStatus = 'available' | 'busy' | 'unavailable' | 'circuit_open';

export interface Agent {
  id: string;
  name: string;
  provider: AgentProvider;
  model: string;
  capabilities: AgentCapability[];
  status: AgentStatus;
  config: AgentConfig;
  metrics: AgentMetrics;
  registeredAt: string;
  lastUsed?: string;
}

export interface AgentConfig {
  apiKeyEnv?: string;
  endpoint?: string;
  maxTokens: number;
  temperature: number;
  timeout: number;
  retries: number;
  costPerToken?: number;
}

export interface AgentMetrics {
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
  avgLatency: number;
  totalTokens: number;
  errorRate: number;
}

// ============================================================================
// Task Types
// ============================================================================

export type TaskType =
  | 'code_generation'
  | 'code_review'
  | 'refactoring'
  | 'documentation'
  | 'testing'
  | 'debugging'
  | 'analysis'
  | 'explanation'
  | 'chat';

export type TaskStatus =
  | 'pending'
  | 'routing'
  | 'in_progress'
  | 'completed'
  | 'failed'
  | 'cancelled';

export type TaskPriority = 'critical' | 'high' | 'medium' | 'low';

export interface Task {
  id: string;
  type: TaskType;
  description: string;
  context: TaskContext;
  priority: TaskPriority;
  status: TaskStatus;
  assignedAgent?: string;
  result?: TaskResult;
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
  metadata?: Record<string, unknown>;
}

export interface TaskContext {
  files?: string[];
  codeSnippet?: string;
  language?: string;
  framework?: string;
  additionalContext?: string;
}

export interface TaskResult {
  success: boolean;
  output?: string;
  error?: string;
  tokensUsed?: number;
  latency?: number;
  agentId?: string;
}

// ============================================================================
// Routing Types
// ============================================================================

export interface RoutingDecision {
  agentId: string;
  confidence: number;
  reasoning: string;
  estimatedCost: number;
  estimatedTime: number;
  alternatives?: AlternativeAgent[];
}

export interface AlternativeAgent {
  agentId: string;
  confidence: number;
  reason: string;
}

export interface RoutingStrategy {
  name: string;
  description: string;
  selectAgent: (task: Task, agents: Agent[]) => RoutingDecision | null;
}

// ============================================================================
// Circuit Breaker Types
// ============================================================================

export type CircuitState = 'closed' | 'open' | 'half_open';

export interface CircuitBreakerConfig {
  failureThreshold: number;
  successThreshold: number;
  timeout: number;
  halfOpenRequests: number;
}

export interface CircuitBreakerState {
  state: CircuitState;
  failures: number;
  successes: number;
  lastFailure?: string;
  lastSuccess?: string;
  openedAt?: string;
}

// ============================================================================
// Orchestration Config
// ============================================================================

export interface OrchestrationConfig {
  defaultAgent?: string;
  fallbackChain: string[];
  circuitBreaker: CircuitBreakerConfig;
  routing: {
    strategy: 'capability' | 'load_balance' | 'cost' | 'latency';
    preferences?: Record<TaskType, string>;
  };
  telemetry: {
    enabled: boolean;
    metricsPath: string;
  };
}

export interface AgentTeam {
  id: string;
  name: string;
  members: string[];
  capabilities: AgentCapability[];
  routingStrategy: 'capability' | 'load_balance' | 'cost' | 'latency';
  createdAt: string;
  updatedAt?: string;
}

// ============================================================================
// ATLAS Service DTOs
// ============================================================================

/**
 * Data Transfer Object for optimization plans
 */
export interface OptimizationPlan {
  id: string;
  repositoryPath: string;
  timestamp: Date;
  metrics: {
    chaosScore: number;
    complexityScore: number;
    issuesCount: number;
  };
  suggestions: OptimizationSuggestion[];
  priority: 'low' | 'medium' | 'high' | 'critical';
  estimatedEffort: number; // hours
}

export interface OptimizationSuggestion {
  id: string;
  type: 'refactor' | 'test' | 'document' | 'simplify' | 'modularize';
  file: string;
  line: number;
  description: string;
  impact: 'low' | 'medium' | 'high';
  confidence: number; // 0-1
}

/**
 * Data Transfer Object for dashboard widgets
 */
export interface DashboardWidget {
  id: string;
  title: string;
  type: 'metric' | 'chart' | 'table' | 'gauge' | 'heatmap';
  data: Record<string, unknown>;
  lastUpdated: Date;
  refreshInterval?: number; // milliseconds
}

export interface DashboardMetric {
  key: string;
  label: string;
  value: number | string;
  unit?: string;
  trend?: 'up' | 'down' | 'stable';
  trendValue?: number;
}

/**
 * Data Transfer Object for repository analysis results
 */
export interface RepositoryAnalysisDTO {
  repositoryPath: string;
  timestamp: Date;
  chaosScore: number;
  complexityScore: number;
  fileCount: number;
  totalLines: number;
  issuesCount: number;
  coverage?: number;
  testCount?: number;
  recommendations: string[];
}

/**
 * Data Transfer Object for telemetry events
 */
export interface TelemetryEventDTO {
  id: string;
  timestamp: Date;
  type: 'optimization' | 'analysis' | 'error' | 'metric' | 'system';
  source: string;
  data: Record<string, unknown>;
  severity: 'low' | 'medium' | 'high' | 'critical';
  tags: string[];
}

/**
 * Data Transfer Object for code changes
 */
export interface CodeChangeDTO {
  id: string;
  repository: string;
  type: 'file' | 'commit' | 'push';
  files: FileChangeDTO[];
  timestamp: Date;
  commitHash?: string;
  author?: string;
  message?: string;
}

export interface FileChangeDTO {
  path: string;
  type: 'added' | 'modified' | 'deleted' | 'renamed';
  oldPath?: string;
  size?: number;
  linesChanged?: number;
}

// ============================================================================
// Code Analysis Types
// ============================================================================

/**
 * Repository metrics for code analysis
 */
export interface RepositoryMetrics {
  totalFiles: number;
  totalLines: number;
  totalFunctions: number;
  totalClasses: number;
  averageComplexity: number;
  testCoverage?: number;
  duplicateCodePercentage?: number;
}

/**
 * Code analysis results
 */
export interface CodeAnalysis {
  repository: string;
  timestamp: Date;
  metrics: RepositoryMetrics;
  functions: FunctionInfo[];
  classes: ClassInfo[];
  imports: ImportInfo[];
  issues: CodeIssue[];
  // Additional properties for compatibility
  chaosScore?: number;
  complexityScore?: number;
  files?: string[];
  totalLines?: number;
}

/**
 * Function information from code analysis
 */
export interface FunctionInfo {
  name: string;
  file: string;
  line: number;
  complexity: number;
  parameters: number;
  linesOfCode: number;
  isAsync: boolean;
  isExported: boolean;
}

/**
 * Class information from code analysis
 */
export interface ClassInfo {
  name: string;
  file: string;
  line: number;
  methods: number;
  properties: number;
  linesOfCode: number;
  isAbstract: boolean;
  isExported: boolean;
}

/**
 * Import information from code analysis
 */
export interface ImportInfo {
  file: string;
  importPath: string;
  isRelative: boolean;
  isTypeOnly: boolean;
  namedImports?: string[];
  defaultImport?: string;
}

/**
 * Code issue found during analysis
 */
export interface CodeIssue {
  type: 'error' | 'warning' | 'info';
  category: 'complexity' | 'duplication' | 'style' | 'security' | 'performance';
  file: string;
  line: number;
  message: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  suggestion?: string;
}

/**
 * Refactoring suggestion from analysis
 */
export interface RefactoringSuggestion {
  id: string;
  type: 'extract' | 'inline' | 'rename' | 'move' | 'simplify';
  file: string;
  startLine: number;
  endLine: number;
  description: string;
  impact: 'low' | 'medium' | 'high';
  confidence: number;
  preview?: string;
}
