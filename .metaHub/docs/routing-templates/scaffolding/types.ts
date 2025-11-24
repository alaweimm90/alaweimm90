/**
 * TRAE Three-Tier LLM Routing System Types Template
 * Comprehensive Type Definitions for Framework-Agnostic Implementation
 *
 * This template provides complete type definitions that can be adapted to any
 * programming language or framework, ensuring type safety and consistency
 * across different implementations.
 *
 * Features:
 * - Framework-agnostic type definitions
 * - Language-flexible interfaces
 * - Compliance-ready structures
 * - Production-hardened error handling
 * - Extensible architecture patterns
 */

// ============================================================================
// CORE SYSTEM TYPES
// ============================================================================

export enum TaskComplexity {
  SIMPLE = 'simple',
  MODERATE = 'moderate',
  COMPLEX = 'complex',
  CRITICAL = 'critical',
}

export enum GeographicRegion {
  NORTH_AMERICA = 'north_america',
  EUROPE = 'europe',
  ASIA_PACIFIC = 'asia_pacific',
  SOUTH_AMERICA = 'south_america',
  AFRICA = 'africa',
  GLOBAL = 'global',
}

export enum RoutingTier {
  TIER_1 = 'tier_1',
  TIER_2 = 'tier_2',
  TIER_3 = 'tier_3',
}

export enum CostOptimizationMode {
  AGGRESSIVE = 'aggressive',
  BALANCED = 'balanced',
  QUALITY_FIRST = 'quality_first',
}

export enum RiskLevel {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical',
}

// ============================================================================
// TASK ANALYSIS TYPES
// ============================================================================

export interface TaskAnalysis {
  complexity: TaskComplexity;
  estimatedTokens: number;
  requiredCapabilities: string[];
  timeSensitivity: 'low' | 'medium' | 'high' | 'critical';
  costSensitivity: 'low' | 'medium' | 'high';
  geographicPreference?: GeographicRegion;
  contextLength: number;
  domain: string[];
  priority?: 'low' | 'normal' | 'high' | 'critical';
  dataClassification?: DataClassification;
  complianceRequirements?: ComplianceFramework[];
}

export enum DataClassification {
  PUBLIC = 'public',
  INTERNAL = 'internal',
  CONFIDENTIAL = 'confidential',
  RESTRICTED = 'restricted',
}

export enum ComplianceFramework {
  HIPAA = 'hipaa',
  GDPR = 'gdpr',
  SOC2 = 'soc2',
  PCI_DSS = 'pci_dss',
}

// ============================================================================
// MODEL CAPABILITY TYPES
// ============================================================================

export interface ModelCapability {
  name: string;
  provider: string;
  model: string;
  tier: RoutingTier;
  region: GeographicRegion;
  maxTokens: number;
  costPerToken: number;
  qualityScore: number;
  latency: number;
  reliability: number;
  specializations: string[];
  limitations: string[];
  compliance: ComplianceFramework[];
  enabled: boolean;
  rateLimits?: RateLimits;
  capabilities?: ModelCapabilities;
  metadata?: ModelMetadata;
}

export interface RateLimits {
  requestsPerMinute: number;
  tokensPerMinute: number;
  requestsPerHour?: number;
  tokensPerHour?: number;
  concurrentRequests?: number;
}

export interface ModelCapabilities {
  multimodal: boolean;
  functionCalling: boolean;
  streaming: boolean;
  fineTuning: boolean;
  jsonMode?: boolean;
  vision?: boolean;
  audio?: boolean;
}

export interface ModelMetadata {
  version: string;
  releaseDate: string;
  trainingDataSize?: string;
  architecture?: string;
  license?: string;
}

// ============================================================================
// ROUTING DECISION TYPES
// ============================================================================

export interface RoutingDecision {
  selectedModel: ModelCapability;
  fallbackChain: ModelCapability[];
  estimatedCost: number;
  estimatedLatency: number;
  confidence: number;
  reasoning: string[];
  costSavings: number;
  riskAssessment: RiskLevel;
  complianceValidation?: ComplianceValidation;
  metadata?: RoutingMetadata;
}

export interface ComplianceValidation {
  framework: ComplianceFramework;
  validated: boolean;
  requirements: string[];
  violations: string[];
  complianceSurcharge?: number;
  auditTrail: AuditEntry[];
}

export interface AuditEntry {
  action: string;
  timestamp: Date;
  details: any;
  complianceImpact: string;
}

export interface RoutingMetadata {
  decisionId: string;
  timestamp: Date;
  processingTime: number;
  alternativesConsidered: number;
  costOptimizationApplied: boolean;
}

// ============================================================================
// COST MANAGEMENT TYPES
// ============================================================================

export interface CostBudget {
  dailyLimit: number;
  monthlyLimit: number;
  perRequestLimit: number;
  currentDailyUsage: number;
  currentMonthlyUsage: number;
  alertThreshold: number;
  complianceBudget?: number;
  emergencyThreshold?: number;
  resetSchedule?: 'daily' | 'monthly';
  currency?: string;
}

export interface CostAnalysis {
  currentSavings: number;
  potentialSavings: number;
  optimizationOpportunities: OptimizationOpportunity[];
  trend: 'improving' | 'stable' | 'degrading';
  costBreakdown: CostBreakdown;
}

export interface OptimizationOpportunity {
  type: string;
  description: string;
  potentialSavings: number;
  implementationEffort: 'low' | 'medium' | 'high';
  risk: 'low' | 'medium' | 'high';
  priority: 'low' | 'normal' | 'high' | 'critical';
}

export interface CostBreakdown {
  baseCost: number;
  optimizationSavings: number;
  geographicAdjustment: number;
  tierDiscount: number;
  finalCost: number;
  currency: string;
}

// ============================================================================
// GEOGRAPHIC FALLBACK TYPES
// ============================================================================

export interface GeographicFallbackChain {
  primary: GeographicRegion;
  fallbacks: GeographicRegion[];
  latencyThreshold: number;
  costMultiplier: number;
  compliancePriority?: boolean;
  healthCheckInterval?: number;
}

export interface RegionHealth {
  region: GeographicRegion;
  healthy: boolean;
  latency: number;
  lastCheck: Date;
  consecutiveFailures: number;
  totalRequests: number;
  successfulRequests: number;
  errorRate: number;
}

// ============================================================================
// PROMPT MANAGEMENT TYPES
// ============================================================================

export interface PromptTemplate {
  modelFamily: string;
  template: string;
  variables: string[];
  optimizations: PromptOptimization[];
  constraints: PromptConstraint[];
  metadata?: TemplateMetadata;
}

export interface PromptOptimization {
  type: 'token_reduction' | 'quality_enhancement' | 'latency_optimization';
  description: string;
  impact: number;
  enabled: boolean;
}

export interface PromptConstraint {
  type: 'max_length' | 'format' | 'content_filter';
  value: any;
  enforcement: 'strict' | 'flexible';
}

export interface TemplateMetadata {
  version: string;
  author: string;
  lastUpdated: Date;
  usageCount: number;
  successRate: number;
}

export interface AdaptedPrompt {
  originalPrompt: string;
  adaptedPrompt: string;
  model: ModelCapability;
  template?: string;
  optimizations: string[];
  metrics: PromptMetrics;
  validation?: PromptValidation;
}

export interface PromptMetrics {
  originalLength: number;
  adaptedLength: number;
  originalTokens: number;
  adaptedTokens: number;
  compressionRatio: number;
  tokenSavings: number;
  estimatedCost: number;
  processingTime: number;
}

export interface PromptValidation {
  valid: boolean;
  tokenCount: number;
  issues: PromptIssue[];
  warnings: PromptIssue[];
  errors: PromptIssue[];
  suggestions: string[];
}

export interface PromptIssue {
  type: string;
  severity: 'error' | 'warning' | 'info';
  message: string;
  suggestion?: string;
  location?: {
    start: number;
    end: number;
  };
}

// ============================================================================
// SYSTEM CONFIGURATION TYPES
// ============================================================================

export interface RoutingConfig {
  costOptimizationMode: CostOptimizationMode;
  geographicFallbacks: GeographicFallbackChain[];
  costBudget: CostBudget;
  emergencyControls: EmergencyControl[];
  cacheEnabled: boolean;
  cacheTTL: number;
  monitoringEnabled: boolean;
  analyticsEnabled: boolean;
  maxRetries: number;
  timeout: number;
  compliance?: ComplianceConfig;
  performance?: PerformanceConfig;
  notificationSettings?: NotificationSettings;
}

export interface ComplianceConfig {
  enabledFrameworks: ComplianceFramework[];
  strictMode: boolean;
  auditLevel: 'basic' | 'detailed' | 'comprehensive';
  dataHandling: DataHandlingRules;
  geographicCompliance: GeographicComplianceRules;
  auditSettings: AuditSettings;
}

export interface DataHandlingRules {
  encryptionRequired: boolean;
  anonymizationRequired: boolean;
  retentionLimits: Record<DataClassification, number>;
  crossBorderTransferAllowed: boolean;
  dataLocalizationRequired: boolean;
}

export interface GeographicComplianceRules {
  restrictedRegions: GeographicRegion[];
  dataLocalizationRequired: boolean;
  sovereigntyRequirements: Record<GeographicRegion, ComplianceFramework[]>;
}

export interface AuditSettings {
  logRetention: number;
  realTimeMonitoring: boolean;
  alertOnViolation: boolean;
  complianceReporting: boolean;
}

export interface PerformanceConfig {
  targetLatency: number;
  maxConcurrentRequests: number;
  rateLimiting: RateLimitingConfig;
  circuitBreaker: CircuitBreakerConfig;
}

export interface RateLimitingConfig {
  enabled: boolean;
  requestsPerMinute: number;
  burstLimit: number;
  strategy: 'fixed_window' | 'sliding_window' | 'token_bucket';
}

export interface CircuitBreakerConfig {
  enabled: boolean;
  failureThreshold: number;
  recoveryTimeout: number;
  monitoringWindow: number;
}

export interface NotificationSettings {
  channels: Record<string, NotificationChannel>;
  alerts: Record<string, AlertConfig>;
}

export interface NotificationChannel {
  enabled: boolean;
  config: any;
  priority: 'low' | 'normal' | 'high' | 'critical';
}

export interface AlertConfig {
  enabled: boolean;
  thresholds: number[];
  channels: string[];
  cooldownPeriod: number;
}

// ============================================================================
// EMERGENCY CONTROL TYPES
// ============================================================================

export interface EmergencyControl {
  enabled: boolean;
  trigger: EmergencyTrigger;
  triggerCondition?: EmergencyTriggerCondition;
  action: EmergencyAction;
  cooldownPeriod: number;
  notificationChannels: string[];
  complianceOverride?: boolean;
  metadata?: EmergencyMetadata;
}

export enum EmergencyTrigger {
  COST_SPIKE = 'cost_spike',
  HIGH_ERROR_RATE = 'high_error_rate',
  LATENCY_SPIKE = 'latency_spike',
  SERVICE_DEGRADATION = 'service_degradation',
  COMPLIANCE_VIOLATION = 'compliance_violation',
  SECURITY_THREAT = 'security_threat',
}

export interface EmergencyTriggerCondition {
  threshold: number;
  timeWindow: number;
  comparison: 'gt' | 'gte' | 'lt' | 'lte' | 'eq';
  metric: string;
}

export enum EmergencyAction {
  FORCE_TIER_3 = 'force_tier_3',
  DISABLE_NON_CRITICAL = 'disable_non_critical',
  GEOGRAPHIC_RESTRICTION = 'geographic_restriction',
  RATE_LIMITING = 'rate_limiting',
  COMPLIANCE_LOCKDOWN = 'compliance_lockdown',
  CIRCUIT_BREAKER = 'circuit_breaker',
}

export interface EmergencyMetadata {
  severity: 'low' | 'medium' | 'high' | 'critical';
  estimatedImpact: string;
  recoveryProcedure: string[];
  stakeholders: string[];
}

// ============================================================================
// REQUEST AND RESPONSE TYPES
// ============================================================================

export interface RoutingContext {
  userId?: string;
  sessionId: string;
  requestId: string;
  timestamp: Date;
  clientRegion?: GeographicRegion;
  priority: 'low' | 'normal' | 'high' | 'critical';
  tags: string[];
  complianceContext?: ComplianceContext;
  userAgent?: string;
  ipAddress?: string;
}

export interface RoutingResult {
  success: boolean;
  response?: any;
  routingDecision: RoutingDecision;
  actualCost: number;
  actualLatency: number;
  error?: RoutingError;
  metadata: RoutingMetadata;
  attempts: number;
  complianceValidation?: ComplianceValidation;
}

export interface RoutingError {
  code: string;
  message: string;
  recoverable: boolean;
  retryAfter?: number;
  alternativeModels?: ModelCapability[];
  suggestedActions?: string[];
}

// ============================================================================
// METRICS AND ANALYTICS TYPES
// ============================================================================

export interface RoutingMetrics {
  totalRequests: number;
  successfulRoutes: number;
  failedRoutes: number;
  averageLatency: number;
  totalCost: number;
  costSavings: number;
  tierDistribution: Record<RoutingTier, number>;
  regionDistribution: Record<GeographicRegion, number>;
  errorRate: number;
  cacheHitRate: number;
  complianceScore: number;
  uptime: number;
}

export interface RoutingAnalytics {
  trackEvent(event: RoutingEvent): void;
  getMetrics(timeRange: string): RoutingMetrics;
  generateReport(timeRange: string): RoutingReport;
  predictCostOptimization(): CostOptimizationPrediction;
}

export interface RoutingEvent {
  type: RoutingEventType;
  context: RoutingContext;
  data: any;
  timestamp: Date;
  severity: 'info' | 'warning' | 'error' | 'critical';
  correlationId?: string;
  source: string;
  metadata?: EventMetadata;
}

export enum RoutingEventType {
  ROUTING_DECISION = 'routing_decision',
  COST_THRESHOLD_EXCEEDED = 'cost_threshold_exceeded',
  EMERGENCY_CONTROL_ACTIVATED = 'emergency_control_activated',
  FALLBACK_TRIGGERED = 'fallback_triggered',
  CACHE_MISS = 'cache_miss',
  CACHE_HIT = 'cache_hit',
  MODEL_FAILURE = 'model_failure',
  GEOGRAPHIC_SWITCH = 'geographic_switch',
  COMPLIANCE_VIOLATION = 'compliance_violation',
  HEALTH_CHECK_FAILED = 'health_check_failed',
  RATE_LIMIT_EXCEEDED = 'rate_limit_exceeded',
  CONFIGURATION_CHANGED = 'configuration_changed',
}

export interface EventMetadata {
  component: string;
  version: string;
  environment: string;
  processingTime?: number;
  retryCount?: number;
  userAgent?: string;
  ipAddress?: string;
}

export interface RoutingReport {
  period: string;
  metrics: RoutingMetrics;
  recommendations: string[];
  alerts: RoutingAlert[];
  costOptimization: CostOptimizationAnalysis;
  complianceReport: ComplianceReport;
}

export interface RoutingAlert {
  type: 'cost' | 'performance' | 'reliability' | 'security' | 'compliance';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  recommendation: string;
  threshold: number;
  actual: number;
  timestamp: Date;
}

export interface CostOptimizationAnalysis {
  currentSavings: number;
  potentialSavings: number;
  optimizationOpportunities: OptimizationOpportunity[];
  trend: 'improving' | 'stable' | 'degrading';
  recommendations: string[];
}

export interface ComplianceReport {
  overallScore: number;
  frameworkScores: Record<ComplianceFramework, number>;
  violations: ComplianceViolation[];
  recommendations: string[];
  auditTrail: AuditEntry[];
}

export interface ComplianceViolation {
  framework: ComplianceFramework;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  remediation: string;
  timestamp: Date;
}

export interface CostOptimizationPrediction {
  predictedSavings: number;
  confidence: number;
  timeHorizon: string;
  assumptions: string[];
  risks: string[];
  recommendations: string[];
}

// ============================================================================
// INTEGRATION INTERFACE TYPES
// ============================================================================

export interface AIEngineIntegration {
  generateText(prompt: string, options?: any): Promise<any>;
  getAvailableModels(): ModelCapability[];
  estimateCost(model: string, tokens: number): number;
  getModelCapabilities(model: string): ModelCapability | null;
  validateCompliance?(model: ModelCapability, context: ComplianceContext): ComplianceValidation;
  getHealthStatus?(): ProviderHealth;
  updateRateLimits?(model: string, limits: RateLimits): void;
}

export interface OrchestrationIntegration {
  coordinateModule(moduleId: string, action: string, context: any): Promise<any>;
  getAssetCoordinator(): any;
  registerRoutingAsset(routingAsset: any): void;
  notifyComplianceEvent?(event: ComplianceEvent): void;
  getModuleHealth?(moduleId: string): Promise<ModuleHealth>;
  emitOrchestrationEvent?(event: OrchestrationEvent): void;
}

export interface ComplianceEvent {
  type: ComplianceEventType;
  framework: ComplianceFramework;
  severity: 'info' | 'warning' | 'error' | 'critical';
  details: any;
  timestamp: Date;
  source: string;
  requestId?: string;
}

export enum ComplianceEventType {
  VIOLATION_DETECTED = 'violation_detected',
  COMPLIANCE_CHECK_PASSED = 'compliance_check_passed',
  AUDIT_LOG_UPDATED = 'audit_log_updated',
  DATA_HANDLING_VIOLATION = 'data_handling_violation',
  GEOGRAPHIC_COMPLIANCE_BREACH = 'geographic_compliance_breach',
  RETENTION_POLICY_VIOLATION = 'retention_policy_violation',
}

export interface ProviderHealth {
  provider: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  latency: number;
  errorRate: number;
  lastCheck: Date;
  services: ServiceHealth[];
}

export interface ServiceHealth {
  service: string;
  status: 'up' | 'down' | 'degraded';
  latency: number;
  errorCount: number;
}

export interface ModuleHealth {
  moduleId: string;
  status: 'up' | 'down' | 'degraded';
  lastActivity: Date;
  metrics: any;
  dependencies: string[];
}

export interface OrchestrationEvent {
  type: OrchestrationEventType;
  source: string;
  target?: string;
  data: any;
  timestamp: Date;
  correlationId?: string;
}

export enum OrchestrationEventType {
  MODULE_COORDINATED = 'module_coordinated',
  ASSET_REGISTERED = 'asset_registered',
  ASSET_UNREGISTERED = 'asset_unregistered',
  HEALTH_CHECK_FAILED = 'health_check_failed',
  COMPLIANCE_EVENT = 'compliance_event',
  COORDINATION_FAILED = 'coordination_failed',
}

// ============================================================================
// UTILITY TYPES
// ============================================================================

export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

export type Required<T, K extends keyof T> = T & Required<Pick<T, K>>;

export type Mutable<T> = {
  -readonly [P in keyof T]: T[P];
};

export type ConfigurationPreset =
  | 'quickStart'
  | 'production'
  | 'aggressiveSavings'
  | 'qualityFirst'
  | 'complianceFirst';

// ============================================================================
// DEFAULT VALUES AND CONSTANTS
// ============================================================================

export const DEFAULT_ROUTING_CONFIG: RoutingConfig = {
  costOptimizationMode: CostOptimizationMode.BALANCED,
  geographicFallbacks: [],
  costBudget: {
    dailyLimit: 100,
    monthlyLimit: 3000,
    perRequestLimit: 10,
    currentDailyUsage: 0,
    currentMonthlyUsage: 0,
    alertThreshold: 0.8,
  },
  emergencyControls: [],
  cacheEnabled: true,
  cacheTTL: 3600,
  monitoringEnabled: true,
  analyticsEnabled: true,
  maxRetries: 3,
  timeout: 30000,
};

export const DEFAULT_MODEL_CAPABILITY: ModelCapability = {
  name: 'Unknown Model',
  provider: 'unknown',
  model: 'unknown',
  tier: RoutingTier.TIER_3,
  region: GeographicRegion.GLOBAL,
  maxTokens: 4096,
  costPerToken: 0.0001,
  qualityScore: 50,
  latency: 1000,
  reliability: 80,
  specializations: [],
  limitations: [],
  compliance: [],
  enabled: false,
};

export const DEFAULT_ROUTING_CONTEXT: RoutingContext = {
  sessionId: 'unknown',
  requestId: 'unknown',
  timestamp: new Date(),
  priority: 'normal',
  tags: [],
};

// ============================================================================
// TYPE GUARDS AND VALIDATION
// ============================================================================

export function isValidTaskComplexity(value: string): value is TaskComplexity {
  return Object.values(TaskComplexity).includes(value as TaskComplexity);
}

export function isValidGeographicRegion(value: string): value is GeographicRegion {
  return Object.values(GeographicRegion).includes(value as GeographicRegion);
}

export function isValidRoutingTier(value: string): value is RoutingTier {
  return Object.values(RoutingTier).includes(value as RoutingTier);
}

export function isValidComplianceFramework(value: string): value is ComplianceFramework {
  return Object.values(ComplianceFramework).includes(value as ComplianceFramework);
}

export function isValidDataClassification(value: string): value is DataClassification {
  return Object.values(DataClassification).includes(value as DataClassification);
}

// ============================================================================
// EXPORT ALL TYPES
// ============================================================================

export default {
  TaskComplexity,
  GeographicRegion,
  RoutingTier,
  CostOptimizationMode,
  RiskLevel,
  DataClassification,
  ComplianceFramework,
  RoutingEventType,
  ComplianceEventType,
  OrchestrationEventType,
  EmergencyTrigger,
  EmergencyAction,
};
