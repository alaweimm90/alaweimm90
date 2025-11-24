/**
 * TRAE Fallback Manager Template
 * Geographic Fallback Chain Management with Intelligent Failover
 *
 * This template provides a complete fallback management system that implements
 * geographic failover strategies, health monitoring, and provider diversity
 * to ensure high availability and reliability.
 *
 * Features:
 * - Geographic failover (6 regions)
 * - Health monitoring with consecutive failure tracking
 * - Smart fallback chain optimization
 * - Provider diversity and redundancy
 * - Latency-aware routing
 * - Compliance-aware geographic selection
 * - Framework and language agnostic design
 */

// ============================================================================
// TYPE DEFINITIONS (Language Agnostic)
// ============================================================================

export interface GeographicFallbackChain {
  primary: GeographicRegion;
  fallbacks: GeographicRegion[];
  latencyThreshold: number;
  costMultiplier: number;
  compliancePriority: boolean;
}

export enum GeographicRegion {
  NORTH_AMERICA = 'north_america',
  EUROPE = 'europe',
  ASIA_PACIFIC = 'asia_pacific',
  SOUTH_AMERICA = 'south_america',
  AFRICA = 'africa',
  GLOBAL = 'global',
}

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
}

export enum RoutingTier {
  TIER_1 = 'tier_1',
  TIER_2 = 'tier_2',
  TIER_3 = 'tier_3',
}

export enum ComplianceFramework {
  HIPAA = 'hipaa',
  GDPR = 'gdpr',
  SOC2 = 'soc2',
  PCI_DSS = 'pci_dss',
}

export interface RoutingDecision {
  selectedModel: ModelCapability;
  fallbackChain: ModelCapability[];
  estimatedCost: number;
  estimatedLatency: number;
  confidence: number;
  reasoning: string[];
  costSavings: number;
  riskAssessment: RiskLevel;
  complianceValidation: ComplianceValidation;
}

export enum RiskLevel {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical',
}

export interface ComplianceValidation {
  framework: ComplianceFramework;
  validated: boolean;
  requirements: string[];
  violations: string[];
  complianceSurcharge?: number;
}

export interface RoutingResult {
  success: boolean;
  response?: any;
  routingDecision: RoutingDecision;
  actualCost: number;
  actualLatency: number;
  error?: RoutingError;
  metadata: any;
  attempts: number;
}

export interface RoutingError {
  code: string;
  message: string;
  recoverable: boolean;
  retryAfter?: number;
  alternativeModels?: ModelCapability[];
  complianceViolation?: boolean;
}

export interface RoutingContext {
  userId?: string;
  sessionId: string;
  requestId: string;
  timestamp: Date;
  clientRegion?: GeographicRegion;
  priority: RequestPriority;
  tags: string[];
  complianceContext?: ComplianceContext;
}

export enum RequestPriority {
  LOW = 'low',
  NORMAL = 'normal',
  HIGH = 'high',
  CRITICAL = 'critical',
}

export interface ComplianceContext {
  dataClassification: DataClassification;
  retentionPolicy: RetentionPolicy;
  auditRequirements: boolean;
  geographicRestrictions: GeographicRegion[];
}

export enum DataClassification {
  PUBLIC = 'public',
  INTERNAL = 'internal',
  CONFIDENTIAL = 'confidential',
  RESTRICTED = 'restricted',
}

export enum RetentionPolicy {
  EPHEMERAL = 'ephemeral',
  SHORT_TERM = 'short_term',
  LONG_TERM = 'long_term',
  PERMANENT = 'permanent',
}

export interface RoutingEvent {
  type: RoutingEventType;
  context: RoutingContext;
  data: any;
  timestamp: Date;
  severity: 'info' | 'warning' | 'error' | 'critical';
}

export enum RoutingEventType {
  GEOGRAPHIC_SWITCH = 'geographic_switch',
  FALLBACK_TRIGGERED = 'fallback_triggered',
  HEALTH_CHECK_FAILED = 'health_check_failed',
  PROVIDER_FAILOVER = 'provider_failover',
}

// ============================================================================
// FALLBACK MANAGER IMPLEMENTATION
// ============================================================================

export class FallbackManager {
  private fallbackChains: Map<GeographicRegion, GeographicFallbackChain> = new Map();
  private regionHealth: Map<GeographicRegion, RegionHealth> = new Map();
  private providerHealth: Map<string, ProviderHealth> = new Map();
  private eventListeners: Array<(event: RoutingEvent) => void> = [];

  // Health check intervals
  private readonly HEALTH_CHECK_INTERVAL = 30000; // 30 seconds
  private readonly HEALTH_TIMEOUT = 5000; // 5 seconds
  private readonly FAILURE_THRESHOLD = 3; // Consecutive failures before marking unhealthy
  private readonly RECOVERY_ATTEMPTS = 2; // Attempts before considering recovered

  constructor(fallbackChains: GeographicFallbackChain[]) {
    this.initializeFallbackChains(fallbackChains);
    this.initializeHealthMonitoring();
    this.startHealthChecks();
  }

  /**
   * Execute request with geographic fallback
   */
  async executeWithFallback(
    decision: RoutingDecision,
    context: RoutingContext,
    executeRequest: (model: ModelCapability) => Promise<RoutingResult>
  ): Promise<RoutingResult> {
    const attempts: ModelAttempt[] = [];
    let lastError: RoutingError | null = null;

    // Get fallback chain for the primary region
    const fallbackChain = this.getFallbackChain(decision.selectedModel.region);

    // Try primary model first
    try {
      const result = await this.tryModel(decision.selectedModel, executeRequest, context);
      attempts.push(result.attempt);

      if (result.success) {
        this.recordSuccess(decision.selectedModel);
        return result.result!;
      }

      lastError = result.error!;
      attempts.push(result.attempt);
    } catch (error) {
      lastError = this.createRoutingError(error as Error, decision.selectedModel);
      attempts.push(this.createFailedAttempt(decision.selectedModel, error as Error));
    }

    // Try fallback models in geographic order
    for (const fallbackRegion of fallbackChain.fallbacks) {
      if (!this.isRegionHealthy(fallbackRegion)) {
        this.emitEvent({
          type: RoutingEventType.GEOGRAPHIC_SWITCH,
          context: {
            sessionId: context.sessionId,
            requestId: context.requestId,
            timestamp: new Date(),
            priority: RequestPriority.NORMAL,
            tags: [
              'geographic-switch',
              `from:${decision.selectedModel.region}`,
              `to:${fallbackRegion}`,
              'reason:unhealthy_region',
            ],
          },
          data: { fallbackRegion, healthStatus: this.regionHealth.get(fallbackRegion) },
          timestamp: new Date(),
          severity: 'warning',
        });
        continue;
      }

      const fallbackModels = this.findModelsInRegion(fallbackRegion, decision.fallbackChain);

      for (const fallbackModel of fallbackModels) {
        try {
          const result = await this.tryModel(fallbackModel, executeRequest, context);
          attempts.push(result.attempt);

          if (result.success) {
            this.recordFallbackSuccess(
              fallbackModel,
              decision.selectedModel.region,
              fallbackRegion
            );
            return result.result!;
          }

          lastError = result.error!;
        } catch (error) {
          lastError = this.createRoutingError(error as Error, fallbackModel);
          attempts.push(this.createFailedAttempt(fallbackModel, error as Error));
        }
      }
    }

    // All fallbacks failed
    this.emitEvent({
      type: RoutingEventType.FALLBACK_TRIGGERED,
      context: {
        sessionId: context.sessionId,
        requestId: context.requestId,
        timestamp: new Date(),
        priority: RequestPriority.HIGH,
        tags: [
          'fallback-failed',
          `attempts:${attempts.length}`,
          ...(lastError?.message ? [`error:${lastError.message}`] : []),
        ],
      },
      data: { attempts, lastError },
      timestamp: new Date(),
      severity: 'error',
    });

    throw lastError || new Error('All fallback options exhausted');
  }

  /**
   * Get optimized fallback chain for a region
   */
  getFallbackChain(primaryRegion: GeographicRegion): GeographicFallbackChain {
    const chain = this.fallbackChains.get(primaryRegion);
    if (!chain) {
      // Return default chain if not configured
      return {
        primary: primaryRegion,
        fallbacks: this.getDefaultFallbacks(primaryRegion),
        latencyThreshold: 1000,
        costMultiplier: 1.5,
        compliancePriority: false,
      };
    }

    // Optimize chain based on current health
    return this.optimizeFallbackChain(chain);
  }

  /**
   * Update region health status
   */
  updateRegionHealth(region: GeographicRegion, healthy: boolean, latency?: number): void {
    const health = this.regionHealth.get(region) || {
      region,
      healthy: true,
      latency: 0,
      lastCheck: new Date(),
      consecutiveFailures: 0,
      totalRequests: 0,
      successfulRequests: 0,
    };

    health.lastCheck = new Date();

    if (healthy) {
      health.healthy = true;
      health.consecutiveFailures = 0;
      health.successfulRequests++;
      if (latency !== undefined) {
        health.latency = latency;
      }
    } else {
      health.consecutiveFailures++;
      if (health.consecutiveFailures >= this.FAILURE_THRESHOLD) {
        health.healthy = false;
      }
    }

    health.totalRequests++;
    this.regionHealth.set(region, health);
  }

  /**
   * Get health status for all regions
   */
  getRegionHealthStatus(): Map<GeographicRegion, RegionHealth> {
    return new Map(this.regionHealth);
  }

  /**
   * Force failover to specific region
   */
  forceFailover(fromRegion: GeographicRegion, toRegion: GeographicRegion): void {
    this.emitEvent({
      type: RoutingEventType.GEOGRAPHIC_SWITCH,
      context: {
        sessionId: 'system',
        requestId: `failover_${Date.now()}`,
        timestamp: new Date(),
        priority: RequestPriority.HIGH,
        tags: ['forced-failover', `from:${fromRegion}`, `to:${toRegion}`],
      },
      data: { forced: true },
      timestamp: new Date(),
      severity: 'warning',
    });

    // Mark source region as unhealthy
    this.updateRegionHealth(fromRegion, false);
  }

  /**
   * Add event listener for fallback events
   */
  addEventListener(listener: (event: RoutingEvent) => void): void {
    this.eventListeners.push(listener);
  }

  // Private methods

  private initializeFallbackChains(configuredChains: GeographicFallbackChain[]): void {
    // Set up configured chains
    configuredChains.forEach(chain => {
      this.fallbackChains.set(chain.primary, chain);
    });

    // Ensure all regions have fallback chains
    Object.values(GeographicRegion).forEach(region => {
      if (!this.fallbackChains.has(region)) {
        this.fallbackChains.set(region, {
          primary: region,
          fallbacks: this.getDefaultFallbacks(region),
          latencyThreshold: 1000,
          costMultiplier: 1.5,
          compliancePriority: false,
        });
      }
    });
  }

  private getDefaultFallbacks(primaryRegion: GeographicRegion): GeographicRegion[] {
    const fallbackMap: Record<GeographicRegion, GeographicRegion[]> = {
      [GeographicRegion.NORTH_AMERICA]: [GeographicRegion.EUROPE, GeographicRegion.ASIA_PACIFIC],
      [GeographicRegion.EUROPE]: [GeographicRegion.NORTH_AMERICA, GeographicRegion.ASIA_PACIFIC],
      [GeographicRegion.ASIA_PACIFIC]: [GeographicRegion.NORTH_AMERICA, GeographicRegion.EUROPE],
      [GeographicRegion.SOUTH_AMERICA]: [GeographicRegion.NORTH_AMERICA, GeographicRegion.EUROPE],
      [GeographicRegion.AFRICA]: [GeographicRegion.EUROPE, GeographicRegion.ASIA_PACIFIC],
      [GeographicRegion.GLOBAL]: [
        GeographicRegion.NORTH_AMERICA,
        GeographicRegion.EUROPE,
        GeographicRegion.ASIA_PACIFIC,
      ],
    };

    return fallbackMap[primaryRegion] || [GeographicRegion.GLOBAL];
  }

  private initializeHealthMonitoring(): void {
    // Initialize health status for all regions
    Object.values(GeographicRegion).forEach(region => {
      this.regionHealth.set(region, {
        region,
        healthy: true,
        latency: 0,
        lastCheck: new Date(),
        consecutiveFailures: 0,
        totalRequests: 0,
        successfulRequests: 0,
      });
    });
  }

  private startHealthChecks(): void {
    setInterval(() => {
      this.performHealthChecks();
    }, this.HEALTH_CHECK_INTERVAL);
  }

  private async performHealthChecks(): Promise<void> {
    const healthCheckPromises = Array.from(this.regionHealth.keys()).map(async region => {
      try {
        const latency = await this.checkRegionLatency(region);
        this.updateRegionHealth(region, true, latency);
      } catch (error) {
        this.updateRegionHealth(region, false);
      }
    });

    await Promise.allSettled(healthCheckPromises);
  }

  private async checkRegionLatency(region: GeographicRegion): Promise<number> {
    // Simplified latency check - in production would ping regional endpoints
    const baseLatencies: Record<GeographicRegion, number> = {
      [GeographicRegion.NORTH_AMERICA]: 50,
      [GeographicRegion.EUROPE]: 100,
      [GeographicRegion.ASIA_PACIFIC]: 200,
      [GeographicRegion.SOUTH_AMERICA]: 150,
      [GeographicRegion.AFRICA]: 250,
      [GeographicRegion.GLOBAL]: 100,
    };

    // Add some jitter to simulate real network conditions
    const jitter = Math.random() * 20 - 10; // ±10ms
    return Math.max(0, baseLatencies[region] + jitter);
  }

  private isRegionHealthy(region: GeographicRegion): boolean {
    const health = this.regionHealth.get(region);
    return health?.healthy ?? false;
  }

  private findModelsInRegion(
    region: GeographicRegion,
    fallbackModels: ModelCapability[]
  ): ModelCapability[] {
    return fallbackModels.filter(
      model => model.region === region || model.region === GeographicRegion.GLOBAL
    );
  }

  private async tryModel(
    model: ModelCapability,
    executeRequest: (model: ModelCapability) => Promise<RoutingResult>,
    context: RoutingContext
  ): Promise<{
    success: boolean;
    result?: RoutingResult;
    error?: RoutingError;
    attempt: ModelAttempt;
  }> {
    const startTime = Date.now();

    try {
      const result = await executeRequest(model);
      const latency = Date.now() - startTime;

      return {
        success: true,
        result,
        attempt: {
          model,
          success: true,
          latency,
          cost: result.actualCost,
          timestamp: new Date(),
        },
      };
    } catch (error) {
      const latency = Date.now() - startTime;

      return {
        success: false,
        error: this.createRoutingError(error as Error, model),
        attempt: {
          model,
          success: false,
          latency,
          cost: 0, // No cost incurred on failure
          error: (error as Error).message,
          timestamp: new Date(),
        },
      };
    }
  }

  private createRoutingError(error: Error, model: ModelCapability): RoutingError {
    return {
      code: this.categorizeError(error),
      message: error.message,
      recoverable: this.isRecoverableError(error),
      retryAfter: this.getRetryAfter(error),
      alternativeModels: this.findAlternativeModels(model),
    };
  }

  private categorizeError(error: Error): string {
    const message = error.message.toLowerCase();

    if (message.includes('timeout') || message.includes('network')) {
      return 'NETWORK_ERROR';
    }
    if (message.includes('rate limit') || message.includes('quota')) {
      return 'RATE_LIMIT_ERROR';
    }
    if (message.includes('authentication') || message.includes('unauthorized')) {
      return 'AUTH_ERROR';
    }
    if (message.includes('server') || message.includes('internal')) {
      return 'SERVER_ERROR';
    }

    return 'UNKNOWN_ERROR';
  }

  private isRecoverableError(error: Error): boolean {
    const recoverableCodes = ['NETWORK_ERROR', 'RATE_LIMIT_ERROR', 'SERVER_ERROR'];
    return recoverableCodes.includes(this.categorizeError(error));
  }

  private getRetryAfter(error: Error): number {
    const errorCode = this.categorizeError(error);

    switch (errorCode) {
      case 'RATE_LIMIT_ERROR':
        return 60; // 1 minute
      case 'SERVER_ERROR':
        return 30; // 30 seconds
      case 'NETWORK_ERROR':
        return 10; // 10 seconds
      default:
        return 15; // 15 seconds default
    }
  }

  private findAlternativeModels(failedModel: ModelCapability): ModelCapability[] {
    // Find models with similar capabilities but different providers/regions
    return this.getAvailableModels().filter(
      model =>
        model.name !== failedModel.name &&
        model.tier === failedModel.tier &&
        model.specializations.some(cap => failedModel.specializations.includes(cap))
    );
  }

  private createFailedAttempt(model: ModelCapability, error: Error): ModelAttempt {
    return {
      model,
      success: false,
      latency: 0,
      cost: 0,
      error: error.message,
      timestamp: new Date(),
    };
  }

  private recordSuccess(model: ModelCapability): void {
    this.updateRegionHealth(model.region, true);
    this.updateProviderHealth(model.provider, true);
  }

  private recordFallbackSuccess(
    fallbackModel: ModelCapability,
    originalRegion: GeographicRegion,
    fallbackRegion: GeographicRegion
  ): void {
    this.updateRegionHealth(fallbackRegion, true);
    this.updateProviderHealth(fallbackModel.provider, true);

    this.emitEvent({
      type: RoutingEventType.GEOGRAPHIC_SWITCH,
      context: {
        sessionId: 'system',
        requestId: `fallback_success_${Date.now()}`,
        timestamp: new Date(),
        priority: RequestPriority.NORMAL,
        tags: [
          'successful-fallback',
          `from:${originalRegion}`,
          `to:${fallbackRegion}`,
          `model:${fallbackModel.name}`,
        ],
      },
      data: { fallbackModel, costSavings: 0 }, // Would calculate actual savings
      timestamp: new Date(),
      severity: 'info',
    });
  }

  private updateProviderHealth(provider: string, healthy: boolean): void {
    const health = this.providerHealth.get(provider) || {
      provider,
      healthy: true,
      lastCheck: new Date(),
      consecutiveFailures: 0,
      totalRequests: 0,
      successfulRequests: 0,
    };

    health.lastCheck = new Date();

    if (healthy) {
      health.healthy = true;
      health.consecutiveFailures = 0;
      health.successfulRequests++;
    } else {
      health.consecutiveFailures++;
      if (health.consecutiveFailures >= this.FAILURE_THRESHOLD) {
        health.healthy = false;
      }
    }

    health.totalRequests++;
    this.providerHealth.set(provider, health);
  }

  private optimizeFallbackChain(chain: GeographicFallbackChain): GeographicFallbackChain {
    // Sort fallbacks by health and latency
    const optimizedFallbacks = chain.fallbacks.sort((a, b) => {
      const healthA = this.regionHealth.get(a);
      const healthB = this.regionHealth.get(b);

      // Prioritize healthy regions
      if (healthA?.healthy !== healthB?.healthy) {
        return healthA?.healthy ? -1 : 1;
      }

      // Then by latency
      return (healthA?.latency || 1000) - (healthB?.latency || 1000);
    });

    return {
      ...chain,
      fallbacks: optimizedFallbacks,
    };
  }

  private getAvailableModels(): ModelCapability[] {
    // Mock implementation - in production would be injected
    return [
      {
        name: 'gpt-4',
        provider: 'openai',
        model: 'gpt-4',
        tier: RoutingTier.TIER_1,
        region: GeographicRegion.NORTH_AMERICA,
        maxTokens: 8192,
        costPerToken: 0.03,
        qualityScore: 95,
        latency: 2000,
        reliability: 98,
        specializations: ['advanced_reasoning'],
        limitations: ['high_cost'],
        compliance: [ComplianceFramework.GDPR, ComplianceFramework.SOC2],
      },
    ];
  }

  private emitEvent(event: RoutingEvent): void {
    this.eventListeners.forEach(listener => {
      try {
        listener(event);
      } catch (error) {
        console.error('Error in fallback manager event listener:', error);
      }
    });
  }
}

// ============================================================================
// SUPPORTING INTERFACES AND TYPES
// ============================================================================

export interface RegionHealth {
  region: GeographicRegion;
  healthy: boolean;
  latency: number;
  lastCheck: Date;
  consecutiveFailures: number;
  totalRequests: number;
  successfulRequests: number;
}

export interface ProviderHealth {
  provider: string;
  healthy: boolean;
  lastCheck: Date;
  consecutiveFailures: number;
  totalRequests: number;
  successfulRequests: number;
}

export interface ModelAttempt {
  model: ModelCapability;
  success: boolean;
  latency: number;
  cost: number;
  error?: string;
  timestamp: Date;
}

// ============================================================================
// USAGE EXAMPLES AND INTEGRATION PATTERNS
// ============================================================================

/**
 * Example: Basic Geographic Fallback
 *
 * const manager = new FallbackManager(fallbackChains);
 * const result = await manager.executeWithFallback(
 *   routingDecision,
 *   context,
 *   async (model) => await aiEngine.generateText(prompt, { model })
 * );
 */

/**
 * Example: Health Monitoring
 *
 * const healthStatus = manager.getRegionHealthStatus();
 * console.log('Region health:', healthStatus);
 *
 * // Force failover if needed
 * manager.forceFailover(GeographicRegion.NORTH_AMERICA, GeographicRegion.EUROPE);
 */

/**
 * Example: Custom Fallback Chains
 *
 * const customChains = [
 *   {
 *     primary: GeographicRegion.NORTH_AMERICA,
 *     fallbacks: [GeographicRegion.EUROPE, GeographicRegion.ASIA_PACIFIC],
 *     latencyThreshold: 1500,
 *     costMultiplier: 1.3,
 *     compliancePriority: true
 *   }
 * ];
 *
 * const manager = new FallbackManager(customChains);
 */

/**
 * Example: Event Monitoring
 *
 * manager.addEventListener((event) => {
 *   switch (event.type) {
 *     case RoutingEventType.GEOGRAPHIC_SWITCH:
 *       console.log('Geographic switch:', event.data);
 *       break;
 *     case RoutingEventType.FALLBACK_TRIGGERED:
 *       console.warn('Fallback triggered:', event.data);
 *       break;
 *   }
 * });
 */

export default FallbackManager;
