/**
 * TRAE Orchestration Adapter Template
 * Framework-Agnostic Orchestration Integration Layer
 *
 * This template provides a complete adapter pattern for integrating with any
 * orchestration system or asset coordinator, enabling seamless coordination
 * between routing components and the broader system architecture.
 *
 * Features:
 * - Asset coordinator integration
 * - Module coordination patterns
 * - Event system integration
 * - Compliance event handling
 * - Framework and language agnostic
 * - Production monitoring and error handling
 */

// ============================================================================
// TYPE DEFINITIONS (Framework Agnostic)
// ============================================================================

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

export enum ComplianceFramework {
  HIPAA = 'hipaa',
  GDPR = 'gdpr',
  SOC2 = 'soc2',
  PCI_DSS = 'pci_dss',
}

export interface ModuleCoordinationRequest {
  moduleId: string;
  action: string;
  context: any;
  priority: CoordinationPriority;
  timeout?: number;
  dependencies?: string[];
}

export enum CoordinationPriority {
  LOW = 'low',
  NORMAL = 'normal',
  HIGH = 'high',
  CRITICAL = 'critical',
}

export interface CoordinationResult {
  success: boolean;
  result?: any;
  error?: CoordinationError;
  metadata: CoordinationMetadata;
}

export interface CoordinationError {
  code: string;
  message: string;
  recoverable: boolean;
  retryAfter?: number;
  alternativeActions?: string[];
}

export interface CoordinationMetadata {
  duration: number;
  coordinatedModules: string[];
  eventsEmitted: number;
  complianceChecks: number;
}

export interface AssetRegistration {
  assetId: string;
  assetType: string;
  capabilities: string[];
  metadata: any;
  healthCheck?: () => Promise<AssetHealth>;
}

export interface AssetHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  lastCheck: Date;
  metrics: any;
  issues: string[];
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
// DEPENDENCY INTERFACES (Framework Agnostic)
// ============================================================================

export interface OrchestrationIntegration {
  coordinateModule(moduleId: string, action: string, context: any): Promise<any>;
  getAssetCoordinator(): any;
  registerRoutingAsset(routingAsset: any): void;
  notifyComplianceEvent(event: ComplianceEvent): void;
  getModuleHealth(moduleId: string): Promise<ModuleHealth>;
  emitOrchestrationEvent(event: OrchestrationEvent): void;
}

export interface ModuleHealth {
  moduleId: string;
  status: 'up' | 'down' | 'degraded';
  lastActivity: Date;
  metrics: any;
  dependencies: string[];
}

// ============================================================================
// ORCHESTRATION ADAPTER IMPLEMENTATION
// ============================================================================

export class OrchestrationAdapter implements OrchestrationIntegration {
  private assetCoordinator: any = null; // Framework-specific asset coordinator
  private routingAsset: any = null;
  private eventListeners: Array<(event: OrchestrationEvent) => void> = [];
  private complianceListeners: Array<(event: ComplianceEvent) => void> = [];
  private registeredAssets: Map<string, AssetRegistration> = new Map();
  private coordinationHistory: CoordinationRecord[] = [];

  private healthCheckInterval?: NodeJS.Timeout;
  private enableHealthMonitoring: boolean;
  private enableEventForwarding: boolean;

  constructor(options?: {
    assetCoordinator?: any;
    enableHealthMonitoring?: boolean;
    enableEventForwarding?: boolean;
    healthCheckInterval?: number;
  }) {
    this.assetCoordinator = options?.assetCoordinator;
    this.enableHealthMonitoring = options?.enableHealthMonitoring !== false;
    this.enableEventForwarding = options?.enableEventForwarding !== false;

    if (this.enableHealthMonitoring) {
      this.startHealthMonitoring(options?.healthCheckInterval || 30000);
    }
  }

  /**
   * Coordinate module action through orchestration
   */
  async coordinateModule(moduleId: string, action: string, context: any): Promise<any> {
    const startTime = Date.now();
    const coordinationId = this.generateCoordinationId();

    try {
      // Create coordination request
      const request: ModuleCoordinationRequest = {
        moduleId,
        action,
        context: {
          ...context,
          coordinationId,
          timestamp: new Date(),
        },
        priority: this.determinePriority(context),
        timeout: context.timeout || 30000,
        dependencies: context.dependencies || [],
      };

      // Log coordination start
      this.logCoordinationStart(request);

      let result: any;

      if (this.assetCoordinator) {
        // Use framework-specific coordination
        result = await this.assetCoordinator.coordinateModule(
          request.moduleId,
          request.action,
          request.context
        );
      } else {
        // Fallback coordination logic
        result = await this.fallbackCoordination(request);
      }

      // Record successful coordination
      const duration = Date.now() - startTime;
      this.recordCoordinationSuccess(request, result, duration);

      // Emit orchestration event
      this.emitOrchestrationEvent({
        type: OrchestrationEventType.MODULE_COORDINATED,
        source: 'routing_adapter',
        target: moduleId,
        data: { request, result, duration },
        timestamp: new Date(),
        correlationId: coordinationId,
      });

      return result;
    } catch (error) {
      const duration = Date.now() - startTime;

      // Record failed coordination
      this.recordCoordinationFailure(moduleId, action, error as Error, duration);

      // Emit failure event
      this.emitOrchestrationEvent({
        type: OrchestrationEventType.COORDINATION_FAILED,
        source: 'routing_adapter',
        target: moduleId,
        data: { action, error: (error as Error).message, duration },
        timestamp: new Date(),
        correlationId: coordinationId,
      });

      throw error;
    }
  }

  /**
   * Get asset coordinator instance
   */
  getAssetCoordinator(): any {
    return this.assetCoordinator;
  }

  /**
   * Register routing asset with orchestration
   */
  registerRoutingAsset(routingAsset: any): void {
    this.routingAsset = routingAsset;

    // Register with asset coordinator if available
    if (this.assetCoordinator) {
      console.log('✅ Routing asset registered with orchestration');

      // Store routing capabilities for orchestration access
      (this.assetCoordinator as any).routingAsset = routingAsset;

      // Register asset for health monitoring
      this.registerAsset({
        assetId: 'routing_system',
        assetType: 'routing',
        capabilities: [
          'route_request',
          'get_metrics',
          'update_config',
          'activate_emergency_mode',
          'force_geographic_failover',
          'get_region_health',
        ],
        metadata: {
          version: '1.0.0',
          provider: 'trae',
          description: 'Intelligent LLM routing system',
        },
        healthCheck: async () => {
          try {
            const metrics = await routingAsset.getMetrics();
            return {
              status: 'healthy',
              lastCheck: new Date(),
              metrics,
              issues: [],
            };
          } catch (error) {
            return {
              status: 'unhealthy',
              lastCheck: new Date(),
              metrics: {},
              issues: [(error as Error).message],
            };
          }
        },
      });
    } else {
      console.log('⚠️  Asset coordinator not available, running in standalone mode');
    }
  }

  /**
   * Notify orchestration of compliance events
   */
  notifyComplianceEvent(event: ComplianceEvent): void {
    // Forward to compliance listeners
    this.complianceListeners.forEach(listener => {
      try {
        listener(event);
      } catch (error) {
        console.error('Error in compliance event listener:', error);
      }
    });

    // Emit orchestration event
    if (this.enableEventForwarding) {
      this.emitOrchestrationEvent({
        type: OrchestrationEventType.COMPLIANCE_EVENT,
        source: 'routing_adapter',
        data: event,
        timestamp: new Date(),
        correlationId: event.requestId,
      });
    }

    // Log compliance event
    this.logComplianceEvent(event);
  }

  /**
   * Get module health status
   */
  async getModuleHealth(moduleId: string): Promise<ModuleHealth> {
    if (this.assetCoordinator && typeof this.assetCoordinator.getModuleHealth === 'function') {
      return await this.assetCoordinator.getModuleHealth(moduleId);
    }

    // Fallback health check
    const asset = this.registeredAssets.get(moduleId);
    if (asset && asset.healthCheck) {
      const health = await asset.healthCheck();
      return {
        moduleId,
        status:
          health.status === 'healthy' ? 'up' : health.status === 'degraded' ? 'degraded' : 'down',
        lastActivity: health.lastCheck,
        metrics: health.metrics,
        dependencies: [],
      };
    }

    // Default response
    return {
      moduleId,
      status: 'up',
      lastActivity: new Date(),
      metrics: {},
      dependencies: [],
    };
  }

  /**
   * Emit orchestration event
   */
  emitOrchestrationEvent(event: OrchestrationEvent): void {
    // Forward to event listeners
    this.eventListeners.forEach(listener => {
      try {
        listener(event);
      } catch (error) {
        console.error('Error in orchestration event listener:', error);
      }
    });

    // Forward to asset coordinator if available
    if (this.assetCoordinator && typeof this.assetCoordinator.emitEvent === 'function') {
      this.assetCoordinator.emitEvent(event);
    }
  }

  /**
   * Add orchestration event listener
   */
  addEventListener(listener: (event: OrchestrationEvent) => void): void {
    this.eventListeners.push(listener);
  }

  /**
   * Add compliance event listener
   */
  addComplianceListener(listener: (event: ComplianceEvent) => void): void {
    this.complianceListeners.push(listener);
  }

  /**
   * Register asset for monitoring
   */
  registerAsset(registration: AssetRegistration): void {
    this.registeredAssets.set(registration.assetId, registration);

    this.emitOrchestrationEvent({
      type: OrchestrationEventType.ASSET_REGISTERED,
      source: 'orchestration_adapter',
      data: registration,
      timestamp: new Date(),
    });
  }

  /**
   * Unregister asset
   */
  unregisterAsset(assetId: string): void {
    this.registeredAssets.delete(assetId);

    this.emitOrchestrationEvent({
      type: OrchestrationEventType.ASSET_UNREGISTERED,
      source: 'orchestration_adapter',
      data: { assetId },
      timestamp: new Date(),
    });
  }

  /**
   * Get registered assets
   */
  getRegisteredAssets(): AssetRegistration[] {
    return Array.from(this.registeredAssets.values());
  }

  /**
   * Get coordination history
   */
  getCoordinationHistory(limit: number = 100): CoordinationRecord[] {
    return this.coordinationHistory.slice(-limit);
  }

  /**
   * Set asset coordinator (for late binding)
   */
  setAssetCoordinator(assetCoordinator: any): void {
    this.assetCoordinator = assetCoordinator;

    // Re-register routing asset if it was registered before coordinator was set
    if (this.routingAsset) {
      this.registerRoutingAsset(this.routingAsset);
    }
  }

  /**
   * Check if orchestration is available
   */
  isAvailable(): boolean {
    return this.assetCoordinator !== null;
  }

  // Private methods

  private async fallbackCoordination(request: ModuleCoordinationRequest): Promise<any> {
    // Simple fallback coordination for standalone mode
    console.log(
      `🔄 Coordinating module ${request.moduleId} action ${request.action} (standalone mode)`
    );

    // Simulate coordination delay
    await new Promise(resolve => setTimeout(resolve, 100));

    // Return mock result based on action
    switch (request.action) {
      case 'get_health':
        return { status: 'healthy', timestamp: new Date() };
      case 'get_metrics':
        return { requestCount: 0, successRate: 1.0 };
      case 'update_config':
        return { success: true, updated: Object.keys(request.context) };
      default:
        return { success: true, mode: 'standalone', action: request.action };
    }
  }

  private determinePriority(context: any): CoordinationPriority {
    if (context.priority === 'critical' || context.urgency === 'high') {
      return CoordinationPriority.CRITICAL;
    }
    if (context.priority === 'high') {
      return CoordinationPriority.HIGH;
    }
    if (context.priority === 'low') {
      return CoordinationPriority.LOW;
    }
    return CoordinationPriority.NORMAL;
  }

  private generateCoordinationId(): string {
    return `coord_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private logCoordinationStart(request: ModuleCoordinationRequest): void {
    console.log(
      `🔄 Starting coordination: ${request.moduleId}.${request.action} (${request.priority})`
    );
  }

  private recordCoordinationSuccess(
    request: ModuleCoordinationRequest,
    result: any,
    duration: number
  ): void {
    const record: CoordinationRecord = {
      id: request.context.coordinationId,
      moduleId: request.moduleId,
      action: request.action,
      success: true,
      duration,
      timestamp: new Date(),
      result:
        typeof result === 'object' ? JSON.stringify(result).substring(0, 200) : String(result),
    };

    this.coordinationHistory.push(record);

    // Keep only last 1000 records
    if (this.coordinationHistory.length > 1000) {
      this.coordinationHistory.shift();
    }
  }

  private recordCoordinationFailure(
    moduleId: string,
    action: string,
    error: Error,
    duration: number
  ): void {
    const record: CoordinationRecord = {
      id: this.generateCoordinationId(),
      moduleId,
      action,
      success: false,
      duration,
      timestamp: new Date(),
      error: error.message,
    };

    this.coordinationHistory.push(record);

    if (this.coordinationHistory.length > 1000) {
      this.coordinationHistory.shift();
    }
  }

  private logComplianceEvent(event: ComplianceEvent): void {
    const level =
      event.severity === 'critical'
        ? 'error'
        : event.severity === 'error'
          ? 'error'
          : event.severity === 'warning'
            ? 'warn'
            : 'info';

    console[level](
      `🔒 Compliance event [${event.framework}]: ${event.type} - ${
        event.details?.message || 'No details'
      }`
    );
  }

  private startHealthMonitoring(interval: number): void {
    this.healthCheckInterval = setInterval(async () => {
      await this.performHealthChecks();
    }, interval);
  }

  private async performHealthChecks(): Promise<void> {
    const healthCheckPromises = Array.from(this.registeredAssets.values())
      .filter(asset => asset.healthCheck)
      .map(async asset => {
        try {
          const health = await asset.healthCheck!();
          if (health.status !== 'healthy') {
            this.emitOrchestrationEvent({
              type: OrchestrationEventType.HEALTH_CHECK_FAILED,
              source: 'orchestration_adapter',
              data: { assetId: asset.assetId, health },
              timestamp: new Date(),
            });
          }
        } catch (error) {
          this.emitOrchestrationEvent({
            type: OrchestrationEventType.HEALTH_CHECK_FAILED,
            source: 'orchestration_adapter',
            data: {
              assetId: asset.assetId,
              health: {
                status: 'unhealthy',
                lastCheck: new Date(),
                metrics: {},
                issues: [(error as Error).message],
              },
            },
            timestamp: new Date(),
          });
        }
      });

    await Promise.allSettled(healthCheckPromises);
  }

  /**
   * Clean up resources
   */
  destroy(): void {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
    }
    this.eventListeners = [];
    this.complianceListeners = [];
    this.registeredAssets.clear();
  }
}

// ============================================================================
// SUPPORTING INTERFACES AND TYPES
// ============================================================================

export interface CoordinationRecord {
  id: string;
  moduleId: string;
  action: string;
  success: boolean;
  duration: number;
  timestamp: Date;
  result?: string;
  error?: string;
}

// ============================================================================
// USAGE EXAMPLES AND INTEGRATION PATTERNS
// ============================================================================

/**
 * Example: Basic Orchestration Setup
 *
 * const adapter = new OrchestrationAdapter({
 *   assetCoordinator: myAssetCoordinator,
 *   enableHealthMonitoring: true,
 *   enableEventForwarding: true
 * });
 *
 * // Register routing system
 * adapter.registerRoutingAsset(routingInstance);
 */

/**
 * Example: Module Coordination
 *
 * const result = await adapter.coordinateModule('cache', 'invalidate', {
 *   pattern: 'routing:*',
 *   priority: 'high'
 * });
 */

/**
 * Example: Compliance Event Handling
 *
 * adapter.addComplianceListener((event) => {
 *   if (event.type === ComplianceEventType.VIOLATION_DETECTED) {
 *     console.error('Compliance violation:', event);
 *     // Trigger remediation actions
 *   }
 * });
 */

/**
 * Example: Asset Registration
 *
 * adapter.registerAsset({
 *   assetId: 'model_registry',
 *   assetType: 'data_store',
 *   capabilities: ['read', 'write', 'query'],
 *   metadata: { provider: 'redis', version: '7.0' },
 *   healthCheck: async () => {
 *     const isConnected = await checkRedisConnection();
 *     return {
 *       status: isConnected ? 'healthy' : 'unhealthy',
 *       lastCheck: new Date(),
 *       metrics: { connections: 10 },
 *       issues: isConnected ? [] : ['Connection failed']
 *     };
 *   }
 * });
 */

/**
 * Example: Orchestration Events
 *
 * adapter.addEventListener((event) => {
 *   switch (event.type) {
 *     case OrchestrationEventType.MODULE_COORDINATED:
 *       console.log('Module coordinated:', event.data);
 *       break;
 *     case OrchestrationEventType.HEALTH_CHECK_FAILED:
 *       console.warn('Health check failed:', event.data);
 *       break;
 *   }
 * });
 */

/**
 * Example: Health Monitoring
 *
 * const health = await adapter.getModuleHealth('routing_system');
 * console.log(`Routing system status: ${health.status}`);
 * console.log(`Last activity: ${health.lastActivity}`);
 */

/**
 * Example: Coordination History
 *
 * const history = adapter.getCoordinationHistory(50);
 * history.forEach(record => {
 *   console.log(`${record.timestamp}: ${record.moduleId}.${record.action} - ${record.success ? 'SUCCESS' : 'FAILED'}`);
 * });
 */

export default OrchestrationAdapter;
