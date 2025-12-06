/**
 * REPZ Feature Flag Manager
 * Core engine for feature flag evaluation and management
 */

import { FeatureFlagCache } from './FeatureFlagCache';
import { FeatureFlagEvaluator } from './FeatureFlagEvaluator';
import type {
  FeatureFlag,
  FeatureFlagConfig,
  FeatureFlagContext,
  EvaluationResult,
  FeatureFlagEvent,
  UserContext
} from '../types';
import { FeatureFlagError } from '../utils/errors';
import { FEATURE_FLAG_DEFAULTS } from '../constants';

/**
 * Main feature flag management class
 * Handles flag loading, caching, evaluation, and analytics
 */
export class FeatureFlagManager {
  private cache: FeatureFlagCache;
  private evaluator: FeatureFlagEvaluator;
  private config: FeatureFlagConfig;
  private flags: Map<string, FeatureFlag> = new Map();
  private pollingInterval?: ReturnType<typeof setInterval>;
  private eventListeners: Map<string, ((event: FeatureFlagEvent) => void)[]> = new Map();

  constructor(config: Partial<FeatureFlagConfig> = {}) {
    this.config = this.mergeConfig(config);
    this.cache = new FeatureFlagCache(this.config.api.cacheTtl);
    this.evaluator = new FeatureFlagEvaluator(this.config);
    
    // Start polling for flag updates
    this.startPolling();
    
    // Load initial flags
    this.loadFlags().catch(error => {
      console.error('Failed to load initial feature flags:', error);
    });
  }

  /**
   * Evaluate a feature flag for the given context
   */
  async evaluate(
    flagId: string,
    context: FeatureFlagContext
  ): Promise<EvaluationResult> {
    const startTime = Date.now();
    
    try {
      // Check cache first
      const cacheKey = this.getCacheKey(flagId, context);
      let result = this.cache.get(cacheKey);
      
      if (!result) {
        // Get flag definition
        const flag = await this.getFlag(flagId);
        
        if (!flag) {
          result = this.createFallbackResult(flagId, 'Flag not found');
        } else if (!flag.enabled) {
          result = this.createFallbackResult(flagId, 'Flag disabled');
        } else {
          // Evaluate flag
          result = await this.evaluator.evaluate(flag, context);
        }
        
        // Cache the result
        this.cache.set(cacheKey, result);
      } else {
        result.metadata.cached = true;
      }
      
      // Update evaluation metrics
      result.metadata.duration = Date.now() - startTime;
      result.metadata.timestamp = new Date().toISOString();
      
      // Emit evaluation event
      this.emitEvent({
        type: 'evaluation',
        flagId,
        userContext: context.user,
        result,
        timestamp: result.metadata.timestamp
      });
      
      return result;
      
    } catch (error) {
      const errorResult = this.createErrorResult(flagId, error);
      
      // Emit error event
      this.emitEvent({
        type: 'error',
        flagId,
        userContext: context.user,
        result: errorResult,
        timestamp: new Date().toISOString(),
        data: { error: error instanceof Error ? error.message : 'Unknown error' }
      });
      
      return errorResult;
    }
  }

  /**
   * Check if a feature flag is enabled (simple boolean check)
   */
  async isEnabled(flagId: string, context: FeatureFlagContext): Promise<boolean> {
    const result = await this.evaluate(flagId, context);
    return result.enabled;
  }

  /**
   * Get feature flag variant (for A/B testing)
   */
  async getVariant(flagId: string, context: FeatureFlagContext): Promise<string | null> {
    const result = await this.evaluate(flagId, context);
    return result.variant || null;
  }

  /**
   * Get feature flag payload
   */
  async getPayload(flagId: string, context: FeatureFlagContext): Promise<any> {
    const result = await this.evaluate(flagId, context);
    return result.payload;
  }

  /**
   * Bulk evaluate multiple flags
   */
  async evaluateMany(
    flagIds: string[],
    context: FeatureFlagContext
  ): Promise<Record<string, EvaluationResult>> {
    const promises = flagIds.map(async flagId => {
      const result = await this.evaluate(flagId, context);
      return [flagId, result] as const;
    });
    
    const results = await Promise.all(promises);
    return Object.fromEntries(results);
  }

  /**
   * Override a flag value (for testing/debugging)
   */
  override(flagId: string, value: boolean | string | number): void {
    if (this.config.environment !== 'production') {
      this.config.development.overrides[flagId] = value;
      
      // Clear cache for this flag
      this.cache.clear(flagId);
      
      // Emit override event
      this.emitEvent({
        type: 'override',
        flagId,
        userContext: {} as UserContext, // No specific user for overrides
        result: this.createFallbackResult(flagId, 'Override applied'),
        timestamp: new Date().toISOString(),
        data: { overrideValue: value }
      });
    }
  }

  /**
   * Clear override for a flag
   */
  clearOverride(flagId: string): void {
    delete this.config.development.overrides[flagId];
    this.cache.clear(flagId);
  }

  /**
   * Get all current flag states for a context
   */
  async getAllFlags(context: FeatureFlagContext): Promise<Record<string, boolean>> {
    const flagIds = Array.from(this.flags.keys());
    const results = await this.evaluateMany(flagIds, context);
    
    return Object.fromEntries(
      Object.entries(results).map(([flagId, result]) => [flagId, result.enabled])
    );
  }

  /**
   * Refresh flags from the server
   */
  async refreshFlags(): Promise<void> {
    await this.loadFlags();
    this.cache.clear(); // Clear cache to force re-evaluation
  }

  /**
   * Add event listener
   */
  on(eventType: string, listener: (event: FeatureFlagEvent) => void): void {
    if (!this.eventListeners.has(eventType)) {
      this.eventListeners.set(eventType, []);
    }
    this.eventListeners.get(eventType)!.push(listener);
  }

  /**
   * Remove event listener
   */
  off(eventType: string, listener: (event: FeatureFlagEvent) => void): void {
    const listeners = this.eventListeners.get(eventType);
    if (listeners) {
      const index = listeners.indexOf(listener);
      if (index > -1) {
        listeners.splice(index, 1);
      }
    }
  }

  /**
   * Destroy the manager and clean up resources
   */
  destroy(): void {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
    }
    this.cache.clear();
    this.eventListeners.clear();
  }

  // Private methods

  private mergeConfig(config: Partial<FeatureFlagConfig>): FeatureFlagConfig {
    return {
      environment: config.environment || 'development',
      api: {
        supabase: config.api?.supabase,
        fallbacks: config.api?.fallbacks || {},
        cacheTtl: config.api?.cacheTtl || FEATURE_FLAG_DEFAULTS.CACHE_TTL,
        pollingInterval: config.api?.pollingInterval || FEATURE_FLAG_DEFAULTS.POLLING_INTERVAL
      },
      analytics: {
        enabled: config.analytics?.enabled ?? true,
        samplingRate: config.analytics?.samplingRate || 1.0,
        onEvent: config.analytics?.onEvent
      },
      development: {
        debug: config.development?.debug ?? (config.environment === 'development'),
        overrides: config.development?.overrides || {},
        disableCache: config.development?.disableCache ?? false
      }
    };
  }

  private async loadFlags(): Promise<void> {
    try {
      if (this.config.api.supabase) {
        // Load from Supabase
        const { data, error } = await this.config.api.supabase
          .from('feature_flags')
          .select('*')
          .eq('environment', this.config.environment);

        if (error) {
          throw new FeatureFlagError(`Failed to load flags: ${error.message}`);
        }

        // Update flags map
        this.flags.clear();
        data?.forEach((flag: FeatureFlag) => {
          this.flags.set(flag.id, flag);
        });
      }
    } catch (error) {
      console.error('Error loading feature flags:', error);
      // Continue with existing flags or fallbacks
    }
  }

  private async getFlag(flagId: string): Promise<FeatureFlag | null> {
    // Check for development overrides
    if (this.config.development.overrides[flagId] !== undefined) {
      const overrideValue = this.config.development.overrides[flagId];
      return this.createOverrideFlag(flagId, overrideValue);
    }

    // Get from loaded flags
    return this.flags.get(flagId) || null;
  }

  private createOverrideFlag(flagId: string, overrideValue: any): FeatureFlag {
    return {
      id: flagId,
      name: `Override: ${flagId}`,
      description: 'Development override flag',
      enabled: Boolean(overrideValue),
      environment: { name: 'development', overrides: {}, killSwitch: false, requiresApproval: false },
      strategy: { type: 'percentage', config: { percentage: 100 } },
      segments: [],
      variants: overrideValue !== true ? [{ 
        id: 'override', 
        name: 'Override', 
        allocation: 100, 
        payload: overrideValue,
        description: 'Override variant'
      }] : [],
      rules: [],
      metadata: {
        owner: 'system',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        category: 'override',
        tags: ['development'],
        temporary: true
      }
    };
  }

  private getCacheKey(flagId: string, context: FeatureFlagContext): string {
    // Create a cache key based on flag ID and relevant context
    const keyParts = [
      flagId,
      context.user.id,
      context.user.tier,
      context.user.role,
      context.user.attributes.subscriptionStatus
    ];
    
    return keyParts.join(':');
  }

  private createFallbackResult(flagId: string, reason: string): EvaluationResult {
    const fallbackValue = this.config.api.fallbacks[flagId];
    
    return {
      flagId,
      enabled: Boolean(fallbackValue),
      variant: typeof fallbackValue === 'string' ? fallbackValue : undefined,
      payload: fallbackValue,
      reason,
      metadata: {
        timestamp: new Date().toISOString(),
        duration: 0,
        cached: false
      }
    };
  }

  private createErrorResult(flagId: string, error: unknown): EvaluationResult {
    return {
      flagId,
      enabled: false,
      reason: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      metadata: {
        timestamp: new Date().toISOString(),
        duration: 0,
        cached: false
      }
    };
  }

  private startPolling(): void {
    if (this.config.api.pollingInterval > 0) {
      this.pollingInterval = setInterval(() => {
        this.loadFlags().catch(error => {
          console.error('Error during flag polling:', error);
        });
      }, this.config.api.pollingInterval * 1000);
    }
  }

  private emitEvent(event: FeatureFlagEvent): void {
    // Sampling check
    if (Math.random() > this.config.analytics.samplingRate) {
      return;
    }

    // Custom event handler
    if (this.config.analytics.onEvent) {
      this.config.analytics.onEvent(event);
    }

    // Event listeners
    const listeners = this.eventListeners.get(event.type) || [];
    listeners.forEach(listener => {
      try {
        listener(event);
      } catch (error) {
        console.error('Error in event listener:', error);
      }
    });

    // Debug logging
    if (this.config.development.debug) {
      console.log('Feature Flag Event:', event);
    }
  }
}
