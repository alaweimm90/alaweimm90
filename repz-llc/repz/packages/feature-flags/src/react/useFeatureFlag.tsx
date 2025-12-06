/**
 * REPZ Feature Flag Hook
 * Primary hook for feature flag evaluation in React components
 */

import { useState, useEffect, useContext } from 'react';
import { FeatureFlagContext } from './FeatureFlagProvider';
import type { EvaluationResult } from '../types';

export interface UseFeatureFlagResult {
  /** Whether the feature is enabled */
  enabled: boolean;
  
  /** Loading state */
  loading: boolean;
  
  /** Error state */
  error: Error | null;
  
  /** Full evaluation result */
  result: EvaluationResult | null;
  
  /** Feature variant (for A/B testing) */
  variant: string | null;
  
  /** Feature payload */
  payload: any;
  
  /** Force re-evaluation */
  refresh: () => Promise<void>;
}

/**
 * Hook to evaluate a feature flag
 * 
 * @param flagId - The feature flag identifier
 * @returns Feature flag evaluation result with loading and error states
 * 
 * @example
 * ```tsx
 * function MyComponent() {
 *   const { enabled, loading, variant } = useFeatureFlag('new-dashboard');
 *   
 *   if (loading) return <Spinner />;
 *   
 *   return enabled ? <NewDashboard variant={variant} /> : <OldDashboard />;
 * }
 * ```
 */
export function useFeatureFlag(flagId: string): UseFeatureFlagResult {
  const context = useContext(FeatureFlagContext);
  
  if (!context) {
    throw new Error('useFeatureFlag must be used within a FeatureFlagProvider');
  }
  
  const { manager, userContext } = context;
  
  const [state, setState] = useState<{
    enabled: boolean;
    loading: boolean;
    error: Error | null;
    result: EvaluationResult | null;
  }>({
    enabled: false,
    loading: true,
    error: null,
    result: null
  });
  
  const evaluateFlag = async () => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }));
      
      const result = await manager.evaluate(flagId, {
        user: userContext,
        timestamp: new Date().toISOString(),
        custom: {}
      });
      
      setState({
        enabled: result.enabled,
        loading: false,
        error: null,
        result
      });
      
    } catch (error) {
      setState({
        enabled: false,
        loading: false,
        error: error instanceof Error ? error : new Error('Unknown error'),
        result: null
      });
    }
  };
  
  useEffect(() => {
    evaluateFlag();
  }, [flagId, userContext.id, userContext.tier, userContext.role]);
  
  const refresh = async () => {
    await evaluateFlag();
  };
  
  return {
    enabled: state.enabled,
    loading: state.loading,
    error: state.error,
    result: state.result,
    variant: state.result?.variant || null,
    payload: state.result?.payload,
    refresh
  };
}

/**
 * Hook to check if a feature flag is enabled (simple boolean)
 * 
 * @param flagId - The feature flag identifier
 * @returns Boolean indicating if the feature is enabled
 * 
 * @example
 * ```tsx
 * function MyComponent() {
 *   const isNewFeatureEnabled = useFeatureFlagEnabled('new-feature');
 *   
 *   return (
 *     <div>
 *       {isNewFeatureEnabled && <NewFeature />}
 *     </div>
 *   );
 * }
 * ```
 */
export function useFeatureFlagEnabled(flagId: string): boolean {
  const { enabled } = useFeatureFlag(flagId);
  return enabled;
}

/**
 * Hook to get feature flag variant
 * 
 * @param flagId - The feature flag identifier
 * @returns The variant string or null
 * 
 * @example
 * ```tsx
 * function MyComponent() {
 *   const variant = useFeatureFlagVariant('button-color-test');
 *   
 *   return (
 *     <Button color={variant === 'red' ? 'red' : 'blue'}>
 *       Click me
 *     </Button>
 *   );
 * }
 * ```
 */
export function useFeatureFlagVariant(flagId: string): string | null {
  const { variant } = useFeatureFlag(flagId);
  return variant;
}

/**
 * Hook to get feature flag payload
 * 
 * @param flagId - The feature flag identifier
 * @returns The payload data
 * 
 * @example
 * ```tsx
 * function MyComponent() {
 *   const config = useFeatureFlagPayload('pricing-config');
 *   
 *   return (
 *     <PricingTable 
 *       discount={config?.discount || 0}
 *       features={config?.features || []}
 *     />
 *   );
 * }
 * ```
 */
export function useFeatureFlagPayload(flagId: string): any {
  const { payload } = useFeatureFlag(flagId);
  return payload;
}