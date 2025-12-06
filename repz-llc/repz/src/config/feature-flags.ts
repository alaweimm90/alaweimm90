/**
 * Feature flags for enterprise refactoring
 * Enables gradual rollout and safe rollback of changes
 */

export interface FeatureFlags {
  // Component consolidation flags
  USE_CONSOLIDATED_COMPONENTS: boolean;
  USE_UNIFIED_DASHBOARD: boolean;
  USE_CENTRALIZED_FORMS: boolean;
  
  // Theme migration flags
  USE_SEMANTIC_TOKENS: boolean;
  USE_MODERN_ANIMATIONS: boolean;
  USE_OPTIMIZED_STYLES: boolean;
  
  // Architecture flags
  USE_CENTRALIZED_ROUTING: boolean;
  USE_UNIFIED_STATE: boolean;
  USE_OPTIMIZED_QUERIES: boolean;
  
  // Performance flags
  ENABLE_CODE_SPLITTING: boolean;
  ENABLE_LAZY_LOADING: boolean;
  ENABLE_CACHE_OPTIMIZATION: boolean;
}

// Production-safe feature flag configuration
export const featureFlags: FeatureFlags = {
  // Phase 2: Component consolidation (enabled for cleanup)
  USE_CONSOLIDATED_COMPONENTS: true,
  USE_UNIFIED_DASHBOARD: true,
  USE_CENTRALIZED_FORMS: true,
  
  // Phase 3: Theme migration (enabled)
  USE_SEMANTIC_TOKENS: true,
  USE_MODERN_ANIMATIONS: true,
  USE_OPTIMIZED_STYLES: true,
  
  // Phase 4: Architecture (enabled)
  USE_CENTRALIZED_ROUTING: true,
  USE_UNIFIED_STATE: true,
  USE_OPTIMIZED_QUERIES: true,
  
  // Performance optimizations (can be enabled independently)
  ENABLE_CODE_SPLITTING: false,
  ENABLE_LAZY_LOADING: false,
  ENABLE_CACHE_OPTIMIZATION: false,
};

// Feature flag helper functions
export const isFeatureEnabled = (flag: keyof FeatureFlags): boolean => {
  return featureFlags[flag];
};

export const enableFeature = (flag: keyof FeatureFlags): void => {
  featureFlags[flag] = true;
  console.log(`Feature flag ${flag} enabled`);
};

export const disableFeature = (flag: keyof FeatureFlags): void => {
  featureFlags[flag] = false;
  console.log(`Feature flag ${flag} disabled`);
};

// Batch operations for phases
export const enablePhase2 = (): void => {
  enableFeature('USE_CONSOLIDATED_COMPONENTS');
  enableFeature('USE_UNIFIED_DASHBOARD');
  enableFeature('USE_CENTRALIZED_FORMS');
};

export const enablePhase3 = (): void => {
  enableFeature('USE_SEMANTIC_TOKENS');
  enableFeature('USE_MODERN_ANIMATIONS');
  enableFeature('USE_OPTIMIZED_STYLES');
};

export const enablePhase4 = (): void => {
  enableFeature('USE_CENTRALIZED_ROUTING');
  enableFeature('USE_UNIFIED_STATE');
  enableFeature('USE_OPTIMIZED_QUERIES');
};

// Emergency rollback
export const disableAllFeatures = (): void => {
  Object.keys(featureFlags).forEach(flag => {
    featureFlags[flag as keyof FeatureFlags] = false;
  });
  console.log('All feature flags disabled - system reverted to baseline');
};