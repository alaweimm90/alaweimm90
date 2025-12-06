/**
 * REPZ Feature Flag Platform - Type Definitions
 * Comprehensive type system for enterprise feature management
 */

/**
 * Core feature flag definition
 */
export interface FeatureFlag {
  /** Unique identifier for the feature flag */
  id: string;
  
  /** Human-readable name */
  name: string;
  
  /** Detailed description of the feature */
  description: string;
  
  /** Feature flag status */
  enabled: boolean;
  
  /** Environment-specific settings */
  environment: FeatureFlagEnvironment;
  
  /** Rollout strategy configuration */
  strategy: FeatureFlagStrategy;
  
  /** Target segments for the feature */
  segments: FeatureFlagSegment[];
  
  /** Feature variants for A/B testing */
  variants: FeatureFlagVariant[];
  
  /** Evaluation rules */
  rules: FeatureFlagRule[];
  
  /** Metadata */
  metadata: {
    /** Feature owner/team */
    owner: string;
    
    /** Creation timestamp */
    createdAt: string;
    
    /** Last update timestamp */
    updatedAt: string;
    
    /** Feature category */
    category: string;
    
    /** Tags for organization */
    tags: string[];
    
    /** Temporary flag indicator */
    temporary: boolean;
    
    /** Planned removal date */
    sunset?: string;
  };
  
  /** Performance and usage metrics */
  metrics?: FeatureFlagMetrics;
}

/**
 * Feature flag configuration
 */
export interface FeatureFlagConfig {
  /** Default environment */
  environment: 'development' | 'staging' | 'production';
  
  /** API configuration */
  api: {
    /** Supabase client instance */
    supabase?: any;
    
    /** Fallback values when API unavailable */
    fallbacks: Record<string, boolean | string | number>;
    
    /** Cache TTL in seconds */
    cacheTtl: number;
    
    /** Polling interval for updates */
    pollingInterval: number;
  };
  
  /** Analytics configuration */
  analytics: {
    /** Enable event tracking */
    enabled: boolean;
    
    /** Sampling rate (0-1) */
    samplingRate: number;
    
    /** Custom event handler */
    onEvent?: (event: FeatureFlagEvent) => void;
  };
  
  /** Development settings */
  development: {
    /** Enable debug logging */
    debug: boolean;
    
    /** Override flags for testing */
    overrides: Record<string, boolean | string | number>;
    
    /** Disable caching in dev */
    disableCache: boolean;
  };
}

/**
 * User context for evaluation
 */
export interface UserContext {
  /** User ID */
  id: string;
  
  /** User role */
  role: 'client' | 'coach' | 'admin';
  
  /** Subscription tier */
  tier: 'core' | 'adaptive' | 'performance' | 'longevity';
  
  /** User attributes */
  attributes: {
    /** Account creation date */
    createdAt: string;
    
    /** Email domain */
    emailDomain: string;
    
    /** Subscription status */
    subscriptionStatus: 'active' | 'inactive' | 'trial' | 'expired';
    
    /** Geographic location */
    location?: GeographicContext;
    
    /** Device information */
    device?: DeviceContext;
    
    /** Custom attributes */
    custom: Record<string, any>;
  };
}

/**
 * Geographic context
 */
export interface GeographicContext {
  /** Country code (ISO 3166-1 alpha-2) */
  country: string;
  
  /** Region/state */
  region?: string;
  
  /** City */
  city?: string;
  
  /** Timezone */
  timezone: string;
}

/**
 * Device context
 */
export interface DeviceContext {
  /** Device type */
  type: 'desktop' | 'mobile' | 'tablet';
  
  /** Operating system */
  os: string;
  
  /** Browser information */
  browser?: string;
  
  /** Screen resolution */
  screenResolution?: string;
  
  /** User agent */
  userAgent?: string;
}

/**
 * Tier-specific context
 */
export interface TierContext {
  /** Current tier */
  tier: 'core' | 'adaptive' | 'performance' | 'longevity';
  
  /** Tier features enabled */
  features: string[];
  
  /** Usage limits */
  limits: Record<string, number>;
  
  /** Tier upgrade date */
  upgradeDate?: string;
}

/**
 * Experiment context for A/B testing
 */
export interface ExperimentContext {
  /** Experiment ID */
  experimentId: string;
  
  /** Variant assignment */
  variant: string;
  
  /** Traffic allocation percentage */
  trafficAllocation: number;
  
  /** Experiment start date */
  startDate: string;
  
  /** Experiment end date */
  endDate?: string;
}

/**
 * Complete evaluation context
 */
export interface FeatureFlagContext {
  /** User information */
  user: UserContext;
  
  /** Request information */
  request?: {
    /** Request path */
    path: string;
    
    /** HTTP method */
    method: string;
    
    /** Request headers */
    headers: Record<string, string>;
    
    /** Query parameters */
    query: Record<string, string>;
  };
  
  /** Current timestamp */
  timestamp: string;
  
  /** Session information */
  session?: {
    /** Session ID */
    id: string;
    
    /** Session start time */
    startTime: string;
    
    /** Page views in session */
    pageViews: number;
  };
  
  /** Additional context */
  custom: Record<string, any>;
}

/**
 * Feature flag evaluation result
 */
export interface EvaluationResult {
  /** Flag ID */
  flagId: string;
  
  /** Evaluation result */
  enabled: boolean;
  
  /** Selected variant (for A/B testing) */
  variant?: string;
  
  /** Variant payload */
  payload?: any;
  
  /** Reason for the result */
  reason: string;
  
  /** Rule that matched */
  matchedRule?: string;
  
  /** Evaluation metadata */
  metadata: {
    /** Evaluation timestamp */
    timestamp: string;
    
    /** Evaluation duration (ms) */
    duration: number;
    
    /** Cache hit/miss */
    cached: boolean;
    
    /** Experiment information */
    experiment?: ExperimentContext;
  };
}

/**
 * Feature flag rollout strategy
 */
export interface FeatureFlagStrategy {
  /** Strategy type */
  type: 'percentage' | 'user_list' | 'tier_based' | 'geographic' | 'custom';
  
  /** Strategy configuration */
  config: {
    /** Percentage rollout (0-100) */
    percentage?: number;
    
    /** Specific user IDs */
    userIds?: string[];
    
    /** Tier restrictions */
    tiers?: ('core' | 'adaptive' | 'performance' | 'longevity')[];
    
    /** Geographic restrictions */
    countries?: string[];
    
    /** Custom rule expression */
    expression?: string;
  };
  
  /** Gradual rollout settings */
  gradual?: {
    /** Start percentage */
    startPercentage: number;
    
    /** End percentage */
    endPercentage: number;
    
    /** Rollout duration (hours) */
    duration: number;
    
    /** Step size */
    stepSize: number;
  };
}

/**
 * User segment definition
 */
export interface FeatureFlagSegment {
  /** Segment ID */
  id: string;
  
  /** Segment name */
  name: string;
  
  /** Segment rules */
  rules: FeatureFlagRule[];
  
  /** Segment description */
  description: string;
}

/**
 * Feature flag rule
 */
export interface FeatureFlagRule {
  /** Rule ID */
  id: string;
  
  /** Rule condition */
  condition: {
    /** Attribute to check */
    attribute: string;
    
    /** Comparison operator */
    operator: 'equals' | 'not_equals' | 'in' | 'not_in' | 'contains' | 'greater_than' | 'less_than' | 'regex';
    
    /** Value to compare against */
    value: any;
  };
  
  /** Rule weight (for prioritization) */
  weight: number;
  
  /** Rule result */
  result: boolean | string | number;
}

/**
 * Feature variant for A/B testing
 */
export interface FeatureFlagVariant {
  /** Variant ID */
  id: string;
  
  /** Variant name */
  name: string;
  
  /** Traffic allocation percentage */
  allocation: number;
  
  /** Variant payload */
  payload?: any;
  
  /** Variant description */
  description: string;
}

/**
 * Environment configuration
 */
export interface FeatureFlagEnvironment {
  /** Environment name */
  name: 'development' | 'staging' | 'production';
  
  /** Environment-specific overrides */
  overrides: Record<string, boolean | string | number>;
  
  /** Kill switch for emergency disable */
  killSwitch: boolean;
  
  /** Approval required for changes */
  requiresApproval: boolean;
}

/**
 * Feature flag metrics
 */
export interface FeatureFlagMetrics {
  /** Evaluation count */
  evaluations: number;
  
  /** Enabled percentage */
  enabledPercentage: number;
  
  /** Performance metrics */
  performance: {
    /** Average evaluation time (ms) */
    avgEvaluationTime: number;
    
    /** 95th percentile evaluation time */
    p95EvaluationTime: number;
    
    /** Error rate */
    errorRate: number;
  };
  
  /** Usage by segment */
  segmentUsage: Record<string, number>;
  
  /** Conversion metrics (if applicable) */
  conversion?: {
    /** Total conversions */
    total: number;
    
    /** Conversion rate */
    rate: number;
    
    /** Conversion by variant */
    byVariant: Record<string, number>;
  };
}

/**
 * Feature flag event for analytics
 */
export interface FeatureFlagEvent {
  /** Event type */
  type: 'evaluation' | 'error' | 'override' | 'experiment_assignment';
  
  /** Flag ID */
  flagId: string;
  
  /** User context */
  userContext: UserContext;
  
  /** Evaluation result */
  result: EvaluationResult;
  
  /** Event timestamp */
  timestamp: string;
  
  /** Additional event data */
  data?: Record<string, any>;
}