/**
 * REPZ Feature Flag Platform
 * Enterprise-grade feature management system for controlled rollouts
 * 
 * Enables:
 * - Decoupled deployment from feature release
 * - A/B testing and experimentation
 * - Gradual rollouts with safety controls
 * - Real-time feature toggling
 * - User segmentation and targeting
 * 
 * @version 1.0.0
 * @author REPZ Team
 * @license MIT
 */

// Core feature flag engine
export { FeatureFlagManager } from './core/FeatureFlagManager';
export { FeatureFlagCache } from './core/FeatureFlagCache';
export { FeatureFlagEvaluator } from './core/FeatureFlagEvaluator';

// Configuration and types
export type {
  FeatureFlag,
  FeatureFlagConfig,
  FeatureFlagContext,
  FeatureFlagRule,
  FeatureFlagSegment,
  FeatureFlagVariant,
  FeatureFlagStrategy,
  FeatureFlagEnvironment,
  FeatureFlagMetrics,
  FeatureFlagEvent,
  EvaluationResult,
  UserContext,
  TierContext,
  GeographicContext,
  DeviceContext,
  ExperimentContext
} from './types';

// Utilities
export { createFeatureFlagConfig } from './utils/config';
export { validateFlag } from './utils/validation';
export { FeatureFlagError, FeatureFlagValidationError } from './utils/errors';

// Constants
export { FEATURE_FLAG_DEFAULTS, TIER_SEGMENTS, ROLLOUT_STRATEGIES } from './constants';