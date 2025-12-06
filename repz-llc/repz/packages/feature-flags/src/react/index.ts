/**
 * REPZ Feature Flags - React Integration
 * Hooks and components for React applications
 */

export { FeatureFlagProvider } from './FeatureFlagProvider';
export { useFeatureFlag } from './useFeatureFlag';
export { useFeatureFlags } from './useFeatureFlags';
export { useFeatureFlagVariant } from './useFeatureFlagVariant';
export { FeatureGate } from './FeatureGate';
export { VariantSwitch } from './VariantSwitch';
export { ExperimentProvider } from './ExperimentProvider';
export { useExperiment } from './useExperiment';

export type {
  FeatureFlagProviderProps,
  UseFeatureFlagResult,
  FeatureGateProps,
  VariantSwitchProps,
  ExperimentProviderProps,
  UseExperimentResult
} from './types';