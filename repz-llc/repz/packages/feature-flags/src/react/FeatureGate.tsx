/**
 * REPZ Feature Gate Component
 * Conditionally renders children based on feature flag state
 */

import React, { ReactNode } from 'react';
import { useFeatureFlag } from './useFeatureFlag';

export interface FeatureGateProps {
  /** Feature flag ID to check */
  flagId: string;
  
  /** Content to render when feature is enabled */
  children: ReactNode;
  
  /** Content to render when feature is disabled */
  fallback?: ReactNode;
  
  /** Content to render while loading */
  loading?: ReactNode;
  
  /** Content to render on error */
  error?: ReactNode;
  
  /** Invert the logic (render children when disabled) */
  not?: boolean;
  
  /** Required variant to render children (for A/B testing) */
  variant?: string;
  
  /** Function to determine if children should render based on full result */
  when?: (result: {
    enabled: boolean;
    variant: string | null;
    payload: any;
  }) => boolean;
}

/**
 * Feature Gate Component
 * 
 * Conditionally renders content based on feature flag evaluation
 * 
 * @example
 * ```tsx
 * // Basic usage
 * <FeatureGate flagId="new-dashboard">
 *   <NewDashboard />
 * </FeatureGate>
 * 
 * // With fallback
 * <FeatureGate 
 *   flagId="new-dashboard" 
 *   fallback={<OldDashboard />}
 * >
 *   <NewDashboard />
 * </FeatureGate>
 * 
 * // Variant-specific rendering
 * <FeatureGate 
 *   flagId="button-color-test" 
 *   variant="red"
 * >
 *   <RedButton />
 * </FeatureGate>
 * 
 * // Inverted logic
 * <FeatureGate flagId="old-feature" not>
 *   <NewFeature />
 * </FeatureGate>
 * 
 * // Custom condition
 * <FeatureGate 
 *   flagId="pricing-experiment"
 *   when={({ enabled, payload }) => enabled && payload?.discount > 20}
 * >
 *   <DiscountBanner />
 * </FeatureGate>
 * ```
 */
export function FeatureGate({
  flagId,
  children,
  fallback = null,
  loading = null,
  error = null,
  not = false,
  variant,
  when
}: FeatureGateProps): JSX.Element | null {
  const { 
    enabled, 
    loading: isLoading, 
    error: evaluationError, 
    variant: flagVariant, 
    payload 
  } = useFeatureFlag(flagId);

  // Handle loading state
  if (isLoading) {
    return <>{loading}</>;
  }

  // Handle error state
  if (evaluationError) {
    return <>{error}</>;
  }

  // Determine if children should render
  let shouldRender = enabled;

  // Apply inversion
  if (not) {
    shouldRender = !shouldRender;
  }

  // Check variant requirement
  if (variant && flagVariant !== variant) {
    shouldRender = false;
  }

  // Apply custom condition
  if (when) {
    shouldRender = when({
      enabled,
      variant: flagVariant,
      payload
    });
  }

  return shouldRender ? <>{children}</> : <>{fallback}</>;
}

/**
 * Feature Toggle Component (alias for FeatureGate)
 */
export const FeatureToggle = FeatureGate;

/**
 * Variant Switch Component
 * Renders different content based on feature flag variant
 */
export interface VariantSwitchProps {
  /** Feature flag ID */
  flagId: string;
  
  /** Variant mappings */
  variants: Record<string, ReactNode>;
  
  /** Default content when no variant matches */
  default?: ReactNode;
  
  /** Content to render while loading */
  loading?: ReactNode;
  
  /** Content to render on error */
  error?: ReactNode;
}

/**
 * Variant Switch Component
 * 
 * Renders different content based on feature flag variant
 * 
 * @example
 * ```tsx
 * <VariantSwitch
 *   flagId="button-color-test"
 *   variants={{
 *     red: <RedButton />,
 *     blue: <BlueButton />,
 *     green: <GreenButton />
 *   }}
 *   default={<DefaultButton />}
 * />
 * ```
 */
export function VariantSwitch({
  flagId,
  variants,
  default: defaultContent = null,
  loading = null,
  error = null
}: VariantSwitchProps): JSX.Element | null {
  const { 
    enabled, 
    loading: isLoading, 
    error: evaluationError, 
    variant 
  } = useFeatureFlag(flagId);

  // Handle loading state
  if (isLoading) {
    return <>{loading}</>;
  }

  // Handle error state
  if (evaluationError) {
    return <>{error}</>;
  }

  // If flag is disabled or no variant, show default
  if (!enabled || !variant) {
    return <>{defaultContent}</>;
  }

  // Render variant content
  const variantContent = variants[variant];
  return <>{variantContent || defaultContent}</>;
}