// UPDATED: Complete tier access hook with new features and legacy cleanup

import React from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { supabase } from '@/integrations/supabase/client';
import { TIER_FEATURES, hasFeatureAccess, type TierFeatures } from '@/constants/featureMatrix';
import { TierType } from '@/constants/tiers';

export interface TierAccessResult {
  userTier: TierType | null;
  isLoading: boolean;
  
  // Feature access checks
  hasFeature: (featureKey: keyof TierFeatures) => boolean;
  hasMinimumTier: (requiredTier: TierType) => boolean;
  hasAccess: (featureKey: keyof TierFeatures) => boolean;
  
  // Specific feature shortcuts (commonly used)
  hasWeeklyCheckins: boolean;
  hasFormReviews: boolean;
  hasDeviceIntegration: boolean;
  hasAIAssistant: boolean;
  hasBiomarkerTracking: boolean;
  hasAutoGroceryLists: boolean;
  hasTravelWorkouts: boolean;
  hasPEDsProtocols: boolean;
  hasNootropics: boolean;
  hasBioregulators: boolean;
  hasInPersonTraining: boolean;
  
  // Access levels
  responseTimeHours: number;
  messageLimit: number;
  dashboardType: 'static_fixed' | 'interactive_adjustable';
  
  // Upgrade prompts
  upgradePrompts: string[];
  nextTier: TierType | null;
  upgradeMessage: string | null;
}

export function useTierAccess(): TierAccessResult {
  const { user, profile, loading } = useAuth();
  const [remoteTier, setRemoteTier] = React.useState<TierType | null>(null);
  
  const normalizedTier = normalizeTierName(profile?.subscription_tier) 
    ?? remoteTier;

  React.useEffect(() => {
    let active = true;
    if (!profile?.subscription_tier && user?.id) {
      supabase
        .from('profiles')
        .select('subscription_tier')
        .eq('id', user.id)
        .single()
        .then(({ data }) => {
          if (!active) return;
          const t = normalizeTierName((data as any)?.subscription_tier);
          if (t) setRemoteTier(t);
        })
        .catch(() => {})
    }
    return () => { active = false };
  }, [profile?.subscription_tier, user?.id]);
  
  // Get features for user's tier
  const tierFeatures = normalizedTier ? TIER_FEATURES[normalizedTier] : null;
  
  // Helper to check feature access
  const hasFeature = (featureKey: keyof TierFeatures): boolean => {
    if (!normalizedTier || !tierFeatures) return false;
    return hasFeatureAccess(normalizedTier, featureKey);
  };
  
  // Check if user has minimum tier level
  const hasMinimumTier = (requiredTier: TierType): boolean => {
    if (!normalizedTier) return false;
    const hierarchy: TierType[] = ['core', 'adaptive', 'performance', 'longevity'];
    const currentIndex = hierarchy.indexOf(normalizedTier);
    const requiredIndex = hierarchy.indexOf(requiredTier);
    return currentIndex >= requiredIndex;
  };
  
  // Get next tier for upgrade prompts
  const getNextTier = (): TierType | null => {
    if (!normalizedTier) return 'core';
    const hierarchy: TierType[] = ['core', 'adaptive', 'performance', 'longevity'];
    const currentIndex = hierarchy.indexOf(normalizedTier);
    if (currentIndex === -1 || currentIndex === hierarchy.length - 1) return null;
    return hierarchy[currentIndex + 1];
  };
  
  // Generate upgrade message based on missing features
  const generateUpgradeMessage = (): string | null => {
    const nextTier = getNextTier();
    if (!nextTier) return null;
    
    const messages: Record<TierType, string> = {
      core: 'Unlock weekly check-ins and progress tracking',
      adaptive: 'Get AI-powered insights and advanced analytics',
      performance: 'Access exclusive longevity protocols and in-person training',
      longevity: null // Top tier
    };
    
    return messages[normalizedTier as TierType] || null;
  };
  
  // Get upgrade prompts based on current tier
  const getUpgradePrompts = (): string[] => {
    if (!normalizedTier) return [];
    
    const prompts: Record<TierType, string[]> = {
      core: ['weekly_checkins', 'biomarker_integration', 'auto_grocery_lists'],
      adaptive: ['ai_fitness_assistant', 'peds_protocols', 'nootropics_productivity'],
      performance: ['bioregulators_protocols', 'custom_cycling_schemes', 'in_person_training'],
      longevity: []
    };
    
    return prompts[normalizedTier] || [];
  };
  
  return {
    userTier: normalizedTier,
    isLoading: loading,
    
    // Feature access
    hasFeature,
    hasMinimumTier,
    hasAccess: hasFeature,
    
    // Specific features (shortcuts)
    hasWeeklyCheckins: hasFeature('weekly_checkins'),
    hasFormReviews: hasFeature('form_review'),
    hasDeviceIntegration: hasFeature('wearable_sync'),
    hasAIAssistant: hasFeature('ai_fitness_assistant'),
    hasBiomarkerTracking: hasFeature('biomarker_integration'),
    hasAutoGroceryLists: hasFeature('auto_grocery_lists'),
    hasTravelWorkouts: hasFeature('travel_workouts'),
    hasPEDsProtocols: hasFeature('peds'),
    hasNootropics: hasFeature('nootropics'),
    hasBioregulators: hasFeature('bioregulators'),
    hasInPersonTraining: hasFeature('in_person_training'),
    
    // Access levels
    responseTimeHours: tierFeatures?.response_time_hours || 72,
    messageLimit: -1, // No message limits in new system
    dashboardType: tierFeatures?.dashboard_type === 'interactive_adjustable' ? 'interactive' : 'static_fixed',
    
    // Upgrade info
    upgradePrompts: getUpgradePrompts(),
    nextTier: getNextTier(),
    upgradeMessage: generateUpgradeMessage()
  };
}

// Helper function to normalize legacy tier names
function normalizeTierName(tier: string | undefined | null): TierType | null {
  if (!tier) return null;
  
  // Handle legacy tier names - convert to canonical names
  const legacyMapping: Record<string, TierType> = {
    'baseline': 'core',
    'prime': 'adaptive', 
    'precision': 'performance',
    'longevity': 'longevity',
    // Also handle current names for robustness
    'core': 'core',
    'adaptive': 'adaptive',
    'performance': 'performance'
  };
  
  return legacyMapping[tier.toLowerCase()] || null;
}

// Export normalized tier type for use in other components
export { normalizeTierName };

// Export useFeatureAccess as alias for backward compatibility
export const useFeatureAccess = useTierAccess;
