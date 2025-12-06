// ============= REPZ TIER FEATURE MATRIX =============
// Based on the canonical tier breakdown table provided by user
// This is the SINGLE SOURCE OF TRUTH for all feature access

import { TierType } from '@/types/fitness';

export interface TierFeatures {
  // Core Platform
  personalized_training: boolean;
  nutrition_plan: boolean;
  dashboard_type: 'static_fixed' | 'interactive_adjustable';
  
  // Coach Access
  qa_access: 'limited' | 'standard' | 'priority' | 'unlimited';
  response_time_hours: 72 | 48 | 24 | 12;
  
  // Progress Tracking
  weekly_checkins: boolean;
  form_review: boolean;
  wearable_sync: boolean;
  sleep_optimization: boolean;
  ai_fitness_assistant: boolean;
  ai_progress_predictors: boolean;
  
  // Advanced AI Features (Performance+)
  voice_coaching: boolean;
  video_analysis: boolean;
  universal_search: boolean;
  production_monitoring: boolean;
  quality_assurance: boolean;
  
  // Convenience
  auto_grocery_lists: boolean;
  travel_workouts: boolean;
  
  // Education
  science_tips: boolean;
  supplement_guide: boolean;
  research_blog_access: boolean;
  
  // Supplementation
  supplement_protocols: boolean;
  peptides: boolean;
  peds: boolean;
  nootropics: boolean;
  bioregulators: boolean;
  custom_cycling: boolean;
  
  // Analytics
  biomarker_integration: boolean;
  blood_work_review: boolean;
  recovery_guidance: boolean;
  hrv_optimization: boolean;
  
  // Community & Access
  telegram_group: boolean;
  exclusive_protocols: boolean;
  early_access_tools: boolean;
  
  // Premium Services
  in_person_training: boolean | string;
}

// CANONICAL FEATURE MATRIX - Based on user-provided tier breakdown table
export const TIER_FEATURES: Record<TierType, TierFeatures> = {
  core: {
    // Core Platform ✅
    personalized_training: true,
    nutrition_plan: true,
    dashboard_type: 'static_fixed',
    
    // Coach Access
    qa_access: 'limited',
    response_time_hours: 72,
    
    // Progress Tracking ❌
    weekly_checkins: false,
    form_review: false,
    wearable_sync: false,
    sleep_optimization: false,
    ai_fitness_assistant: false,
    ai_progress_predictors: false,
    
    // Advanced AI Features ❌
    voice_coaching: false,
    video_analysis: false,
    universal_search: false,
    production_monitoring: false,
    quality_assurance: false,
    
    // Convenience ❌
    auto_grocery_lists: false,
    travel_workouts: false,
    
    // Education ✅
    science_tips: true,
    supplement_guide: false,
    research_blog_access: false,
    
    // Supplementation ❌
    supplement_protocols: false,
    peptides: false,
    peds: false,
    nootropics: false,
    bioregulators: false,
    custom_cycling: false,
    
    // Analytics ❌
    biomarker_integration: false,
    blood_work_review: false,
    recovery_guidance: false,
    hrv_optimization: false,
    
    // Community & Access ❌
    telegram_group: false,
    exclusive_protocols: false,
    early_access_tools: false,
    
    // Premium Services ❌
    in_person_training: false
  },

  adaptive: {
    // Core Platform ✅
    personalized_training: true,
    nutrition_plan: true,
    dashboard_type: 'interactive_adjustable',
    
    // Coach Access
    qa_access: 'standard',
    response_time_hours: 48,
    
    // Progress Tracking ✅
    weekly_checkins: true,
    form_review: true,
    wearable_sync: true,
    sleep_optimization: true,
    ai_fitness_assistant: false,
    ai_progress_predictors: false,
    
    // Advanced AI Features ❌
    voice_coaching: false,
    video_analysis: false,
    universal_search: false,
    production_monitoring: false,
    quality_assurance: false,
    
    // Convenience ✅
    auto_grocery_lists: true,
    travel_workouts: true,
    
    // Education ✅
    science_tips: true,
    supplement_guide: true,
    research_blog_access: true,
    
    // Supplementation ✅
    supplement_protocols: true,
    peptides: true,
    peds: false,
    nootropics: false,
    bioregulators: true,
    custom_cycling: false,
    
    // Analytics ✅
    biomarker_integration: true,
    blood_work_review: true,
    recovery_guidance: true,
    hrv_optimization: true,
    
    // Community & Access ✅
    telegram_group: true,
    exclusive_protocols: true,
    early_access_tools: false,
    
    // Premium Services ❌
    in_person_training: false
  },

  performance: {
    // Core Platform ✅
    personalized_training: true,
    nutrition_plan: true,
    dashboard_type: 'interactive_adjustable',
    
    // Coach Access
    qa_access: 'priority',
    response_time_hours: 24,
    
    // Progress Tracking ✅
    weekly_checkins: true,
    form_review: true,
    wearable_sync: true,
    sleep_optimization: true,
    ai_fitness_assistant: true,
    ai_progress_predictors: true,
    
    // Advanced AI Features ✅
    voice_coaching: true,
    video_analysis: true,
    universal_search: true,
    production_monitoring: true,
    quality_assurance: false, // Coach exclusive
    
    // Convenience ✅
    auto_grocery_lists: true,
    travel_workouts: true,
    
    // Education ✅
    science_tips: true,
    supplement_guide: true,
    research_blog_access: true,
    
    // Supplementation ✅
    supplement_protocols: true,
    peptides: true,
    peds: true,
    nootropics: true,
    bioregulators: true,
    custom_cycling: true,
    
    // Analytics ✅
    biomarker_integration: true,
    blood_work_review: true,
    recovery_guidance: true,
    hrv_optimization: true,
    
    // Community & Access ✅
    telegram_group: true,
    exclusive_protocols: true,
    early_access_tools: true,
    
    // Premium Services ❌
    in_person_training: false
  },

  longevity: {
    // Core Platform ✅
    personalized_training: true,
    nutrition_plan: true,
    dashboard_type: 'interactive_adjustable',
    
    // Coach Access
    qa_access: 'unlimited',
    response_time_hours: 12,
    
    // Progress Tracking ✅
    weekly_checkins: true,
    form_review: true,
    wearable_sync: true,
    sleep_optimization: true,
    ai_fitness_assistant: true,
    ai_progress_predictors: true,
    
    // Advanced AI Features ✅
    voice_coaching: true,
    video_analysis: true,
    universal_search: true,
    production_monitoring: true,
    quality_assurance: true, // Coach + Longevity access
    
    // Convenience ✅
    auto_grocery_lists: true,
    travel_workouts: true,
    
    // Education ✅
    science_tips: true,
    supplement_guide: true,
    research_blog_access: true,
    
    // Supplementation ✅
    supplement_protocols: true,
    peptides: true,
    peds: true,
    nootropics: true,
    bioregulators: true,
    custom_cycling: true,
    
    // Analytics ✅
    biomarker_integration: true,
    blood_work_review: true,
    recovery_guidance: true,
    hrv_optimization: true,
    
    // Community & Access ✅
    telegram_group: true,
    exclusive_protocols: true,
    early_access_tools: true,
    
    // Premium Services ✅
    in_person_training: '2×/week'
  }
};

// Display names for UI
export const FEATURE_DISPLAY_NAMES: Record<string, string> = {
  personalized_training: 'Personalized Training',
  nutrition_plan: 'Nutrition Plan',
  dashboard_type: 'Dashboard Program',
  qa_access: 'Q&A Access',
  response_time_hours: 'Response Time',
  weekly_checkins: 'Weekly Check-ins & Photos',
  form_review: 'Form Review',
  wearable_sync: 'Wearable Sync',
  sleep_optimization: 'Sleep Optimization',
  ai_fitness_assistant: 'AI Fitness Assistant',
  ai_progress_predictors: 'AI Progress Predictors',
  voice_coaching: 'Voice Coaching System',
  video_analysis: 'Video Analysis & Form Check',
  universal_search: 'Universal Platform Search',
  production_monitoring: 'Performance Monitoring',
  quality_assurance: 'Quality Assurance Access',
  auto_grocery_lists: 'Auto Grocery Lists',
  travel_workouts: 'Travel Workouts',
  science_tips: 'Science Tips',
  supplement_guide: 'Supplement Guide',
  research_blog_access: 'Research Blog Access',
  supplement_protocols: 'Supplement Protocols',
  peptides: 'Peptides',
  peds: 'PEDs',
  nootropics: 'Nootropics',
  bioregulators: 'Bioregulators',
  custom_cycling: 'Custom Cycling',
  biomarker_integration: 'Biomarker Integration',
  blood_work_review: 'Blood Work Review',
  recovery_guidance: 'Recovery Guidance',
  hrv_optimization: 'HRV Optimization',
  telegram_group: 'Telegram Group',
  exclusive_protocols: 'Exclusive Protocols',
  early_access_tools: 'Early Access Tools',
  in_person_training: 'In-Person Training'
};

// Feature categories for organization
export const FEATURE_CATEGORIES = {
  'Core Platform': [
    'personalized_training',
    'nutrition_plan',
    'dashboard_type'
  ],
  'Coach Access': [
    'qa_access',
    'response_time_hours'
  ],
  'Progress Tracking': [
    'weekly_checkins',
    'form_review',
    'wearable_sync',
    'sleep_optimization',
    'ai_fitness_assistant',
    'ai_progress_predictors'
  ],
  'Advanced AI Features': [
    'voice_coaching',
    'video_analysis',
    'universal_search',
    'production_monitoring',
    'quality_assurance'
  ],
  'Convenience': [
    'auto_grocery_lists',
    'travel_workouts'
  ],
  'Education': [
    'science_tips',
    'supplement_guide',
    'research_blog_access'
  ],
  'Supplementation': [
    'supplement_protocols',
    'peptides',
    'peds',
    'nootropics',
    'bioregulators',
    'custom_cycling'
  ],
  'Analytics': [
    'biomarker_integration',
    'blood_work_review',
    'recovery_guidance',
    'hrv_optimization'
  ],
  'Community & Access': [
    'telegram_group',
    'exclusive_protocols',
    'early_access_tools'
  ],
  'Premium Services': [
    'in_person_training'
  ]
};

// Helper functions
export function checkFeatureAccess(tierType: TierType, featureKey: keyof TierFeatures): boolean {
  const tierFeatures = TIER_FEATURES[tierType];
  const featureValue = tierFeatures[featureKey];
  
  // Handle boolean features
  if (typeof featureValue === 'boolean') {
    return featureValue;
  }
  
  // Handle string features (considered as "has access")
  if (typeof featureValue === 'string') {
    return true;
  }
  
  return false;
}

export function getFeatureLevel(tierType: TierType, featureKey: keyof TierFeatures): string | number | boolean {
  return TIER_FEATURES[tierType][featureKey];
}

export function getFeatureDisplayValue(tierType: TierType, featureKey: keyof TierFeatures): string {
  const value = getFeatureLevel(tierType, featureKey);
  
  if (typeof value === 'boolean') {
    return value ? '✅' : '❌';
  }
  
  if (typeof value === 'string') {
    return value;
  }
  
  if (typeof value === 'number') {
    if (featureKey === 'response_time_hours') {
      return `${value} hr`;
    }
    return value.toString();
  }
  
  return '❌';
}

export function hasMinimumTierAccess(userTier: TierType, requiredTier: TierType): boolean {
  const tierHierarchy: TierType[] = ['core', 'adaptive', 'performance', 'longevity'];
  const userIndex = tierHierarchy.indexOf(userTier);
  const requiredIndex = tierHierarchy.indexOf(requiredTier);
  
  return userIndex >= requiredIndex;
}

// Legacy compatibility functions
export function getTierAccessLevel(userTier: TierType): number {
  const tierLevels = {
    'core': 0,
    'adaptive': 1,
    'performance': 2,
    'longevity': 3
  };
  
  return tierLevels[userTier] ?? -1;
}

// Helper function for backward compatibility
export const hasFeatureAccess = (userTier: string, feature: string): boolean => {
  const tierFeatures = TIER_FEATURES[userTier as TierType];
  if (!tierFeatures) return false;
  
  const value = tierFeatures[feature as keyof TierFeatures];
  if (typeof value === 'boolean') return value;
  if (typeof value === 'number') return value > 0;
  if (typeof value === 'string') return value !== 'none' && value.length > 0;
  return value !== null;
};

// Export feature variables for validation and forms
export const FEATURE_VARIABLES = Object.keys(FEATURE_DISPLAY_NAMES);

// Export for backward compatibility
export default TIER_FEATURES;