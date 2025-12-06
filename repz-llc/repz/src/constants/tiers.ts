// REPZ COACH PRO - TIER SYSTEM CONFIGURATION
// Single source of truth for all tier-related data

// Core tier type definition
export const TIER_TYPES = ['core', 'adaptive', 'performance', 'longevity'] as const;
export type TierType = typeof TIER_TYPES[number];

// Billing cycle options
export const BILLING_CYCLES = ['monthly', 'quarterly', 'semiannual', 'annual'] as const;
export type BillingCycle = typeof BILLING_CYCLES[number];

// Core tier configuration interface
export interface TierConfig {
  id: TierType;
  displayName: string;
  tagline: string;
  description: string;
  monthlyPrice: number;
  quarterlyPrice: number;
  semiannualPrice: number;
  annualPrice: number;
  features: string[];
  limitedFeatures: string[];
  color: string;
  gradient: string;
  badge?: string;
  isPopular?: boolean;
  isLimited?: boolean;
  ctaText: string;
  savings: {
    quarterly: number;
    semiannual: number;
    annual: number;
  };
  
  // Legacy compatibility properties
  name: string;
  monthly_price: number;
  charm_pricing: {
    monthly: number;
    quarterly: number;
    semiannual: number;
    annual: number;
  };
  conversion_tactics: {
    anchoring: boolean;
    urgency: boolean;
    socialProof: boolean;
    badge?: string;
    scarcity?: {
      spotsLeft: number;
      totalSpots: number;
      waitlistLength: number;
    };
    valueProps?: string[];
  };
  price: number;
  period: string;
  cta: string;
  popular?: boolean;
  variant: string;
  icon: string;
  priceDisplay: string;
  deliveryMethod: string;
  deliveryDescription: string;
  colors: {
    primary: string;
    background: string;
    text: string;
    accent: string;
    border: string;
    badge: string;
  };
}

// MASTER TIER CONFIGURATIONS - Single Source of Truth
export const TIER_CONFIGS: TierConfig[] = [
  {
    id: 'core',
    displayName: 'Core Program',
    tagline: 'Build Your Foundation',
    description: 'Essential coaching with macro-based training and fundamental nutrition guidance.',
    monthlyPrice: 89,
    quarterlyPrice: 85, // Per month when billed quarterly (5% discount)
    semiannualPrice: 80, // Per month when billed semi-annually (10% discount)  
    annualPrice: 71, // Per month when billed annually (20% discount)
    features: [
      'Personalized training program',
      'Nutrition plan',
      'Dashboard: Static/Fixed',
      'Q&A Access (Limited)',
      'Science-Based Tips (daily/weekly)',
      '72h response time'
    ],
    limitedFeatures: [],
    color: 'hsl(211, 100%, 50%)', // Trust Blue
    gradient: 'from-blue-500 to-blue-600',
    ctaText: 'Start Core Program',
    savings: {
      quarterly: 5,
      semiannual: 10,
      annual: 20
    },
    // Legacy compatibility properties
    name: 'core',
    monthly_price: 89,
    charm_pricing: {
      monthly: 89,
      quarterly: 254,
      semiannual: 482,
      annual: 856
    },
    conversion_tactics: {
      anchoring: false,
      urgency: false,
      socialProof: false
    },
    price: 89,
    period: 'month',
    cta: 'Start Core Program',
    popular: false,
    variant: 'default',
    icon: 'Zap',
    priceDisplay: '$89',
    deliveryMethod: 'Dashboard',
    deliveryDescription: 'Essential coaching with macro-based training and fundamental nutrition guidance.',
    colors: {
      primary: 'hsl(211, 100%, 50%)',
      background: '#000000',
      text: '#FFFFFF',
      accent: 'hsl(211, 100%, 50%)',
      border: 'hsl(211, 100%, 50%)',
      badge: 'hsl(211, 100%, 50%)'
    }
  },
  {
    id: 'adaptive',
    displayName: 'Adaptive Engine',
    tagline: 'Intelligent Optimization',
    description: 'AI-powered adaptations with biomarker integration and personalized tracking.',
    monthlyPrice: 149,
    quarterlyPrice: 142, // Per month when billed quarterly (5% discount)
    semiannualPrice: 134, // Per month when billed semi-annually (10% discount)
    annualPrice: 119, // Per month when billed annually (20% discount)
    features: [
      'Everything in Core +',
      'Dashboard: Interactive/Adjustable',
      'Q&A Access (Standard)',
      'Weekly check-ins & photos',
      'Workout form reviews',
      'Wearable device integration',
      'Sleep & recovery optimization',
      'Auto grocery lists',
      'Travel workout generator',
      'Supplement guide & protocols',
      'Research blog access',
      'Peptides protocols',
      'Biomarker integration',
      'Blood work review',
      'Recovery guidance',
      'Bioregulators protocols',
      'Private Telegram group',
      'Exclusive protocols',
      '48h response time'
    ],
    limitedFeatures: ['Interactive dashboard', 'Weekly check-ins', 'Form reviews', 'Wearable sync', 'Sleep optimization', 'Auto grocery lists', 'Travel workouts', 'Supplement guide', 'Research blog', 'Peptides', 'Biomarker integration', 'Blood work review', 'Recovery guidance', 'Bioregulators', 'Telegram group', 'Exclusive protocols'],
    color: 'hsl(25, 95%, 53%)', // REPZ Orange
    gradient: 'from-orange-500 to-red-500',
    badge: 'NEW FEATURES',
    ctaText: 'Upgrade to Adaptive',
    savings: {
      quarterly: 5,
      semiannual: 10,
      annual: 20
    },
    // Legacy compatibility properties
    name: 'adaptive',
    monthly_price: 149,
    charm_pricing: {
      monthly: 149,
      quarterly: 425,
      semiannual: 805,
      annual: 1432
    },
    conversion_tactics: {
      anchoring: false,
      urgency: false,
      socialProof: true,
      badge: 'NEW FEATURES',
      valueProps: ['NEW: Biomarker tracking at this tier!', 'Auto grocery lists save 2 hours/week']
    },
    price: 149,
    period: 'month',
    cta: 'Upgrade to Adaptive',
    popular: false,
    variant: 'premium',
    icon: 'Award',
    priceDisplay: '$149',
    deliveryMethod: 'Interactive Dashboard',
    deliveryDescription: 'AI-powered adaptations with biomarker integration and personalized tracking.',
    colors: {
      primary: 'hsl(25, 95%, 53%)',
      background: '#1E1E1E',
      text: '#FFFFFF',
      accent: 'hsl(25, 95%, 53%)',
      border: 'hsl(25, 95%, 53%)',
      badge: 'hsl(25, 95%, 53%)'
    }
  },
  {
    id: 'performance',
    displayName: 'Prime Suite',
    tagline: 'Elite Optimization',
    description: 'Advanced AI assistant with form analysis and performance enhancement protocols.',
    monthlyPrice: 229,
    quarterlyPrice: 218, // Per month when billed quarterly (5% discount)
    semiannualPrice: 206, // Per month when billed semi-annually (10% discount)
    annualPrice: 183, // Per month when billed annually (20% discount)
    features: [
      'Everything in Adaptive +',
      'Q&A Access (Priority)',
      'AI fitness assistant',
      'Real-time form analysis',
      'AI progress predictors',
      'PEDs protocols',
      'Nootropics & productivity',
      'Custom cycling schemes',
      'HRV optimization',
      'Early access tools',
      '24h response time'
    ],
    limitedFeatures: ['Priority Q&A', 'AI assistant', 'Form analysis', 'AI predictors', 'PEDs protocols', 'Nootropics', 'Custom cycling', 'HRV optimization', 'Early access'],
    color: 'hsl(258, 90%, 66%)', // Sophistication Purple
    gradient: 'from-purple-500 to-indigo-600',
    badge: 'MOST POPULAR',
    isPopular: true,
    ctaText: 'Unlock Performance',
    savings: {
      quarterly: 5,
      semiannual: 10,
      annual: 20
    },
    // Legacy compatibility properties
    name: 'performance',
    monthly_price: 229,
    charm_pricing: {
      monthly: 229,
      quarterly: 653,
      semiannual: 1241,
      annual: 2200
    },
    conversion_tactics: {
      anchoring: true,
      urgency: false,
      socialProof: true,
      badge: 'MOST POPULAR',
      valueProps: ['NEW: AI assistant included', 'PEDs protocols now available', 'Most chosen by athletes']
    },
    price: 229,
    period: 'month',
    cta: 'Unlock Performance',
    popular: true,
    variant: 'premium',
    icon: 'Zap',
    priceDisplay: '$229',
    deliveryMethod: 'Advanced Dashboard + AI',
    deliveryDescription: 'Advanced AI assistant with form analysis and performance enhancement protocols.',
    colors: {
      primary: 'hsl(258, 90%, 66%)',
      background: '#2B0F4A',
      text: '#FFFFFF',
      accent: 'hsl(258, 90%, 66%)',
      border: 'hsl(258, 90%, 66%)',
      badge: 'hsl(258, 90%, 66%)'
    }
  },
  {
    id: 'longevity',
    displayName: 'Elite Concierge',
    tagline: 'Ultimate Optimization',
    description: 'Exclusive bioregulators, in-person training, and white-glove concierge service.',
    monthlyPrice: 349,
    quarterlyPrice: 332, // Per month when billed quarterly (5% discount)
    semiannualPrice: 314, // Per month when billed semi-annually (10% discount)
    annualPrice: 279, // Per month when billed annually (20% discount)
    features: [
      'Everything in Performance +',
      'Q&A Access (Unlimited)',
      'In-person training: 2×/week',
      '12h response time'
    ],
    limitedFeatures: ['Unlimited Q&A', 'In-person training'],
    color: 'hsl(45, 93%, 58%)', // Luxury Gold
    gradient: 'from-yellow-400 to-orange-400',
    badge: 'LIMITED',
    isLimited: true,
    ctaText: 'Request Longevity Access',
    savings: {
      quarterly: 5,
      semiannual: 10,
      annual: 20
    },
    // Legacy compatibility properties
    name: 'longevity',
    monthly_price: 349,
    charm_pricing: {
      monthly: 349,
      quarterly: 995,
      semiannual: 1881,
      annual: 3349
    },
    conversion_tactics: {
      anchoring: false,
      urgency: true,
      socialProof: false,
      badge: 'LIMITED',
      scarcity: {
        spotsLeft: 5,
        totalSpots: 5,
        waitlistLength: 23
      },
      valueProps: ['Only tier with bioregulators', 'In-person training included', 'Early access to longevity tools']
    },
    price: 349,
    period: 'month',
    cta: 'Request Longevity Access',
    popular: false,
    variant: 'enterprise',
    icon: 'Crown',
    priceDisplay: '$349',
    deliveryMethod: 'Concierge Service',
    deliveryDescription: 'Exclusive bioregulators, in-person training, and white-glove concierge service.',
    colors: {
      primary: 'hsl(45, 93%, 58%)',
      background: '#000000',
      text: '#D4AF37',
      accent: 'hsl(45, 93%, 58%)',
      border: 'hsl(45, 93%, 58%)',
      badge: 'hsl(45, 93%, 58%)'
    }
  }
];

// CONVERSION-OPTIMIZED TIER MAPPING
export const CONVERSION_OPTIMIZED_TIERS: Record<TierType, TierConfig> = {
  core: TIER_CONFIGS[0],
  adaptive: TIER_CONFIGS[1],
  performance: TIER_CONFIGS[2],
  longevity: TIER_CONFIGS[3]
};

// =============================================================================
// UTILITY FUNCTIONS - Single, Clear Implementation
// =============================================================================

/**
 * Get tier configuration by string name (with normalization and fallbacks)
 * @param tierName - String tier name (flexible input)
 * @returns TierConfig object
 */
export function getTierConfig(tierName: string): TierConfig {
  // Handle edge cases and normalization
  if (!tierName || tierName.trim() === '') {
    return TIER_CONFIGS[0]; // Default to core
  }

  // Direct match first
  const config = TIER_CONFIGS.find(tier => 
    tier.displayName.toLowerCase().includes(tierName.toLowerCase()) ||
    tier.id === tierName.toLowerCase()
  );

  if (config) return config;

  // Fallback logic for partial matches
  const normalizedTierName = tierName.toLowerCase().replace(/[^a-z]/g, '');
  
  if (normalizedTierName.includes('core')) {
    return TIER_CONFIGS[0];
  } else if (normalizedTierName.includes('adaptive')) {
    return TIER_CONFIGS[1];
  } else if (normalizedTierName.includes('performance')) {
    return TIER_CONFIGS[2];
  } else if (normalizedTierName.includes('longevity') || normalizedTierName.includes('concierge')) {
    return TIER_CONFIGS[3];
  }

  // Ultimate fallback
  return TIER_CONFIGS[0];
}

/**
 * Get tier configuration by TierType (direct lookup)
 * @param tier - TierType enum value
 * @returns TierConfig object
 */
export function getTierConfigByType(tier: TierType): TierConfig {
  return CONVERSION_OPTIMIZED_TIERS[tier];
}

/**
 * Get tier configuration by ID
 * @param id - Tier ID string
 * @returns TierConfig or undefined
 */
export function getTierById(id: string): TierConfig | undefined {
  return TIER_CONFIGS.find(tier => tier.id === id);
}

/**
 * Get tier key/ID from tier name
 * @param tierName - Tier name string
 * @returns Tier ID string
 */
export function getTierKey(tierName: string): string {
  const config = getTierConfig(tierName);
  return config.id;
}

/**
 * Format tier name for display
 * @param tierName - Raw tier name
 * @returns Formatted tier name
 */
export function formatTierName(tierName: string): string {
  return tierName.replace(/^[^\w\s]+\s*/, '').trim();
}

/**
 * Get all tier configurations
 * @returns Array of all TierConfig objects
 */
export function getAllTierConfigs(): TierConfig[] {
  return TIER_CONFIGS;
}

/**
 * Get tier index in array
 * @param tierName - Tier name string
 * @returns Index number
 */
export function getTierIndex(tierName: string): number {
  const config = getTierConfig(tierName);
  return TIER_CONFIGS.findIndex(tier => tier.id === config.id);
}

/**
 * Validate if tier name is valid
 * @param tierName - Tier name to validate
 * @returns Boolean validity
 */
export function isValidTier(tierName: string): boolean {
  return TIER_CONFIGS.some(tier => 
    tier.id === tierName.toLowerCase() || 
    tier.displayName.toLowerCase() === tierName.toLowerCase()
  );
}

/**
 * Get pricing for specific tier and billing cycle
 * @param tier - TierType
 * @param cycle - BillingCycle
 * @returns Price in cents
 */
export function getTierPrice(tier: TierType, cycle: BillingCycle): number {
  const config = getTierConfigByType(tier);
  switch (cycle) {
    case 'monthly': return config.monthlyPrice * 100;
    case 'quarterly': return config.quarterlyPrice * 100;
    case 'semiannual': return config.semiannualPrice * 100;
    case 'annual': return config.annualPrice * 100;
    default: return config.monthlyPrice * 100;
  }
}

/**
 * Calculate savings percentage for billing cycle
 * @param tier - TierType
 * @param cycle - BillingCycle
 * @returns Savings percentage
 */
export function getTierSavings(tier: TierType, cycle: BillingCycle): number {
  const config = getTierConfigByType(tier);
  return config.savings[cycle as keyof typeof config.savings] || 0;
}

// =============================================================================
// FEATURE ACCESS MATRIX - Clear Tier-Based Features
// =============================================================================

export interface TierFeatures {
  // Core Features (All Tiers)
  basic_training_program: boolean;
  nutrition_guidance: boolean;
  exercise_form_library: boolean;
  progress_tracking: boolean;
  community_access: boolean;
  
  // Adaptive+ Features
  weekly_checkins_access: boolean;
  progress_photo_analysis: boolean;
  wearable_integration: boolean;
  auto_grocery_lists: boolean;
  biomarker_integration: boolean;
  travel_workout_generator: boolean;
  
  // Performance+ Features
  ai_fitness_assistant: boolean;
  form_analysis_ai: boolean;
  ai_progress_predictors: boolean;
  peds_protocols: boolean;
  advanced_biomarkers: boolean;
  custom_cycling_schemes: boolean;
  
  // Longevity Exclusive
  bioregulators_protocols: boolean;
  in_person_training: boolean;
  priority_support: boolean;
  concierge_service: boolean;
  executive_health_assessments: boolean;
  direct_coach_access: boolean;
  
  // Support Response Times
  coach_response_time: number; // hours
}

export const TIER_FEATURES: Record<TierType, TierFeatures> = {
  core: {
    // Core Features
    basic_training_program: true,
    nutrition_guidance: true,
    exercise_form_library: true,
    progress_tracking: true,
    community_access: true,
    
    // Adaptive+ Features
    weekly_checkins_access: false,
    progress_photo_analysis: false,
    wearable_integration: false,
    auto_grocery_lists: false,
    biomarker_integration: false,
    travel_workout_generator: false,
    
    // Performance+ Features
    ai_fitness_assistant: false,
    form_analysis_ai: false,
    ai_progress_predictors: false,
    peds_protocols: false,
    advanced_biomarkers: false,
    custom_cycling_schemes: false,
    
    // Longevity Exclusive
    bioregulators_protocols: false,
    in_person_training: false,
    priority_support: false,
    concierge_service: false,
    executive_health_assessments: false,
    direct_coach_access: false,
    
    coach_response_time: 72
  },
  
  adaptive: {
    // Core Features
    basic_training_program: true,
    nutrition_guidance: true,
    exercise_form_library: true,
    progress_tracking: true,
    community_access: true,
    
    // Adaptive+ Features
    weekly_checkins_access: true,
    progress_photo_analysis: true,
    wearable_integration: true,
    auto_grocery_lists: true,
    biomarker_integration: true,
    travel_workout_generator: true,
    
    // Performance+ Features
    ai_fitness_assistant: false,
    form_analysis_ai: false,
    ai_progress_predictors: false,
    peds_protocols: false,
    advanced_biomarkers: false,
    custom_cycling_schemes: false,
    
    // Longevity Exclusive
    bioregulators_protocols: false,
    in_person_training: false,
    priority_support: false,
    concierge_service: false,
    executive_health_assessments: false,
    direct_coach_access: false,
    
    coach_response_time: 48
  },
  
  performance: {
    // Core Features
    basic_training_program: true,
    nutrition_guidance: true,
    exercise_form_library: true,
    progress_tracking: true,
    community_access: true,
    
    // Adaptive+ Features
    weekly_checkins_access: true,
    progress_photo_analysis: true,
    wearable_integration: true,
    auto_grocery_lists: true,
    biomarker_integration: true,
    travel_workout_generator: true,
    
    // Performance+ Features
    ai_fitness_assistant: true,
    form_analysis_ai: true,
    ai_progress_predictors: true,
    peds_protocols: true,
    advanced_biomarkers: true,
    custom_cycling_schemes: true,
    
    // Longevity Exclusive
    bioregulators_protocols: false,
    in_person_training: false,
    priority_support: false,
    concierge_service: false,
    executive_health_assessments: false,
    direct_coach_access: false,
    
    coach_response_time: 24
  },
  
  longevity: {
    // Core Features
    basic_training_program: true,
    nutrition_guidance: true,
    exercise_form_library: true,
    progress_tracking: true,
    community_access: true,
    
    // Adaptive+ Features
    weekly_checkins_access: true,
    progress_photo_analysis: true,
    wearable_integration: true,
    auto_grocery_lists: true,
    biomarker_integration: true,
    travel_workout_generator: true,
    
    // Performance+ Features
    ai_fitness_assistant: true,
    form_analysis_ai: true,
    ai_progress_predictors: true,
    peds_protocols: true,
    advanced_biomarkers: true,
    custom_cycling_schemes: true,
    
    // Longevity Exclusive
    bioregulators_protocols: true,
    in_person_training: true,
    priority_support: true,
    concierge_service: true,
    executive_health_assessments: true,
    direct_coach_access: true,
    
    coach_response_time: 12
  }
};

// =============================================================================
// HELPER FUNCTIONS FOR FEATURE ACCESS
// =============================================================================

/**
 * Get features for a specific tier
 * @param tier - TierType
 * @returns TierFeatures object
 */
export function getTierFeatures(tier: TierType): TierFeatures {
  return TIER_FEATURES[tier];
}

/**
 * Check if a tier has access to a specific feature
 * @param tier - TierType
 * @param feature - Feature key
 * @returns Boolean access
 */
export function hasFeatureAccess(tier: TierType, feature: keyof TierFeatures): boolean {
  return TIER_FEATURES[tier][feature] as boolean;
}

/**
 * Get minimum tier required for a feature
 * @param feature - Feature key
 * @returns TierType or null if feature doesn't exist
 */
export function getMinimumTierForFeature(feature: keyof TierFeatures): TierType | null {
  for (const tier of TIER_TYPES) {
    if (TIER_FEATURES[tier][feature]) {
      return tier;
    }
  }
  return null;
}

// =============================================================================
// LEGACY COMPATIBILITY EXPORTS
// =============================================================================

// Legacy helper functions
export function getTierHierarchy(): TierType[] {
  return ['core', 'adaptive', 'performance', 'longevity'];
}

export function isHigherTier(userTier: TierType, comparisonTier: TierType): boolean {
  const hierarchy = getTierHierarchy();
  const userIndex = hierarchy.indexOf(userTier);
  const comparisonIndex = hierarchy.indexOf(comparisonTier);
  return userIndex > comparisonIndex;
}

export function calculateSavings(tierType: TierType, billingCycle: BillingCycle): number {
  const config = getTierConfigByType(tierType);
  const monthlyPrice = config.monthlyPrice;
  
  switch (billingCycle) {
    case 'quarterly':
      return monthlyPrice * 3 - config.quarterlyPrice;
    case 'semiannual':
      return monthlyPrice * 6 - config.semiannualPrice;
    case 'annual':
      return monthlyPrice * 12 - config.annualPrice;
    default:
      return 0;
  }
}

// Legacy exports with extended TierConfig interface
export const TIER_CONFIGS_LEGACY = TIER_CONFIGS.map(tier => ({
  ...tier,
  // Legacy compatibility properties
  name: tier.id,
  monthly_price: tier.monthlyPrice,
  charm_pricing: {
    monthly: tier.monthlyPrice,
    quarterly: tier.quarterlyPrice,
    semiannual: tier.semiannualPrice,
    annual: tier.annualPrice
  },
  conversion_tactics: {
    anchoring: tier.isPopular || false,
    urgency: tier.badge === 'LIMITED',
    socialProof: tier.isPopular || false,
    badge: tier.badge
  },
  price: tier.monthlyPrice,
  period: 'month',
  cta: tier.ctaText,
  popular: tier.isPopular,
  variant: tier.isPopular ? 'premium' : 'default',
  icon: 'Zap', // Default icon reference
  priceDisplay: `$${tier.monthlyPrice}`,
  deliveryMethod: 'Dashboard',
  deliveryDescription: tier.description,
  colors: {
    primary: tier.color,
    background: '#000000',
    text: '#FFFFFF',
    accent: tier.color,
    border: tier.color,
    badge: tier.color
  }
}));

// Legacy constants
export const TIER_COLORS = {
  core: 'hsl(211, 100%, 50%)',
  adaptive: 'hsl(25, 95%, 53%)',
  performance: 'hsl(258, 90%, 66%)',
  longevity: 'hsl(45, 93%, 58%)'
};

export const TIER_NAMES = {
  CORE: 'Core Program',
  ADAPTIVE: 'Adaptive Engine', 
  PERFORMANCE: 'Prime Suite',
  LONGEVITY: 'Elite Concierge'
} as const;

export const TIER_PRICING = {
  CORE: {
    monthly: 89,
    quarterly: 254,
    semiannual: 482,
    annual: 856
  },
  ADAPTIVE: {
    monthly: 149,
    quarterly: 425,
    semiannual: 805,
    annual: 1432
  },
  PERFORMANCE: {
    monthly: 229,
    quarterly: 653,
    semiannual: 1241,
    annual: 2200
  },
  LONGEVITY: {
    monthly: 349,
    quarterly: 995,
    semiannual: 1881,
    annual: 3349
  }
};

// Export featureMatrix functions
export const checkFeatureAccess = hasFeatureAccess;
export const getFeatureLevel = hasFeatureAccess;
export const getFeatureDisplayValue = (tier: TierType, feature: keyof TierFeatures) => 
  hasFeatureAccess(tier, feature);
export const hasMinimumTierAccess = (userTier: TierType, requiredTier: TierType) => 
  isHigherTier(userTier, requiredTier) || userTier === requiredTier;

export const FEATURE_DISPLAY_NAMES = {};
export const FEATURE_CATEGORIES = {};

// Export default configuration
export default {
  TIER_TYPES,
  TIER_CONFIGS,
  TIER_FEATURES,
  getTierConfig,
  getTierConfigByType,
  getTierFeatures,
  hasFeatureAccess
};