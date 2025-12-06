// Business logic types - Core business entities and rules
// CRITICAL: These types define the core business model from SUPERPROMPT_LOVEABLE.md

// Core business entity types - CANONICAL TIER IDs
export type TierType = 'core' | 'adaptive' | 'performance' | 'longevity';
export type UserRole = 'client' | 'coach' | 'admin';
export type BillingCycle = 'monthly' | 'quarterly' | 'semiannual' | 'annual';

// User and profile types
export interface UserProfile {
  id: string;
  email: string;
  role: UserRole;
  tier?: TierType;
  created_at: Date;
  updated_at: Date;
  last_login?: Date;
  is_active: boolean;
}

export interface CoachProfile extends UserProfile {
  role: 'coach';
  specializations: string[];
  certifications: string[];
  bio?: string;
  years_experience: number;
  clients_assigned: string[];
}

export interface ClientProfile extends UserProfile {
  role: 'client';
  tier: TierType;
  coach_id?: string;
  goals: string[];
  current_program?: string;
  subscription_id?: string;
}

// Subscription and billing types
export interface SubscriptionDetails {
  id: string;
  user_id: string;
  tier: TierType;
  billing_cycle: BillingCycle;
  status: 'active' | 'cancelled' | 'past_due' | 'paused';
  current_period_start: Date;
  current_period_end: Date;
  stripe_subscription_id?: string;
  created_at: Date;
  updated_at: Date;
}

// Pricing structure from SUPERPROMPT_LOVEABLE.md
export interface TierPricing {
  monthly: number;
  quarterly: number; // 5% discount
  semiannual: number; // 10% discount
  annual: number; // 20% discount
}

// Feature and access types - 56+ Feature Matrix from COMPREHENSIVE_DASHBOARD_SPECIFICATION.md
export interface TierFeatures {
  // Core Platform (All Tiers)
  personalizedTraining: boolean;
  nutritionPlan: boolean;
  progressPhotos: boolean;
  scienceTips: boolean;
  qaAccess: boolean;
  responseTimeHours: 12 | 24 | 48 | 72;
  
  // Progress Tracking (Adaptive+)
  weeklyCheckins: boolean;
  formReview: boolean;
  wearableSync: boolean;
  sleepOptimization: boolean;
  
  // Convenience Features (Adaptive+)
  autoGroceryLists: boolean;
  travelWorkouts: boolean;
  
  // Education (Adaptive+)
  supplementGuide: boolean;
  researchBlogAccess: boolean;
  
  // Supplementation (Adaptive+)
  supplementProtocols: boolean;
  peptides: boolean;
  bioregulators: boolean; // CRITICAL: Available at Adaptive tier
  
  // Analytics (Adaptive+)
  biomarkerIntegration: boolean;
  bloodWorkReview: boolean;
  recoveryGuidance: boolean;
  hrvOptimization: boolean;
  
  // Community (Adaptive+)
  telegramGroup: boolean;
  exclusiveProtocols: boolean;
  
  // AI Features (Performance+)
  aiAssistant: boolean;
  aiProgressPredictors: boolean;
  voiceCoaching: boolean;
  videoAnalysis: boolean;
  
  // Advanced Supplementation (Performance+)
  pedsProtocols: boolean;
  nootropics: boolean;
  customCycling: boolean;
  
  // Advanced Tools (Performance+)
  universalSearch: boolean;
  productionMonitoring: boolean;
  earlyAccess: boolean;
  
  // Exclusive Services (Longevity Only)
  inPersonTraining: boolean;
  unlimitedQA: boolean;
  
  // Dashboard Type
  dashboardType: 'static_fixed' | 'interactive_adjustable';
}

export interface TierConfig {
  id: TierType;
  name: string;
  displayName: string;
  description: string;
  price: TierPricing;
  features: TierFeatures;
  popular?: boolean;
  cta: string;
  color: string;
  gradient: string;
}

// Access control types
export interface AccessControl {
  requiredTier: TierType;
  requiredRole?: UserRole;
  requiredSubscriptionStatus?: SubscriptionDetails['status'];
}

export interface FeatureAccess {
  feature: keyof TierFeatures;
  hasAccess: boolean;
  upgradeRequired?: TierType;
  reason?: string;
}

// Business rule types
export interface TierUpgrade {
  from: TierType;
  to: TierType;
  priceDifference: number;
  effectiveDate: Date;
  prorationAmount: number;
}

export interface TierDowngrade {
  from: TierType;
  to: TierType;
  effectiveDate: Date;
  retainedFeatures: string[];
  lostFeatures: string[];
}

// Analytics and metrics types
export interface UserMetrics {
  user_id: string;
  tier: TierType;
  engagement_score: number;
  feature_usage: Record<string, number>;
  session_duration: number;
  last_activity: Date;
}

export interface BusinessMetrics {
  total_users: number;
  users_by_tier: Record<TierType, number>;
  monthly_recurring_revenue: number;
  churn_rate: number;
  upgrade_rate: number;
  average_revenue_per_user: number;
}

// Validation and constraints
export const TIER_HIERARCHY: Record<TierType, number> = {
  'core': 1,
  'adaptive': 2,
  'performance': 3,
  'longevity': 4
};

export const VALID_BILLING_CYCLES: BillingCycle[] = [
  'monthly',
  'quarterly', 
  'semiannual',
  'annual'
];

export const VALID_USER_ROLES: UserRole[] = [
  'client',
  'coach', 
  'admin'
];

export const VALID_TIER_TYPES: TierType[] = [
  'core',
  'adaptive',
  'performance',
  'longevity'
];