// Core fitness coaching platform types

export type TierType = 'core' | 'adaptive' | 'performance' | 'longevity';
export type UserRole = 'coach' | 'client';
export type GenderType = 'male' | 'female';
export type GoalType = 'fat_loss' | 'muscle_gain' | 'maintenance' | 'recomposition' | 'performance';
export type SessionType = 'upper' | 'lower' | 'push' | 'pull' | 'legs' | 'cardio' | 'rest';

export interface TierFeatures {
  // Core features (camelCase for TypeScript compatibility)
  progressPhotos: boolean;
  nutritionTracking: boolean; 
  customWorkouts: boolean;
  coachAccess: boolean;
  weeklyCheckins: boolean;
  formAnalysis: boolean;
  bodyCompositionTracking: boolean;
  injuryPrevention: boolean;
  performanceTracking: boolean;
  hrv_optimization: boolean;
  biomarker_integration: boolean;
  enhanced_analytics: boolean;
  ai_coaching: boolean;
  live_coaching: boolean;
  
  // Legacy features (for backward compatibility)
  dashboard_type?: 'static_fixed' | 'interactive_adjustable' | 'advanced' | 'premium';
  weekly_checkin?: boolean;
  workout_reviews?: boolean;
  science_tips?: boolean;
  supplements_protocol?: boolean;
  peptides_protocol?: boolean;
  peds_protocol?: boolean;
  bloodwork_interpretation?: boolean;
  educational_materials?: boolean;
  response_time_hours?: 12 | 24 | 48 | 72;
  
  // Additional features
  videoAnalysis?: boolean;
  habitTracking?: boolean;
}

// Database Json type compatibility
export type DatabaseTierFeatures = Record<string, unknown>;

export interface SubscriptionTier {
  id: string;
  tier_name: TierType;
  display_name: string;
  price_cents: number;
  stripe_price_id?: string;
  features: TierFeatures;
  is_popular: boolean;
  is_limited: boolean;
  max_clients?: number;
  description: string;
  created_at: string;
}

export interface CoachProfile {
  id: string;
  auth_user_id: string;
  coach_name: string;
  credentials: string[];
  specializations: string[];
  max_longevity_clients: number;
  current_longevity_clients: number;
  created_at: string;
  updated_at: string;
}

export interface ClientProfile {
  id: string;
  auth_user_id?: string;
  coach_id?: string;
  subscription_tier: TierType;
  stripe_subscription_id?: string;
  
  // Demographics
  client_name: string;
  age_years?: number;
  sex?: GenderType;
  height_cm?: number;
  
  // Program Configuration
  program_start_date?: string;
  program_end_date?: string;
  current_week: number;
  training_days_per_week: number;
  primary_goal?: GoalType;
  
  // Metabolic Data
  start_weight_kg?: number;
  target_weight_kg?: number;
  body_fat_percentage?: number;
  activity_level: number;
  
  // Calculated Metrics
  bmi?: number;
  rmr_kcal_day?: number;
  tdee_kcal_day?: number;
  lbm_kg?: number;
  
  // Tier Features
  tier_features: DatabaseTierFeatures;
  
  created_at: string;
  updated_at: string;
}

export interface FitnessUser {
  id: string;
  email: string;
  role: UserRole;
  profile: CoachProfile | ClientProfile;
  subscription_tier?: TierType;
  tier_features?: TierFeatures;
}