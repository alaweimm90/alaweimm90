/**
 * REPZ Services Index
 * Central export for all backend services
 */

// Core Services
export { intakeService } from './intakeService';
export type {
  IntakeFormData,
  IntakeSubmission,
  PersonalInfo,
  FitnessAssessment,
  HealthHistory,
  TrainingExperience,
  Goals,
  Nutrition,
  Lifestyle,
  Schedule,
  Consent,
} from './intakeService';

export { subscriptionService, TIER_FEATURES, TIER_PRICING } from './subscriptionService';
export type {
  Subscription,
  SubscriptionTier,
  BillingPeriod,
  SubscriptionStatus,
  TierFeatures,
} from './subscriptionService';

export { userService } from './userService';
export type {
  User,
  UserRole,
  UserProfile,
  ClientProfile,
  CoachProfile,
} from './userService';

export { onboardingService } from './onboardingService';
export type {
  OnboardingProgress,
  OnboardingStatus,
  OnboardingStep,
} from './onboardingService';
