// Central type exports - Single source of truth for all shared types
// This file serves as the main entry point for all type definitions

// Business types
export type { 
  TierType, 
  UserRole, 
  BillingCycle,
  UserProfile,
  SubscriptionDetails,
  TierFeatures,
  TierConfig
} from './business';

// Component types
export type {
  BaseComponentProps,
  TierCardProps,
  ModalProps,
  FormProps,
  NavigationProps
} from './components';

// API types
export type {
  ApiResponse,
  PaginatedResponse,
  ErrorResponse,
  SuccessResponse
} from './api';

// Utility types
export type {
  Optional,
  RequiredKeys,
  DeepPartial,
  NonEmptyArray,
  Prettify
} from './utilities';

// Type guards
export {
  isTierType,
  isUserRole,
  isBillingCycle,
  isApiResponse
} from './guards';

// Re-export commonly used external types
export type { ComponentProps, ReactNode, PropsWithChildren } from 'react';
export type { Database } from '@/integrations/supabase/types';