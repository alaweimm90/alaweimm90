/**
 * Analytics Configuration
 * Central configuration for all analytics providers and settings
 */

import { AnalyticsConfig } from '@/lib/analytics';

// ============================================================================
// Environment Variables
// ============================================================================

const GA4_MEASUREMENT_ID = import.meta.env.VITE_GA_MEASUREMENT_ID as string;
const GTM_ID = import.meta.env.VITE_GTM_ID as string;
const PLAUSIBLE_DOMAIN = import.meta.env.VITE_PLAUSIBLE_DOMAIN as string;
const DEBUG_MODE = import.meta.env.DEV || (import.meta.env.VITE_ANALYTICS_DEBUG === 'true');

// ============================================================================
// Analytics Configuration
// ============================================================================

export const analyticsConfig: AnalyticsConfig = {
  ga4MeasurementId: GA4_MEASUREMENT_ID,
  gtmId: GTM_ID,
  plausibleDomain: PLAUSIBLE_DOMAIN,
  debugMode: DEBUG_MODE,
  consentRequired: true,
  providers: ['ga4'],
};

// ============================================================================
// Event Naming Conventions
// ============================================================================

/**
 * GA4 Event Naming Convention Guide
 * - Use snake_case for all event names
 * - Maximum 40 characters
 * - Use predefined GA4 events when possible
 * - Custom events should be prefixed with app_
 */

export const EventNames = {
  // Navigation
  PAGE_VIEW: 'page_view',
  SCROLL: 'scroll',
  CLICK: 'click',

  // E-commerce
  VIEW_ITEM: 'view_item',
  VIEW_ITEMS: 'view_items',
  SELECT_ITEM: 'select_item',
  ADD_TO_CART: 'add_to_cart',
  REMOVE_FROM_CART: 'remove_from_cart',
  VIEW_CART: 'view_cart',
  BEGIN_CHECKOUT: 'begin_checkout',
  ADD_SHIPPING_INFO: 'add_shipping_info',
  ADD_PAYMENT_INFO: 'add_payment_info',
  PURCHASE: 'purchase',
  REFUND: 'refund',

  // User Engagement
  SEARCH: 'search',
  SHARE: 'share',
  SIGN_UP: 'sign_up',
  LOGIN: 'login',
  FORM_SUBMIT: 'form_submit',
  FILE_DOWNLOAD: 'file_download',

  // Custom Events
  SCROLL_DEPTH: 'scroll_depth',
  TIME_ON_PAGE: 'time_on_page',
  EXIT_INTENT: 'exit_intent',
  ENGAGEMENT_STARTED: 'engagement_started',
  ENGAGEMENT_ENDED: 'engagement_ended',
  PAGE_ENGAGEMENT: 'page_engagement',
  BOUNCE: 'bounce',
  LONG_TASK: 'long_task',
  SLOW_API_CALL: 'slow_api_call',
  API_ERROR: 'api_error',
  ERROR_LOGGED: 'error_logged',
  SESSION_REPLAY_TRIGGERED: 'session_replay_triggered',
} as const;

// ============================================================================
// User Properties & Dimensions
// ============================================================================

export const UserProperties = {
  // User Tier
  USER_TIER: 'user_tier',
  MEMBERSHIP_LEVEL: 'membership_level',

  // Behavior
  LIFETIME_VALUE: 'lifetime_value',
  PURCHASE_FREQUENCY: 'purchase_frequency',
  AVERAGE_ORDER_VALUE: 'average_order_value',

  // Device
  DEVICE_TYPE: 'device_type',
  OS_TYPE: 'os_type',
  BROWSER_TYPE: 'browser_type',

  // Location
  COUNTRY: 'country',
  CITY: 'city',
  REGION: 'region',

  // Preferences
  PREFERRED_CATEGORY: 'preferred_category',
  MARKETING_CONSENT: 'marketing_consent',
  ANALYTICS_CONSENT: 'analytics_consent',
} as const;

// ============================================================================
// Content Grouping
// ============================================================================

export const ContentGroups = {
  HOME: 'home',
  SHOP: 'shop',
  PRODUCT_DETAIL: 'product_detail',
  CART: 'cart',
  CHECKOUT: 'checkout',
  ACCOUNT: 'account',
  ABOUT: 'about',
  CONTACT: 'contact',
  ADMIN: 'admin',
} as const;

// ============================================================================
// Conversion Goals
// ============================================================================

export const ConversionGoals = {
  PURCHASE: 'purchase',
  EMAIL_SIGNUP: 'email_signup',
  CONTACT_FORM: 'contact_form',
  ACCOUNT_CREATION: 'account_creation',
  WISHLIST_ADD: 'wishlist_add',
} as const;

// ============================================================================
// Error Categories
// ============================================================================

export const ErrorCategories = {
  JAVASCRIPT_ERROR: 'javascript_error',
  NETWORK_ERROR: 'network_error',
  API_ERROR: 'api_error',
  VALIDATION_ERROR: 'validation_error',
  PAYMENT_ERROR: 'payment_error',
  AUTH_ERROR: 'auth_error',
} as const;

// ============================================================================
// Performance Thresholds
// ============================================================================

export const PerformanceThresholds = {
  // Web Vitals thresholds (milliseconds)
  LCP_GOOD: 2500, // Largest Contentful Paint
  LCP_NEEDS_IMPROVEMENT: 4000,

  FID_GOOD: 100, // First Input Delay
  FID_NEEDS_IMPROVEMENT: 300,

  CLS_GOOD: 0.1, // Cumulative Layout Shift
  CLS_NEEDS_IMPROVEMENT: 0.25,

  // API performance
  SLOW_API_THRESHOLD: 2000, // 2 seconds
  TIMEOUT_THRESHOLD: 30000, // 30 seconds

  // Resource loading
  SLOW_RESOURCE_THRESHOLD: 1000, // 1 second
} as const;

// ============================================================================
// E-commerce Constants
// ============================================================================

export const EcommerceConstants = {
  CURRENCY: 'USD',
  DEFAULT_SHIPPING_COST: 9.99,
  FREE_SHIPPING_THRESHOLD: 100,
} as const;

// ============================================================================
// Privacy & Compliance
// ============================================================================

export const PrivacyConfig = {
  // PII patterns to block
  PII_PATTERNS: [
    /email/i,
    /phone/i,
    /ssn/i,
    /credit/i,
    /password/i,
    /token/i,
    /apikey/i,
    /secret/i,
    /address/i,
    /country/i,
    /state/i,
    /city/i,
  ],

  // GDPR Settings
  GDPR_ENABLED: true,
  COOKIE_CONSENT_EXPIRY_DAYS: 365,
  DATA_RETENTION_DAYS: 90,

  // CCPA Settings
  CCPA_ENABLED: true,

  // Cookie Flags
  COOKIE_FLAGS: 'SameSite=Strict; Secure; HttpOnly',
} as const;

// ============================================================================
// Scroll Tracking
// ============================================================================

export const ScrollTrackingConfig = {
  THRESHOLDS: [25, 50, 75, 100] as const,
  THROTTLE_MS: 100,
} as const;

// ============================================================================
// User Behavior Tracking
// ============================================================================

export const UserBehaviorConfig = {
  TRACK_CLICKS: true,
  TRACK_FOCUS: true,
  TRACK_MOUSE_MOVEMENT: false, // Privacy-conscious default
  TRACK_KEYBOARD: false, // Privacy-conscious default
  TRACK_VISIBILITY: true,
  TRACK_EXIT_INTENT: true,
  ENGAGEMENT_TIMEOUT_MS: 30000, // 30 seconds
  THROTTLE_MS: 200,
} as const;

// ============================================================================
// Session Configuration
// ============================================================================

export const SessionConfig = {
  SESSION_TIMEOUT_MS: 1800000, // 30 minutes
  SESSION_ID_LENGTH: 32,
  MAX_BREADCRUMBS: 50,
  MAX_ERROR_LOGS: 100,
} as const;

// ============================================================================
// Debug Configuration
// ============================================================================

export const DebugConfig = {
  LOG_ANALYTICS_EVENTS: DEBUG_MODE,
  LOG_PERFORMANCE_METRICS: DEBUG_MODE,
  LOG_ERRORS: DEBUG_MODE,
  STORE_EVENTS_LOCALLY: DEBUG_MODE,
  MAX_STORED_EVENTS: 100,
  SHOW_CONSENT_BANNER: DEBUG_MODE,
} as const;

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get the appropriate error severity based on error type
 */
export const getErrorSeverity = (errorType: string): 'low' | 'medium' | 'high' | 'critical' => {
  const criticalErrors = ['fatal_error', 'crash', 'unrecoverable'];
  const highErrors = ['api_error', 'network_error', 'auth_error'];
  const mediumErrors = ['validation_error', 'warning'];

  if (criticalErrors.includes(errorType)) return 'critical';
  if (highErrors.includes(errorType)) return 'high';
  if (mediumErrors.includes(errorType)) return 'medium';
  return 'low';
};

/**
 * Check if an error should trigger session replay
 */
export const shouldTriggerSessionReplay = (errorSeverity: 'low' | 'medium' | 'high' | 'critical'): boolean => {
  return errorSeverity === 'critical' || errorSeverity === 'high';
};

/**
 * Get content group from current pathname
 */
export const getContentGroupFromPath = (pathname: string): string => {
  if (pathname === '/') return ContentGroups.HOME;
  if (pathname.startsWith('/shop')) return ContentGroups.SHOP;
  if (pathname.startsWith('/product')) return ContentGroups.PRODUCT_DETAIL;
  if (pathname === '/cart') return ContentGroups.CART;
  if (pathname.startsWith('/checkout')) return ContentGroups.CHECKOUT;
  if (pathname.startsWith('/profile') || pathname.startsWith('/account')) return ContentGroups.ACCOUNT;
  if (pathname === '/about') return ContentGroups.ABOUT;
  if (pathname === '/contact') return ContentGroups.CONTACT;
  if (pathname.startsWith('/admin')) return ContentGroups.ADMIN;
  return 'other';
};

export default analyticsConfig;
