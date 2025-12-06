// ============= STRIPE CONFIGURATION - COMPLETE 16 PRICE IDS =============
// Single source of truth for all Stripe price configurations

import { TierType, BillingCycle } from './tiers';

// Re-export BillingCycle for convenience
export type { BillingCycle };

// Production Stripe Price IDs (16 total)
export const STRIPE_PRICES_PROD: Record<TierType, Record<BillingCycle, string>> = {
  core: {
    monthly: process.env.STRIPE_PRICE_CORE_MONTHLY_PROD || 'price_core_monthly_prod',
    quarterly: process.env.STRIPE_PRICE_CORE_QUARTERLY_PROD || 'price_core_quarterly_prod',
    semiannual: process.env.STRIPE_PRICE_CORE_SEMIANNUAL_PROD || 'price_core_semiannual_prod',
    annual: process.env.STRIPE_PRICE_CORE_ANNUAL_PROD || 'price_core_annual_prod'
  },
  adaptive: {
    monthly: process.env.STRIPE_PRICE_ADAPTIVE_MONTHLY_PROD || 'price_adaptive_monthly_prod',
    quarterly: process.env.STRIPE_PRICE_ADAPTIVE_QUARTERLY_PROD || 'price_adaptive_quarterly_prod',
    semiannual: process.env.STRIPE_PRICE_ADAPTIVE_SEMIANNUAL_PROD || 'price_adaptive_semiannual_prod',
    annual: process.env.STRIPE_PRICE_ADAPTIVE_ANNUAL_PROD || 'price_adaptive_annual_prod'
  },
  performance: {
    monthly: process.env.STRIPE_PRICE_PERFORMANCE_MONTHLY_PROD || 'price_performance_monthly_prod',
    quarterly: process.env.STRIPE_PRICE_PERFORMANCE_QUARTERLY_PROD || 'price_performance_quarterly_prod',
    semiannual: process.env.STRIPE_PRICE_PERFORMANCE_SEMIANNUAL_PROD || 'price_performance_semiannual_prod',
    annual: process.env.STRIPE_PRICE_PERFORMANCE_ANNUAL_PROD || 'price_performance_annual_prod'
  },
  longevity: {
    monthly: process.env.STRIPE_PRICE_LONGEVITY_MONTHLY_PROD || 'price_longevity_monthly_prod',
    quarterly: process.env.STRIPE_PRICE_LONGEVITY_QUARTERLY_PROD || 'price_longevity_quarterly_prod',
    semiannual: process.env.STRIPE_PRICE_LONGEVITY_SEMIANNUAL_PROD || 'price_longevity_semiannual_prod',
    annual: process.env.STRIPE_PRICE_LONGEVITY_ANNUAL_PROD || 'price_longevity_annual_prod'
  }
};

// Test Stripe Price IDs (16 total)
export const STRIPE_PRICES_TEST: Record<TierType, Record<BillingCycle, string>> = {
  core: {
    monthly: process.env.STRIPE_PRICE_CORE_MONTHLY_TEST || 'price_test_core_monthly',
    quarterly: process.env.STRIPE_PRICE_CORE_QUARTERLY_TEST || 'price_test_core_quarterly',
    semiannual: process.env.STRIPE_PRICE_CORE_SEMIANNUAL_TEST || 'price_test_core_semiannual',
    annual: process.env.STRIPE_PRICE_CORE_ANNUAL_TEST || 'price_test_core_annual'
  },
  adaptive: {
    monthly: process.env.STRIPE_PRICE_ADAPTIVE_MONTHLY_TEST || 'price_test_adaptive_monthly',
    quarterly: process.env.STRIPE_PRICE_ADAPTIVE_QUARTERLY_TEST || 'price_test_adaptive_quarterly',
    semiannual: process.env.STRIPE_PRICE_ADAPTIVE_SEMIANNUAL_TEST || 'price_test_adaptive_semiannual',
    annual: process.env.STRIPE_PRICE_ADAPTIVE_ANNUAL_TEST || 'price_test_adaptive_annual'
  },
  performance: {
    monthly: process.env.STRIPE_PRICE_PERFORMANCE_MONTHLY_TEST || 'price_test_performance_monthly',
    quarterly: process.env.STRIPE_PRICE_PERFORMANCE_QUARTERLY_TEST || 'price_test_performance_quarterly',
    semiannual: process.env.STRIPE_PRICE_PERFORMANCE_SEMIANNUAL_TEST || 'price_test_performance_semiannual',
    annual: process.env.STRIPE_PRICE_PERFORMANCE_ANNUAL_TEST || 'price_test_performance_annual'
  },
  longevity: {
    monthly: process.env.STRIPE_PRICE_LONGEVITY_MONTHLY_TEST || 'price_test_longevity_monthly',
    quarterly: process.env.STRIPE_PRICE_LONGEVITY_QUARTERLY_TEST || 'price_test_longevity_quarterly',
    semiannual: process.env.STRIPE_PRICE_LONGEVITY_SEMIANNUAL_TEST || 'price_test_longevity_semiannual',
    annual: process.env.STRIPE_PRICE_LONGEVITY_ANNUAL_TEST || 'price_test_longevity_annual'
  }
};

// Environment-based price selection
export const STRIPE_PRICES = process.env.NODE_ENV === 'production' 
  ? STRIPE_PRICES_PROD 
  : STRIPE_PRICES_TEST;

/**
 * Get Stripe price ID for specific tier and billing cycle
 */
export function getStripePrice(tier: TierType, billingCycle: BillingCycle): string {
  return STRIPE_PRICES[tier][billingCycle];
}

/**
 * Validate that all required Stripe price IDs are configured
 */
export function validateStripeConfig(): { isValid: boolean; missingPrices: string[] } {
  const missingPrices: string[] = [];
  
  Object.entries(STRIPE_PRICES).forEach(([tier, cycles]) => {
    Object.entries(cycles).forEach(([cycle, priceId]) => {
      if (!priceId || priceId.startsWith('price_') && priceId.includes('_prod')) {
        missingPrices.push(`${tier}_${cycle}`);
      }
    });
  });
  
  return {
    isValid: missingPrices.length === 0,
    missingPrices
  };
}

// Billing cycle display names and discounts
export const BILLING_CYCLE_CONFIG = {
  monthly: { 
    displayName: 'Monthly', 
    discount: 0, 
    savings: 'Pay monthly',
    multiplier: 1 
  },
  quarterly: { 
    displayName: '3 Months', 
    discount: 5, 
    savings: '5% off',
    multiplier: 3 
  },
  semiannual: { 
    displayName: '6 Months', 
    discount: 10, 
    savings: '10% off',
    multiplier: 6 
  },
  annual: { 
    displayName: 'Annual', 
    discount: 20, 
    savings: '20% off - 2 months FREE',
    multiplier: 12 
  }
} as const;

/**
 * Get billing cycle configuration
 */
export function getBillingCycleConfig(cycle: BillingCycle) {
  return BILLING_CYCLE_CONFIG[cycle];
}