/**
 * Subscription Service
 * Handles subscription management, tier access, and Stripe integration
 */

import { supabase } from '@/integrations/supabase/client';

// ============================================================================
// TYPES
// ============================================================================

export type SubscriptionTier = 'core' | 'adaptive' | 'performance' | 'longevity';
export type BillingPeriod = 'monthly' | 'quarterly' | 'semiannual' | 'annual';
export type SubscriptionStatus = 'trialing' | 'active' | 'past_due' | 'canceled' | 'unpaid' | 'paused';

export interface Subscription {
  id: string;
  user_id: string;
  client_profile_id?: string;
  stripe_customer_id?: string;
  stripe_subscription_id?: string;
  stripe_price_id?: string;
  tier: SubscriptionTier;
  billing_period: BillingPeriod;
  amount_cents?: number;
  currency: string;
  status: SubscriptionStatus;
  current_period_start?: string;
  current_period_end?: string;
  cancel_at_period_end: boolean;
  canceled_at?: string;
  trial_start?: string;
  trial_end?: string;
  created_at: string;
  updated_at: string;
}

export interface TierFeatures {
  training_program: boolean;
  nutrition_plan: boolean;
  response_time_hours: number;
  biomarkers: boolean;
  wearable_integration: boolean;
  weekly_checkins: boolean;
  ai_assistant: boolean;
  form_analysis: boolean;
  peds_protocols: boolean;
  in_person_training: boolean;
  concierge_service: boolean;
}

// ============================================================================
// TIER CONFIGURATION
// ============================================================================

export const TIER_FEATURES: Record<SubscriptionTier, TierFeatures> = {
  core: {
    training_program: true,
    nutrition_plan: true,
    response_time_hours: 72,
    biomarkers: false,
    wearable_integration: false,
    weekly_checkins: false,
    ai_assistant: false,
    form_analysis: false,
    peds_protocols: false,
    in_person_training: false,
    concierge_service: false,
  },
  adaptive: {
    training_program: true,
    nutrition_plan: true,
    response_time_hours: 48,
    biomarkers: true,
    wearable_integration: true,
    weekly_checkins: true,
    ai_assistant: false,
    form_analysis: false,
    peds_protocols: false,
    in_person_training: false,
    concierge_service: false,
  },
  performance: {
    training_program: true,
    nutrition_plan: true,
    response_time_hours: 24,
    biomarkers: true,
    wearable_integration: true,
    weekly_checkins: true,
    ai_assistant: true,
    form_analysis: true,
    peds_protocols: true,
    in_person_training: false,
    concierge_service: false,
  },
  longevity: {
    training_program: true,
    nutrition_plan: true,
    response_time_hours: 12,
    biomarkers: true,
    wearable_integration: true,
    weekly_checkins: true,
    ai_assistant: true,
    form_analysis: true,
    peds_protocols: true,
    in_person_training: true,
    concierge_service: true,
  },
};

export const TIER_PRICING: Record<SubscriptionTier, Record<BillingPeriod, number>> = {
  core: {
    monthly: 8900,      // $89.00
    quarterly: 25365,   // $84.55/mo (5% off)
    semiannual: 48060,  // $80.10/mo (10% off)
    annual: 85440,      // $71.20/mo (20% off)
  },
  adaptive: {
    monthly: 14900,
    quarterly: 42465,
    semiannual: 80460,
    annual: 143040,
  },
  performance: {
    monthly: 22900,
    quarterly: 65265,
    semiannual: 123660,
    annual: 219840,
  },
  longevity: {
    monthly: 34900,
    quarterly: 99465,
    semiannual: 188460,
    annual: 335040,
  },
};

const TIER_ORDER: SubscriptionTier[] = ['core', 'adaptive', 'performance', 'longevity'];

// ============================================================================
// SUBSCRIPTION SERVICE
// ============================================================================

export const subscriptionService = {
  /**
   * Get current user's subscription
   */
  async getCurrentSubscription(): Promise<{ data?: Subscription; error?: string }> {
    try {
      const { data: userData } = await supabase.auth.getUser();
      if (!userData?.user?.id) {
        return { error: 'Not authenticated' };
      }

      const { data, error } = await supabase
        .from('subscriptions')
        .select('*')
        .eq('user_id', userData.user.id)
        .eq('status', 'active')
        .order('created_at', { ascending: false })
        .limit(1)
        .single();

      if (error && error.code !== 'PGRST116') throw error;

      return { data: data as Subscription };
    } catch (error) {
      console.error('Error getting subscription:', error);
      return { error: error instanceof Error ? error.message : 'Failed to get subscription' };
    }
  },

  /**
   * Get user's tier from profile
   */
  async getUserTier(): Promise<{ tier?: SubscriptionTier; error?: string }> {
    try {
      const { data: userData } = await supabase.auth.getUser();
      if (!userData?.user?.id) {
        return { error: 'Not authenticated' };
      }

      const { data, error } = await supabase
        .from('profiles')
        .select('subscription_tier')
        .eq('id', userData.user.id)
        .single();

      if (error) throw error;

      return { tier: data?.subscription_tier as SubscriptionTier };
    } catch (error) {
      console.error('Error getting user tier:', error);
      return { error: error instanceof Error ? error.message : 'Failed to get tier' };
    }
  },

  /**
   * Check if user has access to a feature
   */
  async hasFeatureAccess(feature: keyof TierFeatures): Promise<boolean> {
    try {
      const { tier } = await this.getUserTier();
      if (!tier) return false;

      return TIER_FEATURES[tier][feature] === true;
    } catch (error) {
      console.error('Error checking feature access:', error);
      return false;
    }
  },

  /**
   * Check if user has minimum tier
   */
  async hasMinimumTier(requiredTier: SubscriptionTier): Promise<boolean> {
    try {
      const { tier } = await this.getUserTier();
      if (!tier) return false;

      const userTierIndex = TIER_ORDER.indexOf(tier);
      const requiredTierIndex = TIER_ORDER.indexOf(requiredTier);

      return userTierIndex >= requiredTierIndex;
    } catch (error) {
      console.error('Error checking tier access:', error);
      return false;
    }
  },

  /**
   * Get tier features
   */
  getTierFeatures(tier: SubscriptionTier): TierFeatures {
    return TIER_FEATURES[tier];
  },

  /**
   * Get tier pricing
   */
  getTierPricing(tier: SubscriptionTier, period: BillingPeriod): number {
    return TIER_PRICING[tier][period];
  },

  /**
   * Create checkout session
   */
  async createCheckoutSession(
    tier: SubscriptionTier,
    billingPeriod: BillingPeriod = 'monthly'
  ): Promise<{ url?: string; error?: string }> {
    try {
      const { data: userData } = await supabase.auth.getUser();
      if (!userData?.user) {
        return { error: 'Not authenticated' };
      }

      const { data, error } = await supabase.functions.invoke('create-checkout', {
        body: {
          tier,
          billing_period: billingPeriod,
          trace_id: crypto.randomUUID(),
        },
      });

      if (error) throw error;

      return { url: data.url };
    } catch (error) {
      console.error('Error creating checkout:', error);
      return { error: error instanceof Error ? error.message : 'Failed to create checkout' };
    }
  },

  /**
   * Open customer portal
   */
  async openCustomerPortal(): Promise<{ url?: string; error?: string }> {
    try {
      const { data, error } = await supabase.functions.invoke('customer-portal', {
        body: {},
      });

      if (error) throw error;

      return { url: data.url };
    } catch (error) {
      console.error('Error opening portal:', error);
      return { error: error instanceof Error ? error.message : 'Failed to open portal' };
    }
  },

  /**
   * Cancel subscription
   */
  async cancelSubscription(atPeriodEnd: boolean = true): Promise<{ success: boolean; error?: string }> {
    try {
      const { data, error } = await supabase.functions.invoke('subscription-management', {
        body: {
          action: 'cancel',
          cancel_at_period_end: atPeriodEnd,
        },
      });

      if (error) throw error;

      return { success: true };
    } catch (error) {
      console.error('Error canceling subscription:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Failed to cancel' };
    }
  },

  /**
   * Update subscription tier
   */
  async updateTier(newTier: SubscriptionTier): Promise<{ success: boolean; error?: string }> {
    try {
      const { data, error } = await supabase.functions.invoke('subscription-management', {
        body: {
          action: 'update',
          new_tier: newTier,
        },
      });

      if (error) throw error;

      return { success: true };
    } catch (error) {
      console.error('Error updating tier:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Failed to update tier' };
    }
  },

  /**
   * Get all subscriptions (admin)
   */
  async getAllSubscriptions(): Promise<{ data?: Subscription[]; error?: string }> {
    try {
      const { data, error } = await supabase
        .from('subscriptions')
        .select('*')
        .order('created_at', { ascending: false });

      if (error) throw error;

      return { data: data as Subscription[] };
    } catch (error) {
      console.error('Error getting subscriptions:', error);
      return { error: error instanceof Error ? error.message : 'Failed to get subscriptions' };
    }
  },

  /**
   * Get subscription stats (admin)
   */
  async getSubscriptionStats(): Promise<{
    data?: {
      total: number;
      active: number;
      byTier: Record<SubscriptionTier, number>;
      mrr: number;
    };
    error?: string;
  }> {
    try {
      const { data: subscriptions, error } = await supabase
        .from('subscriptions')
        .select('tier, status, amount_cents')
        .eq('status', 'active');

      if (error) throw error;

      const stats = {
        total: subscriptions?.length || 0,
        active: subscriptions?.filter(s => s.status === 'active').length || 0,
        byTier: {
          core: 0,
          adaptive: 0,
          performance: 0,
          longevity: 0,
        } as Record<SubscriptionTier, number>,
        mrr: 0,
      };

      subscriptions?.forEach(sub => {
        if (sub.tier in stats.byTier) {
          stats.byTier[sub.tier as SubscriptionTier]++;
        }
        stats.mrr += sub.amount_cents || 0;
      });

      return { data: stats };
    } catch (error) {
      console.error('Error getting stats:', error);
      return { error: error instanceof Error ? error.message : 'Failed to get stats' };
    }
  },
};

export default subscriptionService;
