/**
 * Referral Service
 *
 * Handles all referral-related operations including:
 * - Generating and managing referral codes
 * - Tracking referrals
 * - Applying rewards
 * - Analytics
 */

import { supabase } from '@/integrations/supabase/client';

export interface ReferralCode {
  id: string;
  user_id: string;
  code: string;
  uses_remaining: number | null;
  expires_at: string | null;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface Referral {
  id: string;
  referrer_user_id: string;
  referred_user_id: string;
  referral_code: string;
  status: 'pending' | 'completed' | 'cancelled';
  reward_type: string | null;
  reward_amount: number | null;
  reward_claimed: boolean;
  reward_claimed_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface ReferralReward {
  id: string;
  tier: string;
  reward_type: string;
  reward_value: number;
  referrer_reward: string;
  referee_reward: string;
  min_referrals: number;
  is_active: boolean;
}

export interface ReferralStats {
  total_referrals: number;
  pending_referrals: number;
  completed_referrals: number;
  total_rewards: number;
  unclaimed_rewards: number;
}

/**
 * Get user's referral code
 */
export async function getUserReferralCode(userId: string): Promise<ReferralCode | null> {
  try {
    const { data, error } = await supabase
      .from('referral_codes')
      .select('*')
      .eq('user_id', userId)
      .eq('is_active', true)
      .single();

    if (error) {
      console.error('[Referral Service] Error fetching referral code:', error);
      return null;
    }

    return data;
  } catch (error) {
    console.error('[Referral Service] Exception:', error);
    return null;
  }
}

/**
 * Create a new referral code for user
 */
export async function createReferralCode(
  userId: string,
  options?: {
    usesRemaining?: number;
    expiresAt?: Date;
  }
): Promise<ReferralCode | null> {
  try {
    // Generate code via database function
    const { data: codeData, error: codeError } = await supabase.rpc('generate_referral_code');

    if (codeError || !codeData) {
      console.error('[Referral Service] Error generating code:', codeError);
      return null;
    }

    const { data, error } = await supabase
      .from('referral_codes')
      .insert({
        user_id: userId,
        code: codeData,
        uses_remaining: options?.usesRemaining || null,
        expires_at: options?.expiresAt?.toISOString() || null,
      })
      .select()
      .single();

    if (error) {
      console.error('[Referral Service] Error creating referral code:', error);
      return null;
    }

    return data;
  } catch (error) {
    console.error('[Referral Service] Exception:', error);
    return null;
  }
}

/**
 * Apply a referral code for a new user
 */
export async function applyReferralCode(
  referralCode: string,
  referredUserId: string
): Promise<{ success: boolean; error?: string; referral?: Referral }> {
  try {
    const { data, error } = await supabase.rpc('apply_referral_reward', {
      p_referral_code: referralCode,
      p_referred_user_id: referredUserId,
    });

    if (error) {
      console.error('[Referral Service] Error applying referral code:', error);
      return { success: false, error: error.message };
    }

    if (!data.success) {
      return { success: false, error: data.error };
    }

    return { success: true, referral: data };
  } catch (error) {
    console.error('[Referral Service] Exception:', error);
    return { success: false, error: 'An unexpected error occurred' };
  }
}

/**
 * Get user's referrals (people they referred)
 */
export async function getUserReferrals(userId: string): Promise<Referral[]> {
  try {
    const { data, error } = await supabase
      .from('referrals')
      .select('*')
      .eq('referrer_user_id', userId)
      .order('created_at', { ascending: false });

    if (error) {
      console.error('[Referral Service] Error fetching referrals:', error);
      return [];
    }

    return data || [];
  } catch (error) {
    console.error('[Referral Service] Exception:', error);
    return [];
  }
}

/**
 * Get referral stats for a user
 */
export async function getReferralStats(userId: string): Promise<ReferralStats> {
  try {
    const referrals = await getUserReferrals(userId);

    const stats: ReferralStats = {
      total_referrals: referrals.length,
      pending_referrals: referrals.filter(r => r.status === 'pending').length,
      completed_referrals: referrals.filter(r => r.status === 'completed').length,
      total_rewards: referrals.reduce((sum, r) => sum + (r.reward_amount || 0), 0),
      unclaimed_rewards: referrals
        .filter(r => !r.reward_claimed && r.reward_amount)
        .reduce((sum, r) => sum + (r.reward_amount || 0), 0),
    };

    return stats;
  } catch (error) {
    console.error('[Referral Service] Exception:', error);
    return {
      total_referrals: 0,
      pending_referrals: 0,
      completed_referrals: 0,
      total_rewards: 0,
      unclaimed_rewards: 0,
    };
  }
}

/**
 * Claim a referral reward
 */
export async function claimReferralReward(referralId: string): Promise<boolean> {
  try {
    const { error } = await supabase
      .from('referrals')
      .update({
        reward_claimed: true,
        reward_claimed_at: new Date().toISOString(),
        status: 'completed',
      })
      .eq('id', referralId);

    if (error) {
      console.error('[Referral Service] Error claiming reward:', error);
      return false;
    }

    return true;
  } catch (error) {
    console.error('[Referral Service] Exception:', error);
    return false;
  }
}

/**
 * Get available referral rewards
 */
export async function getReferralRewards(): Promise<ReferralReward[]> {
  try {
    const { data, error } = await supabase
      .from('referral_rewards')
      .select('*')
      .eq('is_active', true)
      .order('tier');

    if (error) {
      console.error('[Referral Service] Error fetching rewards:', error);
      return [];
    }

    return data || [];
  } catch (error) {
    console.error('[Referral Service] Exception:', error);
    return [];
  }
}

/**
 * Deactivate a referral code
 */
export async function deactivateReferralCode(codeId: string): Promise<boolean> {
  try {
    const { error } = await supabase
      .from('referral_codes')
      .update({ is_active: false })
      .eq('id', codeId);

    if (error) {
      console.error('[Referral Service] Error deactivating code:', error);
      return false;
    }

    return true;
  } catch (error) {
    console.error('[Referral Service] Exception:', error);
    return false;
  }
}

/**
 * Update referral status
 */
export async function updateReferralStatus(
  referralId: string,
  status: 'pending' | 'completed' | 'cancelled'
): Promise<boolean> {
  try {
    const { error } = await supabase
      .from('referrals')
      .update({ status })
      .eq('id', referralId);

    if (error) {
      console.error('[Referral Service] Error updating status:', error);
      return false;
    }

    return true;
  } catch (error) {
    console.error('[Referral Service] Exception:', error);
    return false;
  }
}

export default {
  getUserReferralCode,
  createReferralCode,
  applyReferralCode,
  getUserReferrals,
  getReferralStats,
  claimReferralReward,
  getReferralRewards,
  deactivateReferralCode,
  updateReferralStatus,
};
