/**
 * Whoop Integration
 *
 * This module provides integration with Whoop for advanced biometric tracking.
 *
 * Features:
 * - Strain score monitoring
 * - Recovery status
 * - Heart rate variability (HRV)
 * - Sleep performance
 * - Respiratory rate
 *
 * @see https://developer.whoop.com/
 */

export interface WhoopConfig {
  userId: string;
  accessToken?: string;
}

export interface WhoopData {
  strain?: {
    score: number; // 0-21 scale
    averageHeartRate: number;
    maxHeartRate: number;
    calories: number;
    timestamp: Date;
  };
  recovery?: {
    score: number; // 0-100 percentage
    hrv: number; // in milliseconds
    restingHeartRate: number;
    sleepPerformance: number; // percentage
    timestamp: Date;
  };
  sleep?: {
    duration: number; // in minutes
    efficiency: number; // percentage
    remSleep: number; // in minutes
    deepSleep: number; // in minutes
    lightSleep: number; // in minutes
    respiratoryRate: number;
    timestamp: Date;
  };
}

/**
 * Connect to Whoop
 *
 * Initiates OAuth flow for Whoop integration
 *
 * @param config - Configuration for Whoop connection
 * @returns Promise<boolean> - Success status
 */
export const connectWhoop = async (config: WhoopConfig): Promise<boolean> => {
  try {
    const { userId, accessToken } = config;

    if (!accessToken) {
      // Initiate OAuth flow
      const clientId = import.meta.env.VITE_WHOOP_CLIENT_ID;
      const redirectUri = `${window.location.origin}/integrations/whoop/callback`;
      const scope = 'read:recovery read:cycles read:workout read:sleep read:profile';

      const authUrl = new URL('https://api.whoop.com/oauth/authorize');
      authUrl.searchParams.set('client_id', clientId);
      authUrl.searchParams.set('redirect_uri', redirectUri);
      authUrl.searchParams.set('response_type', 'code');
      authUrl.searchParams.set('scope', scope);
      authUrl.searchParams.set('state', userId);

      // Redirect to Whoop authorization
      window.location.href = authUrl.toString();
      return false;
    }

    // Store access token in database
    const { supabase } = await import('@/integrations/supabase/client');
    const { error } = await supabase
      .from('user_integrations')
      .upsert({
        user_id: userId,
        provider: 'whoop',
        access_token: accessToken,
        is_active: true,
        last_sync: new Date().toISOString(),
      }, { onConflict: 'user_id,provider' });

    if (error) {
      console.error('[Whoop] Failed to store credentials:', error);
      return false;
    }

    console.log('[Whoop] Successfully connected for user:', userId);
    return true;
  } catch (error) {
    console.error('[Whoop] Connection error:', error);
    return false;
  }
};

/**
 * Sync Whoop data
 *
 * Fetches latest biometric data from Whoop
 *
 * @param userId - User ID to sync data for
 * @returns Promise<WhoopData | null> - Synced Whoop data
 */
export const syncWhoopData = async (userId: string): Promise<WhoopData | null> => {
  try {
    // Get access token from database
    const { supabase } = await import('@/integrations/supabase/client');
    const { data: integration, error } = await supabase
      .from('user_integrations')
      .select('access_token, is_active')
      .eq('user_id', userId)
      .eq('provider', 'whoop')
      .single();

    if (error || !integration || !integration.is_active) {
      console.error('[Whoop] No active integration found for user:', userId);
      return null;
    }

    const accessToken = integration.access_token;
    const baseUrl = 'https://api.whoop.com/v1';

    // Fetch recovery data
    const recoveryResponse = await fetch(`${baseUrl}/recovery`, {
      headers: { 'Authorization': `Bearer ${accessToken}` },
    });
    const recoveryData = await recoveryResponse.json();

    // Fetch cycle data (strain)
    const cycleResponse = await fetch(`${baseUrl}/cycle`, {
      headers: { 'Authorization': `Bearer ${accessToken}` },
    });
    const cycleData = await cycleResponse.json();

    // Fetch sleep data
    const sleepResponse = await fetch(`${baseUrl}/sleep`, {
      headers: { 'Authorization': `Bearer ${accessToken}` },
    });
    const sleepData = await sleepResponse.json();

    // Transform to internal format
    const whoopData: WhoopData = {
      strain: cycleData?.score ? {
        score: cycleData.score.strain,
        averageHeartRate: cycleData.score.average_heart_rate,
        maxHeartRate: cycleData.score.max_heart_rate,
        calories: cycleData.score.kilojoule,
        timestamp: new Date(cycleData.created_at),
      } : undefined,
      recovery: recoveryData?.score ? {
        score: recoveryData.score.recovery_score,
        hrv: recoveryData.score.hrv_rmssd_milli,
        restingHeartRate: recoveryData.score.resting_heart_rate,
        sleepPerformance: recoveryData.score.sleep_performance_percentage,
        timestamp: new Date(recoveryData.created_at),
      } : undefined,
      sleep: sleepData?.sleep ? {
        duration: sleepData.sleep.total_in_bed_time_milli / 60000, // convert to minutes
        efficiency: sleepData.sleep.sleep_efficiency,
        remSleep: sleepData.sleep.rem_sleep_duration / 60000,
        deepSleep: sleepData.sleep.slow_wave_sleep_duration / 60000,
        lightSleep: sleepData.sleep.light_sleep_duration / 60000,
        respiratoryRate: sleepData.sleep.respiratory_rate,
        timestamp: new Date(sleepData.created_at),
      } : undefined,
    };

    // Update last_sync and synced_data in database
    await supabase
      .from('user_integrations')
      .update({
        last_sync: new Date().toISOString(),
        synced_data: whoopData,
      })
      .eq('user_id', userId)
      .eq('provider', 'whoop');

    console.log('[Whoop] Successfully synced data for user:', userId);
    return whoopData;
  } catch (error) {
    console.error('[Whoop] Sync error:', error);
    return null;
  }
};

/**
 * Disconnect Whoop
 *
 * Revokes access and stops data synchronization
 *
 * @param userId - User ID to disconnect
 * @returns Promise<boolean> - Success status
 */
export const disconnectWhoop = async (userId: string): Promise<boolean> => {
  try {
    const { supabase } = await import('@/integrations/supabase/client');

    // Get current integration
    const { data: integration } = await supabase
      .from('user_integrations')
      .select('access_token')
      .eq('user_id', userId)
      .eq('provider', 'whoop')
      .single();

    if (integration?.access_token) {
      // Revoke OAuth token with Whoop
      try {
        const clientId = import.meta.env.VITE_WHOOP_CLIENT_ID;
        const clientSecret = import.meta.env.VITE_WHOOP_CLIENT_SECRET;

        await fetch('https://api.whoop.com/oauth/revoke', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Authorization': `Basic ${btoa(`${clientId}:${clientSecret}`)}`,
          },
          body: new URLSearchParams({
            token: integration.access_token,
          }),
        });
      } catch (revokeError) {
        console.warn('[Whoop] Token revocation failed, continuing with disconnect:', revokeError);
      }
    }

    // Mark integration as inactive and clear tokens
    const { error } = await supabase
      .from('user_integrations')
      .update({
        is_active: false,
        access_token: null,
        refresh_token: null,
      })
      .eq('user_id', userId)
      .eq('provider', 'whoop');

    if (error) {
      console.error('[Whoop] Failed to disconnect:', error);
      return false;
    }

    console.log('[Whoop] Successfully disconnected for user:', userId);
    return true;
  } catch (error) {
    console.error('[Whoop] Disconnect error:', error);
    return false;
  }
};

/**
 * Get Whoop connection status
 *
 * Check if user has connected Whoop
 *
 * @param userId - User ID to check
 * @returns Promise<boolean> - Connection status
 */
export const getWhoopStatus = async (userId: string): Promise<boolean> => {
  try {
    const { supabase } = await import('@/integrations/supabase/client');
    const { data: integration, error } = await supabase
      .from('user_integrations')
      .select('is_active')
      .eq('user_id', userId)
      .eq('provider', 'whoop')
      .single();

    if (error || !integration) {
      return false;
    }

    return integration.is_active ?? false;
  } catch (error) {
    console.error('[Whoop] Status check error:', error);
    return false;
  }
};
