/**
 * Strava Integration
 *
 * This module provides integration with Strava for syncing cycling and running activities.
 *
 * Features:
 * - Activity data synchronization
 * - Training load metrics
 * - Performance analytics
 * - Route tracking
 *
 * @see https://developers.strava.com/
 */

export interface StravaConfig {
  userId: string;
  clientId?: string;
  clientSecret?: string;
}

export interface StravaActivity {
  id: string;
  name: string;
  type: 'Run' | 'Ride' | 'Swim' | 'Other';
  distance: number; // in meters
  duration: number; // in seconds
  elevationGain: number; // in meters
  averageHeartRate?: number;
  maxHeartRate?: number;
  calories?: number;
  timestamp: Date;
}

/**
 * Connect to Strava
 *
 * Initiates OAuth flow for Strava integration
 *
 * @param config - Configuration for Strava connection
 * @returns Promise<boolean> - Success status
 */
export const connectStrava = async (config: StravaConfig): Promise<boolean> => {
  try {
    const { userId, clientId, clientSecret } = config;

    if (!clientId && !clientSecret) {
      // Initiate OAuth flow
      const stravaClientId = import.meta.env.VITE_STRAVA_CLIENT_ID;
      const redirectUri = `${window.location.origin}/integrations/strava/callback`;
      const scope = 'read,activity:read_all,profile:read_all';

      const authUrl = new URL('https://www.strava.com/oauth/authorize');
      authUrl.searchParams.set('client_id', stravaClientId);
      authUrl.searchParams.set('redirect_uri', redirectUri);
      authUrl.searchParams.set('response_type', 'code');
      authUrl.searchParams.set('scope', scope);
      authUrl.searchParams.set('state', userId);
      authUrl.searchParams.set('approval_prompt', 'auto');

      // Redirect to Strava authorization
      window.location.href = authUrl.toString();
      return false;
    }

    // Store access token in database
    const { supabase } = await import('@/integrations/supabase/client');
    const { error } = await supabase
      .from('user_integrations')
      .upsert({
        user_id: userId,
        provider: 'strava',
        access_token: clientSecret, // This would be the access token from OAuth callback
        is_active: true,
        last_sync: new Date().toISOString(),
      }, { onConflict: 'user_id,provider' });

    if (error) {
      console.error('[Strava] Failed to store credentials:', error);
      return false;
    }

    console.log('[Strava] Successfully connected for user:', userId);
    return true;
  } catch (error) {
    console.error('[Strava] Connection error:', error);
    return false;
  }
};

/**
 * Sync Strava activities
 *
 * Fetches recent activities from Strava
 *
 * @param userId - User ID to sync data for
 * @param after - Only fetch activities after this timestamp
 * @returns Promise<StravaActivity[]> - Array of activities
 */
export const syncStravaData = async (
  userId: string,
  after?: Date
): Promise<StravaActivity[]> => {
  try {
    const { supabase } = await import('@/integrations/supabase/client');

    // Get access token from database
    const { data: integration, error } = await supabase
      .from('user_integrations')
      .select('access_token, is_active')
      .eq('user_id', userId)
      .eq('provider', 'strava')
      .single();

    if (error || !integration || !integration.is_active) {
      console.error('[Strava] No active integration found for user:', userId);
      return [];
    }

    const accessToken = integration.access_token;

    // Fetch activities from Strava API
    const url = new URL('https://www.strava.com/api/v3/athlete/activities');
    if (after) {
      url.searchParams.set('after', Math.floor(after.getTime() / 1000).toString());
    }
    url.searchParams.set('per_page', '30');

    const response = await fetch(url.toString(), {
      headers: {
        'Authorization': `Bearer ${accessToken}`,
      },
    });

    if (!response.ok) {
      throw new Error(`Strava API error: ${response.statusText}`);
    }

    const stravaActivities = await response.json();

    // Transform to internal format
    const activities: StravaActivity[] = stravaActivities.map((activity: any) => ({
      id: activity.id.toString(),
      name: activity.name,
      type: activity.type,
      distance: activity.distance,
      duration: activity.moving_time,
      elevationGain: activity.total_elevation_gain,
      averageHeartRate: activity.average_heartrate,
      maxHeartRate: activity.max_heartrate,
      calories: activity.calories,
      timestamp: new Date(activity.start_date),
    }));

    // Update last sync time and synced data
    await supabase
      .from('user_integrations')
      .update({
        last_sync: new Date().toISOString(),
        synced_data: { latest_activities: activities },
      })
      .eq('user_id', userId)
      .eq('provider', 'strava');

    console.log(`[Strava] Successfully synced ${activities.length} activities for user:`, userId);
    return activities;
  } catch (error) {
    console.error('[Strava] Sync error:', error);
    return [];
  }
};

/**
 * Disconnect Strava
 *
 * Revokes access and stops data synchronization
 *
 * @param userId - User ID to disconnect
 * @returns Promise<boolean> - Success status
 */
export const disconnectStrava = async (userId: string): Promise<boolean> => {
  try {
    const { supabase } = await import('@/integrations/supabase/client');

    // Get current integration
    const { data: integration } = await supabase
      .from('user_integrations')
      .select('access_token')
      .eq('user_id', userId)
      .eq('provider', 'strava')
      .single();

    if (integration?.access_token) {
      // Revoke access token with Strava
      try {
        await fetch('https://www.strava.com/oauth/deauthorize', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${integration.access_token}`,
          },
        });
      } catch (revokeError) {
        console.warn('[Strava] Token revocation failed, continuing with disconnect:', revokeError);
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
      .eq('provider', 'strava');

    if (error) {
      console.error('[Strava] Failed to disconnect:', error);
      return false;
    }

    console.log('[Strava] Successfully disconnected for user:', userId);
    return true;
  } catch (error) {
    console.error('[Strava] Disconnect error:', error);
    return false;
  }
};

/**
 * Get Strava connection status
 *
 * Check if user has connected Strava
 *
 * @param userId - User ID to check
 * @returns Promise<boolean> - Connection status
 */
export const getStravaStatus = async (userId: string): Promise<boolean> => {
  try {
    const { supabase } = await import('@/integrations/supabase/client');
    const { data: integration, error } = await supabase
      .from('user_integrations')
      .select('is_active')
      .eq('user_id', userId)
      .eq('provider', 'strava')
      .single();

    if (error || !integration) {
      return false;
    }

    return integration.is_active ?? false;
  } catch (error) {
    console.error('[Strava] Status check error:', error);
    return false;
  }
};
