/**
 * Google Calendar Integration
 *
 * This module provides integration with Google Calendar for workout scheduling.
 *
 * Features:
 * - Sync workouts to calendar
 * - Show availability conflicts
 * - Send calendar invites for group sessions
 * - Reminder notifications
 *
 * @see https://developers.google.com/calendar
 */

export interface CalendarConfig {
  userId: string;
  clientId?: string;
  clientSecret?: string;
  calendarId?: string;
}

export interface CalendarEvent {
  id?: string;
  title: string;
  description?: string;
  startTime: Date;
  endTime: Date;
  location?: string;
  reminders?: {
    method: 'email' | 'popup';
    minutes: number;
  }[];
}

/**
 * Connect to Google Calendar
 *
 * Initiates OAuth flow for Google Calendar integration
 *
 * @param config - Configuration for Google Calendar connection
 * @returns Promise<boolean> - Success status
 */
export const connectGoogleCalendar = async (config: CalendarConfig): Promise<boolean> => {
  try {
    const { userId, clientId, clientSecret } = config;

    if (!clientId && !clientSecret) {
      // Initiate OAuth flow
      const googleClientId = import.meta.env.VITE_GOOGLE_CLIENT_ID;
      const redirectUri = `${window.location.origin}/integrations/google-calendar/callback`;
      const scope = 'https://www.googleapis.com/auth/calendar https://www.googleapis.com/auth/calendar.events';

      const authUrl = new URL('https://accounts.google.com/o/oauth2/v2/auth');
      authUrl.searchParams.set('client_id', googleClientId);
      authUrl.searchParams.set('redirect_uri', redirectUri);
      authUrl.searchParams.set('response_type', 'code');
      authUrl.searchParams.set('scope', scope);
      authUrl.searchParams.set('access_type', 'offline');
      authUrl.searchParams.set('state', userId);

      // Redirect to Google OAuth
      window.location.href = authUrl.toString();
      return false;
    }

    // Store credentials in database
    const { supabase } = await import('@/integrations/supabase/client');
    const { error } = await supabase
      .from('user_integrations')
      .upsert({
        user_id: userId,
        provider: 'google_calendar',
        access_token: clientSecret, // This would be the access token from OAuth callback
        is_active: true,
        synced_data: { calendar_id: config.calendarId || 'primary' },
        last_sync: new Date().toISOString(),
      }, { onConflict: 'user_id,provider' });

    if (error) {
      console.error('[Google Calendar] Failed to store credentials:', error);
      return false;
    }

    console.log('[Google Calendar] Successfully connected for user:', userId);
    return true;
  } catch (error) {
    console.error('[Google Calendar] Connection error:', error);
    return false;
  }
};

/**
 * Sync workout to Google Calendar
 *
 * Creates or updates a workout event in Google Calendar
 *
 * @param userId - User ID
 * @param event - Event details to sync
 * @returns Promise<string | null> - Event ID if successful
 */
export const syncWorkoutToCalendar = async (
  userId: string,
  event: CalendarEvent
): Promise<string | null> => {
  try {
    const { supabase } = await import('@/integrations/supabase/client');

    // Get user's integration details
    const { data: integration, error } = await supabase
      .from('user_integrations')
      .select('access_token, synced_data, is_active')
      .eq('user_id', userId)
      .eq('provider', 'google_calendar')
      .single();

    if (error || !integration || !integration.is_active) {
      console.error('[Google Calendar] No active integration found');
      return null;
    }

    const calendarId = integration.synced_data?.calendar_id || 'primary';
    const accessToken = integration.access_token;

    // Prepare event data for Google Calendar API
    const calendarEvent = {
      summary: event.title,
      description: event.description,
      location: event.location,
      start: {
        dateTime: event.startTime.toISOString(),
        timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      },
      end: {
        dateTime: event.endTime.toISOString(),
        timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      },
      reminders: {
        useDefault: !event.reminders,
        overrides: event.reminders,
      },
    };

    // Create or update event in Google Calendar
    const method = event.id ? 'PUT' : 'POST';
    const endpoint = event.id
      ? `https://www.googleapis.com/calendar/v3/calendars/${calendarId}/events/${event.id}`
      : `https://www.googleapis.com/calendar/v3/calendars/${calendarId}/events`;

    const response = await fetch(endpoint, {
      method,
      headers: {
        'Authorization': `Bearer ${accessToken}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(calendarEvent),
    });

    if (!response.ok) {
      throw new Error(`Google Calendar API error: ${response.statusText}`);
    }

    const data = await response.json();
    console.log('[Google Calendar] Successfully synced workout for user:', userId);
    return data.id;
  } catch (error) {
    console.error('[Google Calendar] Sync error:', error);
    return null;
  }
};

/**
 * Check availability
 *
 * Checks if user has conflicts during proposed workout time
 *
 * @param userId - User ID
 * @param startTime - Proposed start time
 * @param endTime - Proposed end time
 * @returns Promise<boolean> - True if time slot is available
 */
export const checkAvailability = async (
  userId: string,
  startTime: Date,
  endTime: Date
): Promise<boolean> => {
  try {
    const { supabase } = await import('@/integrations/supabase/client');

    // Get user's integration details
    const { data: integration, error } = await supabase
      .from('user_integrations')
      .select('access_token, synced_data, is_active')
      .eq('user_id', userId)
      .eq('provider', 'google_calendar')
      .single();

    if (error || !integration || !integration.is_active) {
      console.warn('[Google Calendar] No active integration, assuming available');
      return true;
    }

    const calendarId = integration.synced_data?.calendar_id || 'primary';
    const accessToken = integration.access_token;

    // Query Google Calendar for events in time range
    const url = new URL(`https://www.googleapis.com/calendar/v3/calendars/${calendarId}/events`);
    url.searchParams.set('timeMin', startTime.toISOString());
    url.searchParams.set('timeMax', endTime.toISOString());
    url.searchParams.set('singleEvents', 'true');

    const response = await fetch(url.toString(), {
      headers: {
        'Authorization': `Bearer ${accessToken}`,
      },
    });

    if (!response.ok) {
      throw new Error(`Google Calendar API error: ${response.statusText}`);
    }

    const data = await response.json();

    // Check if there are any conflicting events
    const hasConflict = data.items && data.items.length > 0;

    console.log(`[Google Calendar] Availability check for user ${userId}: ${!hasConflict ? 'available' : 'busy'}`);
    return !hasConflict;
  } catch (error) {
    console.error('[Google Calendar] Availability check error:', error);
    return true; // Assume available on error
  }
};

/**
 * Disconnect Google Calendar
 *
 * Revokes access and stops calendar synchronization
 *
 * @param userId - User ID to disconnect
 * @returns Promise<boolean> - Success status
 */
export const disconnectGoogleCalendar = async (userId: string): Promise<boolean> => {
  try {
    const { supabase } = await import('@/integrations/supabase/client');

    // Get current integration
    const { data: integration } = await supabase
      .from('user_integrations')
      .select('access_token')
      .eq('user_id', userId)
      .eq('provider', 'google_calendar')
      .single();

    if (integration?.access_token) {
      // Revoke OAuth token with Google
      try {
        await fetch(`https://oauth2.googleapis.com/revoke?token=${integration.access_token}`, {
          method: 'POST',
        });
      } catch (revokeError) {
        console.warn('[Google Calendar] Token revocation failed, continuing with disconnect:', revokeError);
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
      .eq('provider', 'google_calendar');

    if (error) {
      console.error('[Google Calendar] Failed to disconnect:', error);
      return false;
    }

    console.log('[Google Calendar] Successfully disconnected for user:', userId);
    return true;
  } catch (error) {
    console.error('[Google Calendar] Disconnect error:', error);
    return false;
  }
};

/**
 * Get Google Calendar connection status
 *
 * Check if user has connected Google Calendar
 *
 * @param userId - User ID to check
 * @returns Promise<boolean> - Connection status
 */
export const getGoogleCalendarStatus = async (userId: string): Promise<boolean> => {
  try {
    const { supabase } = await import('@/integrations/supabase/client');
    const { data: integration, error } = await supabase
      .from('user_integrations')
      .select('is_active')
      .eq('user_id', userId)
      .eq('provider', 'google_calendar')
      .single();

    if (error || !integration) {
      return false;
    }

    return integration.is_active ?? false;
  } catch (error) {
    console.error('[Google Calendar] Status check error:', error);
    return false;
  }
};
