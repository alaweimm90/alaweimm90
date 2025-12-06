import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.45.0';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { provider, authCode, clientId } = await req.json();
    
    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    );

    // Get user from auth header
    const authHeader = req.headers.get('Authorization')!;
    const token = authHeader.replace('Bearer ', '');
    const { data: { user }, error: userError } = await supabaseClient.auth.getUser(token);
    
    if (userError || !user) {
      throw new Error('Unauthorized');
    }

    let integrationData = {};

    switch (provider) {
      case 'apple-health':
        integrationData = await handleAppleHealthIntegration(authCode);
        break;
      case 'google-fit':
        integrationData = await handleGoogleFitIntegration(authCode);
        break;
      case 'fitbit':
        integrationData = await handleFitbitIntegration(authCode);
        break;
      case 'oura':
        integrationData = await handleOuraIntegration(authCode);
        break;
      case 'google-calendar':
        integrationData = await handleGoogleCalendarIntegration(authCode);
        break;
      case 'myfitnesspal':
        integrationData = await handleMyFitnessPalIntegration(authCode);
        break;
      case 'zoom':
        integrationData = await handleZoomIntegration(authCode);
        break;
      default:
        throw new Error('Unsupported provider');
    }

    // Store integration data in database
    const { error: dbError } = await supabaseClient
      .from('user_integrations')
      .upsert({
        user_id: user.id,
        provider: provider,
        status: 'connected',
        access_token: integrationData.accessToken,
        refresh_token: integrationData.refreshToken,
        expires_at: integrationData.expiresAt,
        last_sync: new Date().toISOString(),
        data: integrationData.data
      }, { onConflict: 'user_id,provider' });

    if (dbError) throw dbError;

    return new Response(JSON.stringify({ 
      success: true, 
      provider,
      data: integrationData.data 
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });

  } catch (error) {
    console.error('Integration error:', error);
    return new Response(JSON.stringify({ 
      error: 'Integration failed. Please try again.' 
    }), {
      status: 400,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }
});

async function handleAppleHealthIntegration(authCode: string) {
  // In a real implementation, this would handle Apple HealthKit OAuth
  // For now, return mock data
  return {
    accessToken: 'mock_apple_token',
    refreshToken: 'mock_apple_refresh',
    expiresAt: new Date(Date.now() + 3600000).toISOString(),
    data: {
      steps: 8547,
      heartRate: 72,
      calories: 2240,
      sleep: 7.5,
      activeMinutes: 45
    }
  };
}

async function handleGoogleFitIntegration(authCode: string) {
  // In a real implementation, this would use Google Fit API
  return {
    accessToken: 'mock_google_token',
    refreshToken: 'mock_google_refresh',
    expiresAt: new Date(Date.now() + 3600000).toISOString(),
    data: {
      steps: 9234,
      heartRate: 68,
      calories: 2156,
      activeMinutes: 52
    }
  };
}

async function handleFitbitIntegration(authCode: string) {
  // In a real implementation, this would use Fitbit OAuth
  return {
    accessToken: 'mock_fitbit_token',
    refreshToken: 'mock_fitbit_refresh',
    expiresAt: new Date(Date.now() + 3600000).toISOString(),
    data: {
      steps: 7892,
      heartRate: 75,
      calories: 2089,
      sleep: 8.2,
      activeMinutes: 38
    }
  };
}

async function handleOuraIntegration(authCode: string) {
  // In a real implementation, this would use Oura API
  return {
    accessToken: 'mock_oura_token',
    refreshToken: 'mock_oura_refresh',
    expiresAt: new Date(Date.now() + 3600000).toISOString(),
    data: {
      sleep: 8.1,
      readiness: 85,
      activity: 420,
      heartRateVariability: 45,
      temperature: 36.8
    }
  };
}

async function handleGoogleCalendarIntegration(authCode: string) {
  // In a real implementation, this would use Google Calendar API
  return {
    accessToken: 'mock_calendar_token',
    refreshToken: 'mock_calendar_refresh',
    expiresAt: new Date(Date.now() + 3600000).toISOString(),
    data: {
      events: [
        {
          id: '1',
          title: 'Morning Workout',
          start: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(),
          end: new Date(Date.now() + 25 * 60 * 60 * 1000).toISOString()
        }
      ]
    }
  };
}

async function handleMyFitnessPalIntegration(authCode: string) {
  // In a real implementation, this would use MyFitnessPal API
  return {
    accessToken: 'mock_mfp_token',
    refreshToken: 'mock_mfp_refresh',
    expiresAt: new Date(Date.now() + 3600000).toISOString(),
    data: {
      calories: 1847,
      protein: 142,
      carbs: 183,
      fat: 62,
      meals: [
        { name: 'Breakfast', calories: 420, time: '08:00' },
        { name: 'Lunch', calories: 650, time: '12:30' }
      ]
    }
  };
}

async function handleZoomIntegration(authCode: string) {
  // In a real implementation, this would use Zoom API
  return {
    accessToken: 'mock_zoom_token',
    refreshToken: 'mock_zoom_refresh',
    expiresAt: new Date(Date.now() + 3600000).toISOString(),
    data: {
      meetings: [
        {
          id: 'zoom_meeting_1',
          topic: 'Coaching Session',
          startTime: new Date(Date.now() + 2 * 60 * 60 * 1000).toISOString(),
          duration: 60,
          joinUrl: 'https://zoom.us/j/mock'
        }
      ]
    }
  };
}