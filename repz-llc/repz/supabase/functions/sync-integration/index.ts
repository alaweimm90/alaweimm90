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
    const { provider } = await req.json();
    
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

    // Get user's integration
    const { data: integration, error: integrationError } = await supabaseClient
      .from('user_integrations')
      .select('*')
      .eq('user_id', user.id)
      .eq('provider', provider)
      .single();

    if (integrationError || !integration) {
      throw new Error('Integration not found or not connected');
    }

    let syncedData = {};

    // Sync data based on provider
    switch (provider) {
      case 'apple-health':
        syncedData = await syncAppleHealthData(integration.access_token);
        break;
      case 'google-fit':
        syncedData = await syncGoogleFitData(integration.access_token);
        break;
      case 'fitbit':
        syncedData = await syncFitbitData(integration.access_token);
        break;
      case 'oura':
        syncedData = await syncOuraData(integration.access_token);
        break;
      case 'google-calendar':
        syncedData = await syncGoogleCalendarData(integration.access_token);
        break;
      case 'myfitnesspal':
        syncedData = await syncMyFitnessPalData(integration.access_token);
        break;
      case 'zoom':
        syncedData = await syncZoomData(integration.access_token);
        break;
      default:
        throw new Error('Unsupported provider');
    }

    // Update integration with new data
    const { error: updateError } = await supabaseClient
      .from('user_integrations')
      .update({
        data: syncedData,
        last_sync: new Date().toISOString()
      })
      .eq('user_id', user.id)
      .eq('provider', provider);

    if (updateError) throw updateError;

    return new Response(JSON.stringify({ 
      success: true, 
      provider,
      data: syncedData,
      lastSync: new Date().toISOString()
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });

  } catch (error) {
    console.error('Sync error:', error);
    return new Response(JSON.stringify({ 
      error: error.message 
    }), {
      status: 400,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }
});

async function syncAppleHealthData(accessToken: string) {
  // Mock updated data - in production this would fetch from Apple HealthKit
  return {
    steps: Math.floor(Math.random() * 3000) + 7000,
    heartRate: Math.floor(Math.random() * 20) + 65,
    calories: Math.floor(Math.random() * 500) + 2000,
    sleep: Math.floor(Math.random() * 2) + 7,
    activeMinutes: Math.floor(Math.random() * 30) + 30,
    timestamp: new Date().toISOString()
  };
}

async function syncGoogleFitData(accessToken: string) {
  // Mock updated data - in production this would fetch from Google Fit API
  return {
    steps: Math.floor(Math.random() * 3000) + 8000,
    heartRate: Math.floor(Math.random() * 15) + 68,
    calories: Math.floor(Math.random() * 400) + 2100,
    activeMinutes: Math.floor(Math.random() * 40) + 40,
    timestamp: new Date().toISOString()
  };
}

async function syncFitbitData(accessToken: string) {
  // Mock updated data - in production this would fetch from Fitbit API
  return {
    steps: Math.floor(Math.random() * 2500) + 7500,
    heartRate: Math.floor(Math.random() * 18) + 70,
    calories: Math.floor(Math.random() * 450) + 2050,
    sleep: Math.floor(Math.random() * 1.5) + 7.5,
    activeMinutes: Math.floor(Math.random() * 25) + 35,
    timestamp: new Date().toISOString()
  };
}

async function syncOuraData(accessToken: string) {
  // Mock updated data - in production this would fetch from Oura API
  return {
    sleep: Math.floor(Math.random() * 1.5) + 7.5,
    readiness: Math.floor(Math.random() * 20) + 80,
    activity: Math.floor(Math.random() * 100) + 350,
    heartRateVariability: Math.floor(Math.random() * 15) + 40,
    temperature: (Math.random() * 0.5) + 36.5,
    timestamp: new Date().toISOString()
  };
}

async function syncGoogleCalendarData(accessToken: string) {
  // Mock updated data - in production this would fetch from Google Calendar API
  const now = new Date();
  return {
    events: [
      {
        id: Math.random().toString(),
        title: 'Sync Test Event',
        start: new Date(now.getTime() + Math.random() * 48 * 60 * 60 * 1000).toISOString(),
        end: new Date(now.getTime() + Math.random() * 48 * 60 * 60 * 1000 + 60 * 60 * 1000).toISOString()
      }
    ],
    timestamp: new Date().toISOString()
  };
}

async function syncMyFitnessPalData(accessToken: string) {
  // Mock updated data - in production this would fetch from MyFitnessPal API
  return {
    calories: Math.floor(Math.random() * 400) + 1600,
    protein: Math.floor(Math.random() * 50) + 120,
    carbs: Math.floor(Math.random() * 60) + 150,
    fat: Math.floor(Math.random() * 30) + 50,
    meals: [
      { 
        name: 'Updated Meal', 
        calories: Math.floor(Math.random() * 200) + 300, 
        time: new Date().toLocaleTimeString().slice(0, 5)
      }
    ],
    timestamp: new Date().toISOString()
  };
}

async function syncZoomData(accessToken: string) {
  // Mock updated data - in production this would fetch from Zoom API
  return {
    meetings: [
      {
        id: 'updated_meeting_' + Math.random().toString(),
        topic: 'Updated Coaching Session',
        startTime: new Date(Date.now() + Math.random() * 72 * 60 * 60 * 1000).toISOString(),
        duration: 60,
        joinUrl: 'https://zoom.us/j/updated_mock'
      }
    ],
    timestamp: new Date().toISOString()
  };
}