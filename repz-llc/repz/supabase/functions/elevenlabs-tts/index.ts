import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

// Coach personality configurations
const COACH_VOICES = {
  motivational: {
    voiceId: '9BWtsMINqrJLrRacOk9x', // Aria - energetic and encouraging
    model: 'eleven_turbo_v2',
    stability: 0.5,
    similarity_boost: 0.8,
    style: 0.3,
    use_speaker_boost: true
  },
  technical: {
    voiceId: 'CwhRBWXzGAHq8TQ4Fs17', // Roger - clear and instructional
    model: 'eleven_turbo_v2',
    stability: 0.7,
    similarity_boost: 0.7,
    style: 0.1,
    use_speaker_boost: true
  },
  supportive: {
    voiceId: 'EXAVITQu4vr4xnSDxMaL', // Sarah - warm and encouraging
    model: 'eleven_turbo_v2',
    stability: 0.6,
    similarity_boost: 0.8,
    style: 0.4,
    use_speaker_boost: true
  },
  intense: {
    voiceId: 'TX3LPaxmHKxFdv7VOQHJ', // Liam - powerful and commanding
    model: 'eleven_turbo_v2',
    stability: 0.4,
    similarity_boost: 0.9,
    style: 0.6,
    use_speaker_boost: true
  }
};

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { text, coachPersonality = 'motivational', priority = 'normal' } = await req.json();
    
    if (!text || text.trim().length === 0) {
      throw new Error('Text is required for speech synthesis');
    }

    const ELEVENLABS_API_KEY = Deno.env.get('ELEVENLABS_API_KEY');
    if (!ELEVENLABS_API_KEY) {
      throw new Error('ElevenLabs API key not configured');
    }

    // Get voice configuration for the selected coach personality
    const voiceConfig = COACH_VOICES[coachPersonality as keyof typeof COACH_VOICES] || COACH_VOICES.motivational;
    
    console.log(`Generating speech for coach: ${coachPersonality}, text: "${text.substring(0, 50)}..."`);

    // Call ElevenLabs Text-to-Speech API
    const response = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${voiceConfig.voiceId}`, {
      method: 'POST',
      headers: {
        'Accept': 'audio/mpeg',
        'Content-Type': 'application/json',
        'xi-api-key': ELEVENLABS_API_KEY,
      },
      body: JSON.stringify({
        text: text,
        model_id: voiceConfig.model,
        voice_settings: {
          stability: voiceConfig.stability,
          similarity_boost: voiceConfig.similarity_boost,
          style: voiceConfig.style,
          use_speaker_boost: voiceConfig.use_speaker_boost
        },
        output_format: 'mp3_44100_128'
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('ElevenLabs API error:', response.status, errorText);
      throw new Error(`ElevenLabs API error: ${response.status} - ${errorText}`);
    }

    // Convert audio buffer to base64
    const arrayBuffer = await response.arrayBuffer();
    const base64Audio = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));

    console.log(`Speech generated successfully, audio size: ${arrayBuffer.byteLength} bytes`);

    return new Response(
      JSON.stringify({ 
        audioContent: base64Audio,
        coachPersonality,
        textLength: text.length,
        audioSize: arrayBuffer.byteLength
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      },
    );

  } catch (error) {
    console.error('Error in elevenlabs-tts function:', error);
    return new Response(
      JSON.stringify({ 
        error: 'Text-to-speech conversion failed. Please try again.',
        timestamp: new Date().toISOString()
      }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      },
    );
  }
});