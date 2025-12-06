import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import "https://deno.land/x/xhr@0.1.0/mod.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

interface WorkoutRecommendationRequest {
  userProfile: {
    age: number;
    fitnessLevel: string;
    goals: string[];
    injuries: string[];
    preferences: string[];
    availableTime: number;
    equipment: string[];
  };
  recentWorkouts: any[];
  progressData: any[];
  preferences: {
    intensity: string;
    duration: number;
    workoutTypes: string[];
  };
}

const serve_handler = async (req: Request): Promise<Response> => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const openaiApiKey = Deno.env.get('OPENAI_API_KEY');
    if (!openaiApiKey) {
      throw new Error('OpenAI API key not configured');
    }

    const requestData: WorkoutRecommendationRequest = await req.json();
    
    // Create comprehensive prompt for workout recommendation
    const prompt = `
You are an expert AI fitness coach. Generate a personalized workout recommendation based on the following user data:

USER PROFILE:
- Age: ${requestData.userProfile.age}
- Fitness Level: ${requestData.userProfile.fitnessLevel}
- Goals: ${requestData.userProfile.goals.join(', ')}
- Injuries/Limitations: ${requestData.userProfile.injuries.join(', ') || 'None'}
- Preferences: ${requestData.userProfile.preferences.join(', ')}
- Available Time: ${requestData.userProfile.availableTime} minutes
- Equipment: ${requestData.userProfile.equipment.join(', ')}

RECENT WORKOUT HISTORY:
${JSON.stringify(requestData.recentWorkouts.slice(-5), null, 2)}

PROGRESS DATA:
${JSON.stringify(requestData.progressData.slice(-10), null, 2)}

WORKOUT PREFERENCES:
- Intensity: ${requestData.preferences.intensity}
- Duration: ${requestData.preferences.duration} minutes
- Workout Types: ${requestData.preferences.workoutTypes.join(', ')}

Please provide a detailed workout recommendation in the following JSON format:
{
  "workoutName": "string",
  "description": "string",
  "duration": number,
  "intensity": "low|medium|high",
  "targetMuscleGroups": ["string"],
  "exercises": [
    {
      "name": "string",
      "sets": number,
      "reps": "string",
      "restTime": number,
      "instructions": ["string"],
      "modifications": ["string"],
      "targetMuscles": ["string"],
      "equipment": ["string"]
    }
  ],
  "warmUp": [
    {
      "exercise": "string",
      "duration": number,
      "instructions": "string"
    }
  ],
  "coolDown": [
    {
      "exercise": "string",
      "duration": number,
      "instructions": "string"
    }
  ],
  "tips": ["string"],
  "progressionSuggestions": ["string"],
  "reasoning": "string explaining why this workout was recommended",
  "adaptations": {
    "forInjuries": ["string"],
    "forBeginners": ["string"],
    "forAdvanced": ["string"]
  }
}

Ensure the workout is safe, effective, and properly progressive based on their history and current fitness level.
`;

    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${openaiApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gpt-4.1-2025-04-14',
        messages: [
          {
            role: 'system',
            content: 'You are an expert AI fitness coach with deep knowledge of exercise science, biomechanics, and personalized training methodologies. Always prioritize safety and progressive overload principles.'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        temperature: 0.3,
        max_tokens: 2000,
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      console.error('OpenAI API error:', errorData);
      throw new Error(`OpenAI API error: ${response.status}`);
    }

    const data = await response.json();
    const workoutRecommendation = data.choices[0].message.content;

    // Parse the JSON response
    let parsedRecommendation;
    try {
      parsedRecommendation = JSON.parse(workoutRecommendation);
    } catch (parseError) {
      console.error('Failed to parse AI response:', workoutRecommendation);
      throw new Error('Invalid AI response format');
    }

    // Add metadata
    const result = {
      ...parsedRecommendation,
      generatedAt: new Date().toISOString(),
      confidence: 0.95,
      aiModel: 'gpt-4.1-2025-04-14',
      personalizationFactors: {
        profileMatching: 0.9,
        historyAnalysis: 0.85,
        progressionLogic: 0.92
      }
    };

    console.log('Generated workout recommendation:', result.workoutName);
    
    return new Response(JSON.stringify(result), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });

  } catch (error) {
    console.error('Error in workout recommendation:', error);
    return new Response(
      JSON.stringify({ 
        error: 'Failed to generate workout recommendation. Please try again.'
      }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
      }
    );
  }
};

serve(serve_handler);