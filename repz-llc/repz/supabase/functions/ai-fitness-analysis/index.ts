import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.7.1'

// Type definitions
type SupabaseClient = ReturnType<typeof createClient>;

interface ClientProfile {
  client_name: string;
  age_years: number;
  start_weight_kg: number;
  height_cm: number;
  primary_goal: string;
  activity_level: string;
  tdee_kcal_day: number;
  training_days_per_week: number;
  subscription_tier: string;
  current_week: number;
  target_weight_kg?: number;
  program_start_date?: string;
}

interface AnalysisData {
  [key: string]: unknown;
}

interface AIResponse {
  confidence?: number;
  recommendations?: string;
  generated_at?: string;
  [key: string]: unknown;
}

const openAIApiKey = Deno.env.get('OPENAI_API_KEY');
const supabaseUrl = Deno.env.get('SUPABASE_URL')!;
const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!;

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const supabase = createClient(supabaseUrl, supabaseServiceKey);
    const { clientId, analysisType, data } = await req.json();

    console.log(`AI Analysis request: ${analysisType} for client ${clientId}`);

    if (!openAIApiKey) {
      throw new Error('OpenAI API key not configured');
    }

    // Get client profile and recent data
    const { data: clientProfile, error: profileError } = await supabase
      .from('client_profiles')
      .select('*')
      .eq('auth_user_id', clientId)
      .single();

    if (profileError) {
      throw new Error(`Failed to get client profile: ${profileError.message}`);
    }

    let aiResponse;
    
    switch (analysisType) {
      case 'nutrition_recommendations':
        aiResponse = await generateNutritionRecommendations(clientProfile, data);
        break;
      case 'progress_prediction':
        aiResponse = await generateProgressPrediction(clientProfile, data);
        break;
      case 'program_adjustments':
        aiResponse = await generateProgramAdjustments(clientProfile, data);
        break;
      case 'form_analysis':
        aiResponse = await generateFormAnalysis(clientProfile, data);
        break;
      default:
        throw new Error(`Unknown analysis type: ${analysisType}`);
    }

    // Store AI analysis result
    const { error: insertError } = await supabase
      .from('ai_analysis_results')
      .insert({
        client_id: clientId,
        analysis_type: analysisType,
        input_data: data,
        ai_response: aiResponse,
        confidence_score: aiResponse.confidence || 0.8,
        created_at: new Date().toISOString()
      });

    if (insertError) {
      console.error('Failed to store AI analysis:', insertError);
    }

    return new Response(JSON.stringify({ 
      success: true, 
      analysis: aiResponse,
      timestamp: new Date().toISOString()
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });

  } catch (error) {
    console.error('Error in AI analysis:', error);
    return new Response(JSON.stringify({ 
      success: false, 
      error: 'AI analysis failed. Please try again.' 
    }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }
});

async function generateNutritionRecommendations(clientProfile: ClientProfile, weeklyData: AnalysisData): Promise<AIResponse> {
  const prompt = `As an elite sports nutritionist, analyze this client's profile and recent data to provide personalized nutrition recommendations.

Client Profile:
- Name: ${clientProfile.client_name}
- Age: ${clientProfile.age_years}
- Weight: ${clientProfile.start_weight_kg}kg
- Height: ${clientProfile.height_cm}cm
- Goal: ${clientProfile.primary_goal}
- Activity Level: ${clientProfile.activity_level}
- TDEE: ${clientProfile.tdee_kcal_day} kcal/day
- Training Days: ${clientProfile.training_days_per_week}/week

Recent Weekly Data:
${JSON.stringify(weeklyData, null, 2)}

Provide specific, actionable nutrition recommendations including:
1. Caloric intake adjustments
2. Macronutrient distribution
3. Meal timing around workouts
4. Supplement suggestions
5. Hydration targets
6. Progress indicators to track

Format as structured JSON with sections for each recommendation type.`;

  return await callOpenAI(prompt, 'nutrition_expert');
}

async function generateProgressPrediction(clientProfile: ClientProfile, historicalData: AnalysisData): Promise<AIResponse> {
  const prompt = `As a data-driven fitness coach, analyze this client's historical progress data to predict future outcomes.

Client Profile:
- Current Weight: ${clientProfile.start_weight_kg}kg
- Target Weight: ${clientProfile.target_weight_kg}kg
- Goal: ${clientProfile.primary_goal}
- Program Start: ${clientProfile.program_start_date}
- Current Week: ${clientProfile.current_week}

Historical Data:
${JSON.stringify(historicalData, null, 2)}

Provide predictions for:
1. Weight/body composition changes (next 4-12 weeks)
2. Strength progression estimates
3. Performance milestones
4. Potential plateaus and how to overcome them
5. Timeline to reach goals
6. Risk factors and recommendations

Include confidence levels for each prediction.`;

  return await callOpenAI(prompt, 'data_analyst');
}

async function generateProgramAdjustments(clientProfile: ClientProfile, performanceData: AnalysisData): Promise<AIResponse> {
  const prompt = `As an elite strength coach, analyze this client's recent performance data and suggest program modifications.

Client Profile:
- Tier: ${clientProfile.subscription_tier}
- Goal: ${clientProfile.primary_goal}
- Training Experience: ${clientProfile.current_week} weeks
- Available Days: ${clientProfile.training_days_per_week}/week

Recent Performance Data:
${JSON.stringify(performanceData, null, 2)}

Suggest specific program adjustments:
1. Volume modifications (sets/reps)
2. Intensity changes (weight/effort)
3. Exercise substitutions
4. Recovery adjustments
5. Periodization tweaks
6. Progression strategies

Explain the reasoning behind each adjustment and expected outcomes.`;

  return await callOpenAI(prompt, 'strength_coach');
}

async function generateFormAnalysis(clientProfile: ClientProfile, formData: AnalysisData): Promise<AIResponse> {
  const prompt = `As a movement specialist, analyze this exercise form data and provide detailed feedback.

Client: ${clientProfile.client_name}
Exercise: ${formData.exerciseName}
Form Score: ${formData.formScore}/10
Confidence: ${formData.confidenceScore}

Analysis Data:
${JSON.stringify(formData.analysisData, null, 2)}

Provide:
1. Detailed form breakdown
2. Specific areas for improvement
3. Corrective cues and exercises
4. Injury risk assessment
5. Progression recommendations
6. Alternative exercises if needed

Be specific and actionable in your feedback.`;

  return await callOpenAI(prompt, 'movement_specialist');
}

async function callOpenAI(prompt: string, role: string): Promise<AIResponse> {
  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${openAIApiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'gpt-4.1-2025-04-14',
      messages: [
        {
          role: 'system',
          content: `You are an expert ${role}. Provide detailed, scientific, and actionable advice. Always structure your response as valid JSON with clear sections and confidence scores where appropriate.`
        },
        {
          role: 'user',
          content: prompt
        }
      ],
      temperature: 0.7,
      max_tokens: 2000
    }),
  });

  const data = await response.json();
  
  if (!response.ok) {
    throw new Error(`OpenAI API error: ${data.error?.message || 'Unknown error'}`);
  }

  try {
    return JSON.parse(data.choices[0].message.content);
  } catch (parseError) {
    // If JSON parsing fails, return structured response
    return {
      recommendations: data.choices[0].message.content,
      confidence: 0.8,
      generated_at: new Date().toISOString()
    };
  }
}