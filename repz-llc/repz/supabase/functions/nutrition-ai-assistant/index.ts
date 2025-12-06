import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import "https://deno.land/x/xhr@0.1.0/mod.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

interface NutritionAIRequest {
  userProfile: {
    age: number;
    weight: number;
    height: number;
    activityLevel: string;
    goals: string[];
    dietaryRestrictions: string[];
    allergies: string[];
    preferences: string[];
  };
  currentMeals: any[];
  fitnessData: {
    workoutIntensity: string;
    weeklyWorkouts: number;
    currentProgram: string;
  };
  healthMetrics: {
    energyLevels: number;
    sleepQuality: number;
    stressLevel: number;
    digestiveHealth: number;
  };
  nutritionGoals: {
    calories: number;
    protein: number;
    carbs: number;
    fats: number;
    fiber: number;
    water: number;
  };
  requestType: 'meal_plan' | 'meal_suggestion' | 'nutrition_analysis' | 'supplement_advice';
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

    const requestData: NutritionAIRequest = await req.json();
    
    // Create specialized prompt based on request type
    let prompt = '';
    
    if (requestData.requestType === 'meal_plan') {
      prompt = `
You are an expert nutritionist and dietitian with extensive knowledge of sports nutrition and metabolic health. Create a personalized meal plan based on the following user data:

USER PROFILE:
- Age: ${requestData.userProfile.age}
- Weight: ${requestData.userProfile.weight}kg
- Height: ${requestData.userProfile.height}cm
- Activity Level: ${requestData.userProfile.activityLevel}
- Goals: ${requestData.userProfile.goals.join(', ')}
- Dietary Restrictions: ${requestData.userProfile.dietaryRestrictions.join(', ') || 'None'}
- Allergies: ${requestData.userProfile.allergies.join(', ') || 'None'}
- Preferences: ${requestData.userProfile.preferences.join(', ')}

FITNESS DATA:
- Workout Intensity: ${requestData.fitnessData.workoutIntensity}
- Weekly Workouts: ${requestData.fitnessData.weeklyWorkouts}
- Current Program: ${requestData.fitnessData.currentProgram}

HEALTH METRICS (1-10 scale):
- Energy Levels: ${requestData.healthMetrics.energyLevels}
- Sleep Quality: ${requestData.healthMetrics.sleepQuality}
- Stress Level: ${requestData.healthMetrics.stressLevel}
- Digestive Health: ${requestData.healthMetrics.digestiveHealth}

NUTRITION GOALS:
- Calories: ${requestData.nutritionGoals.calories}
- Protein: ${requestData.nutritionGoals.protein}g
- Carbs: ${requestData.nutritionGoals.carbs}g
- Fats: ${requestData.nutritionGoals.fats}g
- Fiber: ${requestData.nutritionGoals.fiber}g
- Water: ${requestData.nutritionGoals.water}L

CURRENT MEALS: ${JSON.stringify(requestData.currentMeals, null, 2)}

Please provide a comprehensive meal plan in the following JSON format:
{
  "mealPlan": {
    "breakfast": {
      "name": "string",
      "ingredients": [
        {
          "item": "string",
          "amount": "string",
          "calories": number,
          "protein": number,
          "carbs": number,
          "fats": number
        }
      ],
      "instructions": ["string"],
      "prepTime": number,
      "nutritionSummary": {
        "calories": number,
        "protein": number,
        "carbs": number,
        "fats": number,
        "fiber": number
      },
      "benefits": ["string"]
    },
    "lunch": { /* same structure */ },
    "dinner": { /* same structure */ },
    "snacks": [
      {
        "name": "string",
        "timing": "string",
        "ingredients": ["string"],
        "nutritionSummary": { /* same structure */ }
      }
    ]
  },
  "dailyNutritionSummary": {
    "totalCalories": number,
    "macroBreakdown": {
      "protein": { "grams": number, "percentage": number },
      "carbs": { "grams": number, "percentage": number },
      "fats": { "grams": number, "percentage": number }
    },
    "micronutrients": ["string"],
    "hydration": "string"
  },
  "supplementRecommendations": [
    {
      "supplement": "string",
      "dosage": "string",
      "timing": "string",
      "reasoning": "string",
      "priority": "high|medium|low"
    }
  ],
  "mealTiming": {
    "preWorkout": "string",
    "postWorkout": "string",
    "recommendations": ["string"]
  },
  "alternatives": {
    "vegetarian": ["string"],
    "quickOptions": ["string"],
    "batchCooking": ["string"]
  },
  "tips": ["string"],
  "nutritionEducation": ["string"]
}
`;
    } else if (requestData.requestType === 'nutrition_analysis') {
      prompt = `
Analyze the current nutrition intake and provide detailed insights:

CURRENT MEALS: ${JSON.stringify(requestData.currentMeals, null, 2)}
NUTRITION GOALS: ${JSON.stringify(requestData.nutritionGoals, null, 2)}
USER PROFILE: ${JSON.stringify(requestData.userProfile, null, 2)}

Provide a detailed analysis in JSON format with nutritional gaps, recommendations, and optimizations.
`;
    }

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
            content: 'You are an expert registered dietitian nutritionist with specialization in sports nutrition, metabolic health, and personalized nutrition planning. Always provide evidence-based recommendations and consider individual needs, preferences, and health conditions.'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        temperature: 0.3,
        max_tokens: 3000,
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      console.error('OpenAI API error:', errorData);
      throw new Error(`OpenAI API error: ${response.status}`);
    }

    const data = await response.json();
    const nutritionRecommendation = data.choices[0].message.content;

    // Parse the JSON response
    let parsedRecommendation;
    try {
      parsedRecommendation = JSON.parse(nutritionRecommendation);
    } catch (parseError) {
      console.error('Failed to parse AI response:', nutritionRecommendation);
      throw new Error('Invalid AI response format');
    }

    // Add metadata
    const result = {
      ...parsedRecommendation,
      metadata: {
        generatedAt: new Date().toISOString(),
        requestType: requestData.requestType,
        aiModel: 'gpt-4.1-2025-04-14',
        validFor: '7 days',
        disclaimer: 'This AI-generated nutrition advice is for informational purposes only. Consult with a healthcare provider or registered dietitian for personalized medical nutrition therapy.'
      }
    };

    console.log('Generated nutrition recommendation:', requestData.requestType);
    
    return new Response(JSON.stringify(result), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });

  } catch (error) {
    console.error('Error in nutrition AI:', error);
    return new Response(
      JSON.stringify({ 
        error: 'Failed to generate nutrition recommendation. Please try again.'
      }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
};

serve(serve_handler);