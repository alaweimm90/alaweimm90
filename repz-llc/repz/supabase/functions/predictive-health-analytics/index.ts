import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import "https://deno.land/x/xhr@0.1.0/mod.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

interface PredictiveAnalysisRequest {
  userId: string;
  historicalData: {
    workouts: any[];
    progressMeasurements: any[];
    healthMetrics: any[];
    sleepData: any[];
    nutritionData: any[];
  };
  currentGoals: string[];
  timeframe: string; // '1week', '1month', '3months', '6months'
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

    const requestData: PredictiveAnalysisRequest = await req.json();
    
    // Analyze trends and patterns
    const analysisPrompt = `
You are an expert data scientist specializing in fitness and health analytics. Analyze the following user data to provide predictive insights and recommendations:

USER ID: ${requestData.userId}
ANALYSIS TIMEFRAME: ${requestData.timeframe}
CURRENT GOALS: ${requestData.currentGoals.join(', ')}

HISTORICAL DATA:
Workouts (last 3 months): ${JSON.stringify(requestData.historicalData.workouts.slice(-20), null, 2)}
Progress Measurements: ${JSON.stringify(requestData.historicalData.progressMeasurements.slice(-15), null, 2)}
Health Metrics: ${JSON.stringify(requestData.historicalData.healthMetrics.slice(-15), null, 2)}
Sleep Data: ${JSON.stringify(requestData.historicalData.sleepData.slice(-15), null, 2)}
Nutrition Data: ${JSON.stringify(requestData.historicalData.nutritionData.slice(-10), null, 2)}

Please provide a comprehensive predictive analysis in the following JSON format:
{
  "predictions": {
    "weightGoals": {
      "currentTrend": "string",
      "predictedChange": number,
      "confidenceLevel": number,
      "timeToGoal": "string",
      "factors": ["string"]
    },
    "fitnessGoals": {
      "strengthGains": {
        "predicted": "string",
        "confidence": number,
        "keyExercises": ["string"]
      },
      "cardioImprovement": {
        "predicted": "string",
        "confidence": number,
        "metrics": ["string"]
      },
      "flexibilityProgress": {
        "predicted": "string",
        "confidence": number
      }
    },
    "healthMetrics": {
      "energyLevels": {
        "trend": "string",
        "factors": ["string"]
      },
      "sleepQuality": {
        "trend": "string",
        "recommendations": ["string"]
      },
      "recoveryTime": {
        "predicted": "string",
        "optimizations": ["string"]
      }
    }
  },
  "riskAssessment": {
    "injuryRisk": {
      "level": "low|medium|high",
      "factors": ["string"],
      "preventionStrategies": ["string"]
    },
    "burnoutRisk": {
      "level": "low|medium|high",
      "indicators": ["string"],
      "mitigationStrategies": ["string"]
    },
    "plateauRisk": {
      "level": "low|medium|high",
      "areas": ["string"],
      "strategies": ["string"]
    }
  },
  "recommendations": {
    "immediate": [
      {
        "type": "string",
        "action": "string",
        "reasoning": "string",
        "impact": "string"
      }
    ],
    "shortTerm": [
      {
        "type": "string",
        "action": "string",
        "reasoning": "string",
        "timeframe": "string"
      }
    ],
    "longTerm": [
      {
        "type": "string",
        "action": "string",
        "reasoning": "string",
        "expectedOutcome": "string"
      }
    ]
  },
  "insights": {
    "patterns": ["string"],
    "correlations": ["string"],
    "opportunities": ["string"],
    "challenges": ["string"]
  },
  "confidence": {
    "overall": number,
    "dataQuality": number,
    "trendReliability": number,
    "predictionAccuracy": number
  }
}

Focus on actionable insights and evidence-based predictions. Consider physiological adaptation timelines and individual variability.
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
            content: 'You are an expert data scientist and fitness researcher with deep knowledge of exercise physiology, behavioral patterns, and predictive modeling. Provide accurate, evidence-based analysis while being transparent about limitations and confidence levels.'
          },
          {
            role: 'user',
            content: analysisPrompt
          }
        ],
        temperature: 0.2,
        max_tokens: 2500,
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      console.error('OpenAI API error:', errorData);
      throw new Error(`OpenAI API error: ${response.status}`);
    }

    const data = await response.json();
    const analysisResult = data.choices[0].message.content;

    // Parse the JSON response
    let parsedAnalysis;
    try {
      parsedAnalysis = JSON.parse(analysisResult);
    } catch (parseError) {
      console.error('Failed to parse AI response:', analysisResult);
      throw new Error('Invalid AI response format');
    }

    // Add metadata and statistical confidence
    const result = {
      ...parsedAnalysis,
      metadata: {
        userId: requestData.userId,
        generatedAt: new Date().toISOString(),
        timeframe: requestData.timeframe,
        dataPoints: {
          workouts: requestData.historicalData.workouts.length,
          measurements: requestData.historicalData.progressMeasurements.length,
          healthMetrics: requestData.historicalData.healthMetrics.length,
          sleepData: requestData.historicalData.sleepData.length,
          nutritionData: requestData.historicalData.nutritionData.length
        },
        aiModel: 'gpt-4.1-2025-04-14',
        analysisVersion: '1.0'
      },
      validUntil: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString() // 1 week
    };

    console.log('Generated predictive analysis for user:', requestData.userId);
    
    return new Response(JSON.stringify(result), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });

  } catch (error) {
    console.error('Error in predictive analysis:', error);
    return new Response(
      JSON.stringify({ 
        error: 'Failed to generate health analysis. Please try again.'
      }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
};

serve(serve_handler);