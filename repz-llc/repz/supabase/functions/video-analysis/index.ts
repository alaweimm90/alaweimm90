import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import "https://deno.land/x/xhr@0.1.0/mod.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

// Exercise form analysis prompts and scoring criteria
const EXERCISE_ANALYSIS_PROMPTS = {
  squat: {
    keyPoints: [
      "Feet shoulder-width apart with toes slightly pointed out",
      "Knees track over toes without caving inward", 
      "Back maintains neutral spine throughout movement",
      "Hips initiate the movement by sitting back",
      "Depth reaches at least hip crease below knee level",
      "Chest stays up and core remains engaged"
    ],
    commonErrors: [
      "Knee valgus (knees caving inward)",
      "Forward lean with excessive anterior pelvic tilt",
      "Insufficient depth",
      "Weight shifting to toes instead of staying on heels",
      "Rounded upper back"
    ]
  },
  deadlift: {
    keyPoints: [
      "Feet hip-width apart under the barbell",
      "Neutral spine maintained throughout the lift",
      "Bar stays close to body during entire movement", 
      "Hips and shoulders rise at the same rate",
      "Full hip extension at the top",
      "Controlled descent with hip hinge pattern"
    ],
    commonErrors: [
      "Rounded back (flexion)",
      "Bar drifting away from body",
      "Hyperextension at the top",
      "Knees shooting forward",
      "Uneven bar path"
    ]
  },
  bench_press: {
    keyPoints: [
      "Five points of contact (head, upper back, glutes, both feet)",
      "Shoulder blades retracted and depressed",
      "Bar touches chest at nipple line",
      "Elbows at 45-degree angle to torso",
      "Full range of motion with controlled tempo",
      "Stable core and leg drive"
    ],
    commonErrors: [
      "Flaring elbows too wide",
      "Bouncing bar off chest", 
      "Partial range of motion",
      "Loss of shoulder blade retraction",
      "Excessive arch in lower back"
    ]
  }
};

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { videoFrames, exerciseType, analysisType = 'form_check' } = await req.json();

    if (!videoFrames || !Array.isArray(videoFrames) || videoFrames.length === 0) {
      throw new Error('Video frames are required for analysis');
    }

    if (!exerciseType || !EXERCISE_ANALYSIS_PROMPTS[exerciseType as keyof typeof EXERCISE_ANALYSIS_PROMPTS]) {
      throw new Error(`Unsupported exercise type: ${exerciseType}`);
    }

    const OPENAI_API_KEY = Deno.env.get('OPENAI_API_KEY');
    if (!OPENAI_API_KEY) {
      throw new Error('OpenAI API key not configured');
    }

    const exerciseData = EXERCISE_ANALYSIS_PROMPTS[exerciseType as keyof typeof EXERCISE_ANALYSIS_PROMPTS];
    
    console.log(`Analyzing ${videoFrames.length} frames for ${exerciseType} exercise`);

    // Prepare images for GPT-4 Vision analysis
    const imageMessages = videoFrames.slice(0, 10).map((frame: string, index: number) => ({
      type: "image_url",
      image_url: {
        url: frame.startsWith('data:') ? frame : `data:image/jpeg;base64,${frame}`,
        detail: "high"
      }
    }));

    const systemPrompt = `You are an expert exercise physiologist and biomechanics analyst. Analyze the provided video frames showing a ${exerciseType} exercise performance.

Key Points to Evaluate for ${exerciseType}:
${exerciseData.keyPoints.map((point, i) => `${i + 1}. ${point}`).join('\n')}

Common Errors to Watch For:
${exerciseData.commonErrors.map((error, i) => `- ${error}`).join('\n')}

Provide analysis in the following JSON format:
{
  "overallScore": 85,
  "phase": "eccentric",
  "keyPointsAnalysis": [
    {
      "point": "Knee tracking",
      "score": 90,
      "feedback": "Excellent knee alignment throughout movement",
      "status": "good"
    }
  ],
  "criticalErrors": [
    {
      "error": "Slight forward lean",
      "severity": "minor",
      "correction": "Focus on sitting back into hips earlier"
    }
  ],
  "recommendations": [
    "Maintain chest up position",
    "Focus on controlled tempo"
  ],
  "confidenceLevel": 85
}

Score from 0-100, where 80+ is excellent, 60-79 is good, 40-59 needs work, below 40 is poor form.`;

    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${OPENAI_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gpt-4o',
        messages: [
          {
            role: 'system',
            content: systemPrompt
          },
          {
            role: 'user',
            content: [
              {
                type: "text",
                text: `Please analyze this ${exerciseType} exercise performance from these video frames. Focus on form, technique, and safety.`
              },
              ...imageMessages
            ]
          }
        ],
        max_tokens: 1500,
        temperature: 0.3
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('OpenAI API error:', response.status, errorText);
      throw new Error(`OpenAI API error: ${response.status}`);
    }

    const data = await response.json();
    const analysisText = data.choices[0].message.content;

    // Try to parse JSON from the response
    let analysisResult;
    try {
      const jsonMatch = analysisText.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        analysisResult = JSON.parse(jsonMatch[0]);
      } else {
        throw new Error('No JSON found in response');
      }
    } catch (parseError) {
      // Fallback: create structured response from text
      console.warn('Could not parse JSON response, creating fallback structure');
      analysisResult = {
        overallScore: 75,
        phase: "analysis_complete",
        keyPointsAnalysis: [
          {
            point: "General Form",
            score: 75,
            feedback: analysisText.substring(0, 200) + "...",
            status: "needs_work"
          }
        ],
        criticalErrors: [],
        recommendations: ["Review the detailed analysis provided"],
        confidenceLevel: 70,
        rawAnalysis: analysisText
      };
    }

    // Add metadata
    analysisResult.exerciseType = exerciseType;
    analysisResult.analysisType = analysisType;
    analysisResult.framesAnalyzed = Math.min(videoFrames.length, 10);
    analysisResult.timestamp = new Date().toISOString();

    console.log(`Analysis complete for ${exerciseType}:`, {
      score: analysisResult.overallScore,
      errors: analysisResult.criticalErrors?.length || 0,
      confidence: analysisResult.confidenceLevel
    });

    return new Response(
      JSON.stringify(analysisResult),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      },
    );

  } catch (error) {
    console.error('Error in video-analysis function:', error);
    return new Response(
      JSON.stringify({ 
        error: error.message,
        timestamp: new Date().toISOString()
      }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      },
    );
  }
});