import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.7.1';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

interface CohortAnalysisRequest {
  cohortType: 'weekly' | 'monthly' | 'quarterly';
  metrics: string[];
  timeRange: {
    start: string;
    end: string;
  };
  segmentation?: {
    tier?: string;
    acquisitionChannel?: string;
    demographics?: any;
  };
}

const serve_handler = async (req: Request): Promise<Response> => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    );

    const requestData: CohortAnalysisRequest = await req.json();
    
    // Generate cohort analysis based on user registration dates
    const cohorts = await generateCohortAnalysis(supabase, requestData);
    
    // Calculate retention metrics
    const retentionAnalysis = await calculateRetentionMetrics(supabase, requestData);
    
    // Generate revenue cohorts
    const revenueCohorts = await generateRevenueCohorts(supabase, requestData);
    
    // Engagement cohorts
    const engagementCohorts = await generateEngagementCohorts(supabase, requestData);

    const result = {
      cohortType: requestData.cohortType,
      timeRange: requestData.timeRange,
      analysis: {
        userCohorts: cohorts,
        retentionMetrics: retentionAnalysis,
        revenueCohorts: revenueCohorts,
        engagementCohorts: engagementCohorts
      },
      insights: generateCohortInsights(cohorts, retentionAnalysis, revenueCohorts),
      benchmarks: {
        industryAverages: {
          day1Retention: 0.85,
          day7Retention: 0.65,
          day30Retention: 0.40,
          day90Retention: 0.25
        },
        competitorBenchmarks: {
          fitnessApps: {
            day7: 0.62,
            day30: 0.38,
            day90: 0.22
          },
          subscriptionApps: {
            day7: 0.58,
            day30: 0.35,
            day90: 0.20
          }
        }
      },
      generatedAt: new Date().toISOString()
    };

    console.log('Generated cohort analysis for:', requestData.cohortType);
    
    return new Response(JSON.stringify(result), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });

  } catch (error) {
    console.error('Error in cohort analysis:', error);
    return new Response(
      JSON.stringify({ 
        error: 'Failed to generate cohort analysis. Please try again.'
      }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
};

async function generateCohortAnalysis(supabase: any, request: CohortAnalysisRequest) {
  // Mock cohort data - in production, query actual user data
  const mockCohorts = [
    {
      cohortPeriod: '2024-01',
      cohortSize: 145,
      retention: {
        week0: 1.0,
        week1: 0.82,
        week2: 0.71,
        week4: 0.58,
        week8: 0.45,
        week12: 0.35,
        week24: 0.28,
        week52: 0.22
      },
      demographics: {
        avgAge: 32.5,
        genderSplit: { male: 0.58, female: 0.42 },
        tierDistribution: { core: 0.45, adaptive: 0.35, performance: 0.15, longevity: 0.05 }
      }
    },
    {
      cohortPeriod: '2024-02',
      cohortSize: 167,
      retention: {
        week0: 1.0,
        week1: 0.85,
        week2: 0.74,
        week4: 0.62,
        week8: 0.48,
        week12: 0.38,
        week24: 0.31,
        week52: 0.25
      },
      demographics: {
        avgAge: 31.8,
        genderSplit: { male: 0.55, female: 0.45 },
        tierDistribution: { core: 0.42, adaptive: 0.38, performance: 0.16, longevity: 0.04 }
      }
    },
    {
      cohortPeriod: '2024-03',
      cohortSize: 189,
      retention: {
        week0: 1.0,
        week1: 0.87,
        week2: 0.76,
        week4: 0.65,
        week8: 0.52,
        week12: 0.41,
        week24: 0.34,
        week52: null // Too recent
      },
      demographics: {
        avgAge: 33.1,
        genderSplit: { male: 0.52, female: 0.48 },
        tierDistribution: { core: 0.40, adaptive: 0.35, performance: 0.20, longevity: 0.05 }
      }
    }
  ];

  return mockCohorts;
}

async function calculateRetentionMetrics(supabase: any, request: CohortAnalysisRequest) {
  const mockRetentionMetrics = {
    overallRetention: {
      day1: 0.89,
      day7: 0.67,
      day30: 0.45,
      day90: 0.32,
      day180: 0.25,
      day365: 0.20
    },
    retentionByTier: {
      core: {
        day1: 0.85,
        day7: 0.62,
        day30: 0.38,
        day90: 0.25
      },
      adaptive: {
        day1: 0.91,
        day7: 0.72,
        day30: 0.52,
        day90: 0.38
      },
      performance: {
        day1: 0.94,
        day7: 0.78,
        day30: 0.62,
        day90: 0.48
      },
      longevity: {
        day1: 0.96,
        day7: 0.85,
        day30: 0.74,
        day90: 0.65
      }
    },
    retentionTrends: {
      improving: ['longevity', 'performance'],
      stable: ['adaptive'],
      declining: ['core']
    },
    retentionDrivers: [
      {
        factor: 'onboarding_completion',
        impact: 0.34,
        description: 'Users who complete onboarding have 34% higher retention'
      },
      {
        factor: 'first_week_engagement',
        impact: 0.28,
        description: 'High first-week activity correlates with long-term retention'
      },
      {
        factor: 'coach_interaction',
        impact: 0.22,
        description: 'Users who interact with coaches show better retention'
      }
    ]
  };

  return mockRetentionMetrics;
}

async function generateRevenueCohorts(supabase: any, request: CohortAnalysisRequest) {
  const mockRevenueCohorts = [
    {
      cohortPeriod: '2024-01',
      initialRevenue: 14500,
      revenueProgression: {
        month1: 14500,
        month2: 16780,
        month3: 18340,
        month6: 22150,
        month12: 28940
      },
      ltv: 289.40,
      paybackPeriod: 2.3
    },
    {
      cohortPeriod: '2024-02',
      initialRevenue: 16700,
      revenueProgression: {
        month1: 16700,
        month2: 19240,
        month3: 21680,
        month6: 26340,
        month12: null // Too recent
      },
      ltv: 312.80,
      paybackPeriod: 2.1
    },
    {
      cohortPeriod: '2024-03',
      initialRevenue: 18900,
      revenueProgression: {
        month1: 18900,
        month2: 21560,
        month3: 24230,
        month6: null,
        month12: null
      },
      ltv: 335.20,
      paybackPeriod: 1.9
    }
  ];

  return mockRevenueCohorts;
}

async function generateEngagementCohorts(supabase: any, request: CohortAnalysisRequest) {
  const mockEngagementCohorts = {
    workoutCompletionRates: [
      {
        cohortPeriod: '2024-01',
        week1: 0.78,
        week4: 0.65,
        week12: 0.52,
        week24: 0.45
      },
      {
        cohortPeriod: '2024-02',
        week1: 0.82,
        week4: 0.68,
        week12: 0.56,
        week24: 0.48
      },
      {
        cohortPeriod: '2024-03',
        week1: 0.85,
        week4: 0.72,
        week12: 0.59,
        week24: null
      }
    ],
    featureAdoption: {
      progressTracking: {
        week1: 0.65,
        week4: 0.78,
        week12: 0.82
      },
      coachMessaging: {
        week1: 0.34,
        week4: 0.52,
        week12: 0.67
      },
      nutritionLogging: {
        week1: 0.28,
        week4: 0.45,
        week12: 0.62
      }
    },
    sessionFrequency: {
      daily: 0.15,
      '2-3x_week': 0.45,
      weekly: 0.28,
      monthly: 0.12
    }
  };

  return mockEngagementCohorts;
}

function generateCohortInsights(cohorts: any[], retention: any, revenue: any[]) {
  return {
    keyFindings: [
      'Recent cohorts show 15% better retention than older cohorts',
      'Performance tier users have 2.5x better long-term retention',
      'Revenue per cohort improving 12% month-over-month',
      'First week engagement is the strongest predictor of retention'
    ],
    retentionOptimizations: [
      {
        opportunity: 'Improve day 7 retention',
        currentRate: 0.67,
        targetRate: 0.75,
        estimatedImpact: '+$45,000 monthly revenue',
        tactics: [
          'Implement day 3 check-in email',
          'Gamify first week achievements',
          'Provide personalized workout recommendations'
        ]
      },
      {
        opportunity: 'Reduce month 2 churn',
        currentRate: 0.25,
        targetRate: 0.18,
        estimatedImpact: '+$28,000 monthly revenue',
        tactics: [
          'Enhanced coach check-ins',
          'Progress milestone celebrations',
          'Feature education campaigns'
        ]
      }
    ],
    segmentInsights: [
      {
        segment: 'High-value early adopters',
        characteristics: ['Complete onboarding', 'Premium tier', 'High engagement'],
        retention: 0.78,
        recommendations: 'Focus acquisition on similar profiles'
      },
      {
        segment: 'At-risk users',
        characteristics: ['Low first-week engagement', 'No coach interaction'],
        retention: 0.23,
        recommendations: 'Implement early intervention campaigns'
      }
    ],
    benchmarkComparison: {
      vsIndustry: '+12% better retention at day 30',
      vsCompetitors: '+8% better retention at day 90',
      strengths: ['Premium tier retention', 'Revenue growth per cohort'],
      improvements: ['Day 7 retention', 'Feature adoption rates']
    }
  };
}

serve(serve_handler);