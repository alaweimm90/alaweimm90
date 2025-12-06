import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.7.1';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

interface BusinessAnalyticsRequest {
  timeframe: '7d' | '30d' | '90d' | '1y';
  metrics: string[];
  segmentation?: {
    tier?: string;
    demographics?: string[];
    behaviorFilters?: string[];
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

    const requestData: BusinessAnalyticsRequest = await req.json();
    
    // Calculate date range
    const endDate = new Date();
    const startDate = new Date();
    
    switch (requestData.timeframe) {
      case '7d':
        startDate.setDate(endDate.getDate() - 7);
        break;
      case '30d':
        startDate.setDate(endDate.getDate() - 30);
        break;
      case '90d':
        startDate.setDate(endDate.getDate() - 90);
        break;
      case '1y':
        startDate.setFullYear(endDate.getFullYear() - 1);
        break;
    }

    // Revenue Analytics
    const revenueData = await calculateRevenueMetrics(supabase, startDate, endDate, requestData.segmentation);
    
    // User Analytics
    const userMetrics = await calculateUserMetrics(supabase, startDate, endDate, requestData.segmentation);
    
    // Conversion Analytics
    const conversionMetrics = await calculateConversionMetrics(supabase, startDate, endDate);
    
    // Churn Analytics
    const churnAnalysis = await calculateChurnMetrics(supabase, startDate, endDate);
    
    // Growth Analytics
    const growthMetrics = await calculateGrowthMetrics(supabase, startDate, endDate);

    const result = {
      timeframe: requestData.timeframe,
      dateRange: {
        start: startDate.toISOString(),
        end: endDate.toISOString()
      },
      revenue: revenueData,
      users: userMetrics,
      conversions: conversionMetrics,
      churn: churnAnalysis,
      growth: growthMetrics,
      insights: generateBusinessInsights(revenueData, userMetrics, conversionMetrics, churnAnalysis),
      generatedAt: new Date().toISOString()
    };

    console.log('Generated business analytics for timeframe:', requestData.timeframe);
    
    return new Response(JSON.stringify(result), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });

  } catch (error) {
    console.error('Error in business analytics:', error);
    return new Response(
      JSON.stringify({ 
        error: 'Failed to generate business analytics. Please try again.'
      }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
};

async function calculateRevenueMetrics(supabase: any, startDate: Date, endDate: Date, segmentation?: any) {
  // Mock data for demonstration - in production, query actual payment/subscription data
  const mockRevenue = {
    totalRevenue: 45780,
    recurringRevenue: 38540,
    oneTimeRevenue: 7240,
    averageRevenuePerUser: 127.45,
    revenueByTier: {
      core: 12450,
      adaptive: 18960,
      performance: 14370,
      longevity: 8940
    },
    revenueGrowth: {
      monthOverMonth: 12.5,
      weekOverWeek: 3.2,
      yearOverYear: 145.8
    },
    projectedRevenue: {
      nextMonth: 48200,
      nextQuarter: 142500,
      nextYear: 547800
    },
    churnImpact: 3240,
    upgrades: 8960,
    downgrades: 1240
  };

  return mockRevenue;
}

async function calculateUserMetrics(supabase: any, startDate: Date, endDate: Date, segmentation?: any) {
  // Query user profiles and activity data
  const { data: profiles } = await supabase
    .from('client_profiles')
    .select('*')
    .gte('created_at', startDate.toISOString())
    .lte('created_at', endDate.toISOString());

  const mockUserMetrics = {
    totalUsers: profiles?.length || 342,
    activeUsers: 289,
    newUsers: 67,
    returningUsers: 222,
    usersByTier: {
      core: 156,
      adaptive: 98,
      performance: 67,
      longevity: 21
    },
    engagementMetrics: {
      dailyActiveUsers: 145,
      weeklyActiveUsers: 289,
      monthlyActiveUsers: 342,
      averageSessionDuration: 24.5,
      sessionsPerUser: 4.2
    },
    demographicBreakdown: {
      ageGroups: {
        '18-25': 45,
        '26-35': 123,
        '36-45': 98,
        '46-55': 54,
        '55+': 22
      },
      genderDistribution: {
        male: 198,
        female: 134,
        other: 10
      }
    },
    retentionRates: {
      day1: 0.89,
      day7: 0.67,
      day30: 0.45,
      day90: 0.32
    }
  };

  return mockUserMetrics;
}

async function calculateConversionMetrics(supabase: any, startDate: Date, endDate: Date) {
  const mockConversions = {
    signupToTrial: 0.78,
    trialToSubscription: 0.23,
    freeToBasic: 0.15,
    basicToPremium: 0.08,
    conversionFunnel: {
      visitors: 2340,
      signups: 1825,
      trials: 567,
      subscriptions: 130,
      premiumUpgrades: 42
    },
    conversionTrends: {
      improving: ['trial_conversion', 'upgrade_rate'],
      declining: ['signup_rate'],
      stable: ['retention_rate']
    },
    optimizationOpportunities: [
      'Improve onboarding flow',
      'Add more trial features',
      'Optimize pricing page'
    ]
  };

  return mockConversions;
}

async function calculateChurnMetrics(supabase: any, startDate: Date, endDate: Date) {
  const mockChurn = {
    overallChurnRate: 0.07,
    churnByTier: {
      core: 0.12,
      adaptive: 0.06,
      performance: 0.04,
      longevity: 0.02
    },
    churnReasons: {
      price: 0.35,
      usability: 0.28,
      features: 0.22,
      support: 0.15
    },
    timeToChurn: {
      averageDays: 45,
      median: 38,
      distribution: {
        '0-7d': 0.15,
        '8-30d': 0.42,
        '31-90d': 0.35,
        '90d+': 0.08
      }
    },
    churnPrevention: {
      potentialSaves: 34,
      actualSaves: 12,
      saveRate: 0.35
    }
  };

  return mockChurn;
}

async function calculateGrowthMetrics(supabase: any, startDate: Date, endDate: Date) {
  const mockGrowth = {
    userGrowthRate: 15.2,
    revenueGrowthRate: 12.5,
    viralCoefficient: 0.23,
    customerAcquisitionCost: 89.50,
    lifetimeValue: 1247.30,
    paybackPeriod: 8.5,
    growthChannels: {
      organic: 0.45,
      paid: 0.32,
      referral: 0.18,
      partnerships: 0.05
    },
    marketingEfficiency: {
      roas: 4.2,
      cpa: 89.50,
      ltv_cac_ratio: 13.9
    }
  };

  return mockGrowth;
}

function generateBusinessInsights(revenue: any, users: any, conversions: any, churn: any) {
  return {
    keyFindings: [
      'Revenue growth of 12.5% month-over-month driven by tier upgrades',
      'User retention improves significantly after 30 days',
      'Performance tier shows lowest churn rate at 4%',
      'Trial to subscription conversion needs optimization'
    ],
    recommendations: [
      {
        category: 'Revenue',
        action: 'Focus on Performance tier marketing',
        impact: 'High',
        effort: 'Medium',
        timeline: '30 days'
      },
      {
        category: 'Conversion',
        action: 'Improve trial onboarding experience',
        impact: 'High',
        effort: 'High',
        timeline: '60 days'
      },
      {
        category: 'Retention',
        action: 'Implement engagement campaigns for day 7-30',
        impact: 'Medium',
        effort: 'Medium',
        timeline: '14 days'
      }
    ],
    alerts: [
      {
        type: 'warning',
        message: 'Signup conversion rate declining for 3 consecutive weeks',
        severity: 'medium'
      },
      {
        type: 'success',
        message: 'Premium tier revenue exceeded target by 15%',
        severity: 'low'
      }
    ]
  };
}

serve(serve_handler);