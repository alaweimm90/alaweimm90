import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.7.1';
import { z } from 'https://deno.land/x/zod@v3.22.4/mod.ts';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

// Validation schemas
const variantSchema = z.object({
  name: z.string().min(1).max(100),
  description: z.string().max(500),
  weight: z.number().min(0).max(1),
  config: z.record(z.any()),
});

const testConfigSchema = z.object({
  name: z.string().min(1).max(200),
  hypothesis: z.string().min(1).max(1000),
  variants: z.array(variantSchema).min(2).max(10),
  metrics: z.array(z.string().max(100)).min(1).max(20),
  duration: z.number().int().min(1).max(365),
  targetSampleSize: z.number().int().min(10).max(1000000),
});

const eventDataSchema = z.object({
  eventType: z.string().min(1).max(100),
  value: z.number().optional(),
  metadata: z.record(z.any()).optional(),
});

const abTestRequestSchema = z.object({
  action: z.enum(['create', 'get_results', 'list_tests', 'assign_variant', 'record_event']),
  testId: z.string().max(200).optional(),
  userId: z.string().max(200).optional(),
  testConfig: testConfigSchema.optional(),
  eventData: eventDataSchema.optional(),
});

const serve_handler = async (req: Request): Promise<Response> => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    );

    // Parse and validate input
    const body = await req.json();
    const validation = abTestRequestSchema.safeParse(body);

    if (!validation.success) {
      const errors = validation.error.errors.map(e => `${e.path.join('.')}: ${e.message}`).join(', ');
      console.log('Validation failed:', errors);
      return new Response(
        JSON.stringify({ error: `Invalid request format: ${errors}` }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    const requestData = validation.data;
    
    // Validate required fields based on action
    if (requestData.action === 'create' && !requestData.testConfig) {
      return new Response(
        JSON.stringify({ error: 'testConfig is required for create action' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }
    if ((requestData.action === 'assign_variant' || requestData.action === 'record_event' || requestData.action === 'get_results') && !requestData.testId) {
      return new Response(
        JSON.stringify({ error: 'testId is required for this action' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }
    if ((requestData.action === 'assign_variant' || requestData.action === 'record_event') && !requestData.userId) {
      return new Response(
        JSON.stringify({ error: 'userId is required for this action' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }
    if (requestData.action === 'record_event' && !requestData.eventData) {
      return new Response(
        JSON.stringify({ error: 'eventData is required for record_event action' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }
    
    let result;
    
    switch (requestData.action) {
      case 'create':
        result = await createABTest(supabase, requestData.testConfig!);
        break;
      case 'assign_variant':
        result = await assignVariant(supabase, requestData.testId!, requestData.userId!);
        break;
      case 'record_event':
        result = await recordEvent(supabase, requestData.testId!, requestData.userId!, requestData.eventData!);
        break;
      case 'get_results':
        result = await getTestResults(supabase, requestData.testId!);
        break;
      case 'list_tests':
        result = await listActiveTests(supabase);
        break;
      default:
        throw new Error('Invalid action');
    }

    console.log('A/B Test action completed:', requestData.action);
    
    return new Response(JSON.stringify(result), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });

  } catch (error) {
    console.error('Error in A/B testing:', error);
    // Return generic error message, log details server-side
    return new Response(
      JSON.stringify({ 
        error: 'A/B testing operation failed. Please try again.'
      }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
};

async function createABTest(supabase: any, config: any) {
  const testId = `test_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;
  
  // In production, store in database
  const test = {
    id: testId,
    name: config.name,
    hypothesis: config.hypothesis,
    status: 'active',
    createdAt: new Date().toISOString(),
    variants: config.variants,
    metrics: config.metrics,
    duration: config.duration,
    targetSampleSize: config.targetSampleSize,
    currentSampleSize: 0,
    results: null
  };

  // Mock storage - in production, save to database
  console.log('Created A/B test:', test);
  
  return {
    success: true,
    testId: testId,
    test: test
  };
}

async function assignVariant(supabase: any, testId: string, userId: string) {
  // Get test configuration
  const test = await getTestById(testId);
  
  if (!test) {
    throw new Error('Test not found');
  }

  // Check if user already has assignment
  const existingAssignment = await getUserAssignment(testId, userId);
  
  if (existingAssignment) {
    return existingAssignment;
  }

  // Assign variant based on weights
  const variant = selectVariantByWeight(test.variants, userId);
  
  const assignment = {
    testId,
    userId,
    variant: variant.name,
    assignedAt: new Date().toISOString(),
    config: variant.config
  };

  // In production, store assignment in database
  console.log('Assigned variant:', assignment);
  
  return assignment;
}

async function recordEvent(supabase: any, testId: string, userId: string, eventData: any) {
  const event = {
    testId,
    userId,
    eventType: eventData.eventType,
    value: eventData.value,
    metadata: eventData.metadata,
    timestamp: new Date().toISOString()
  };

  // In production, store event in database
  console.log('Recorded A/B test event:', event);
  
  return {
    success: true,
    event: event
  };
}

async function getTestResults(supabase: any, testId: string) {
  // Mock results - in production, calculate from actual data
  const mockResults = {
    testId,
    status: 'active',
    duration: 30,
    daysRunning: 14,
    sampleSize: 1250,
    targetSampleSize: 2000,
    variants: [
      {
        name: 'control',
        sampleSize: 625,
        conversionRate: 0.156,
        confidence: 0.95,
        metrics: {
          signups: 98,
          conversions: 15,
          revenue: 1890,
          engagementRate: 0.67
        }
      },
      {
        name: 'variant_a',
        sampleSize: 625,
        conversionRate: 0.184,
        confidence: 0.89,
        metrics: {
          signups: 115,
          conversions: 21,
          revenue: 2340,
          engagementRate: 0.72
        }
      }
    ],
    statisticalSignificance: {
      isSignificant: false,
      confidence: 0.89,
      pValue: 0.11,
      minimumDetectableEffect: 0.05,
      powerAnalysis: 0.78
    },
    insights: {
      leadingVariant: 'variant_a',
      improvementPercentage: 17.9,
      estimatedImpact: '+$450 monthly revenue',
      recommendation: 'Continue test for 2 more weeks to reach statistical significance',
      riskAssessment: 'Low risk - positive trend observed'
    },
    timeline: {
      startDate: '2024-01-15',
      expectedEndDate: '2024-02-14',
      actualEndDate: null
    }
  };

  return mockResults;
}

async function listActiveTests(supabase: any) {
  // Mock active tests
  const mockTests = [
    {
      id: 'test_pricing_page_v2',
      name: 'Pricing Page Redesign',
      status: 'active',
      hypothesis: 'New pricing layout will increase conversions by 15%',
      startDate: '2024-01-15',
      variants: ['control', 'new_layout'],
      metrics: ['conversion_rate', 'revenue'],
      progress: 0.65
    },
    {
      id: 'test_onboarding_flow',
      name: 'Streamlined Onboarding',
      status: 'active',
      hypothesis: 'Reduced onboarding steps will improve completion rate',
      startDate: '2024-01-20',
      variants: ['current', 'simplified'],
      metrics: ['completion_rate', 'time_to_complete'],
      progress: 0.42
    },
    {
      id: 'test_email_subject_lines',
      name: 'Welcome Email Subject Lines',
      status: 'completed',
      hypothesis: 'Personalized subject lines increase open rates',
      startDate: '2024-01-01',
      endDate: '2024-01-14',
      variants: ['generic', 'personalized'],
      metrics: ['open_rate', 'click_rate'],
      winner: 'personalized',
      improvement: '23% higher open rate'
    }
  ];

  return {
    activeTests: mockTests.filter(t => t.status === 'active'),
    completedTests: mockTests.filter(t => t.status === 'completed'),
    totalTests: mockTests.length
  };
}

// Helper functions
async function getTestById(testId: string) {
  // Mock test data - in production, query database
  return {
    id: testId,
    variants: [
      { name: 'control', weight: 0.5, config: { layout: 'current' } },
      { name: 'variant_a', weight: 0.5, config: { layout: 'new' } }
    ]
  };
}

async function getUserAssignment(testId: string, userId: string) {
  // Mock assignment check - in production, query database
  return null;
}

function selectVariantByWeight(variants: any[], userId: string) {
  // Use userId hash for consistent assignment
  const hash = simpleHash(userId);
  const random = (hash % 10000) / 10000;
  
  let cumulativeWeight = 0;
  for (const variant of variants) {
    cumulativeWeight += variant.weight;
    if (random <= cumulativeWeight) {
      return variant;
    }
  }
  
  return variants[0]; // Fallback
}

function simpleHash(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  return Math.abs(hash);
}

serve(serve_handler);