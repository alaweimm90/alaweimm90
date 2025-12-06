import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.7.1';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

interface PerformanceMetric {
  id: string;
  timestamp: string;
  metric_name: string;
  value: number;
  metadata: Record<string, any>;
  severity: 'info' | 'warning' | 'error' | 'critical';
}

interface SystemHealth {
  overall_status: 'healthy' | 'degraded' | 'critical';
  api_response_time: number;
  database_health: boolean;
  error_rate: number;
  active_users: number;
  last_updated: string;
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { action, data } = await req.json();

    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    );

    switch (action) {
      case 'log_performance_metric':
        return await logPerformanceMetric(supabase, data);
      
      case 'get_system_health':
        return await getSystemHealth(supabase);
      
      case 'get_performance_metrics':
        return await getPerformanceMetrics(supabase, data);
      
      case 'log_error':
        return await logError(supabase, data);
      
      case 'get_error_summary':
        return await getErrorSummary(supabase, data);
      
      default:
        throw new Error(`Unknown action: ${action}`);
    }

  } catch (error) {
    console.error('Error in production-monitoring function:', error);
    return new Response(
      JSON.stringify({ 
        error: 'Monitoring operation failed. Please try again.',
        timestamp: new Date().toISOString()
      }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      },
    );
  }
});

async function logPerformanceMetric(supabase: any, metricData: any) {
  const { data, error } = await supabase
    .from('performance_metrics')
    .insert({
      metric_name: metricData.name,
      value: metricData.value,
      metadata: metricData.metadata || {},
      severity: metricData.severity || 'info',
      timestamp: new Date().toISOString()
    });

  if (error) throw error;

  console.log(`Performance metric logged: ${metricData.name} = ${metricData.value}`);

  return new Response(
    JSON.stringify({ success: true, data }),
    { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
  );
}

async function getSystemHealth(supabase: any) {
  const now = new Date();
  const oneHourAgo = new Date(now.getTime() - 60 * 60 * 1000);

  // Get recent performance metrics
  const { data: metrics, error: metricsError } = await supabase
    .from('performance_metrics')
    .select('*')
    .gte('timestamp', oneHourAgo.toISOString())
    .order('timestamp', { ascending: false });

  if (metricsError) throw metricsError;

  // Calculate health indicators
  const apiResponseTimes = metrics
    .filter((m: any) => m.metric_name === 'api_response_time')
    .map((m: any) => m.value);
  
  const errorRates = metrics
    .filter((m: any) => m.metric_name === 'error_rate')
    .map((m: any) => m.value);

  const avgResponseTime = apiResponseTimes.length > 0 
    ? apiResponseTimes.reduce((a: number, b: number) => a + b, 0) / apiResponseTimes.length 
    : 0;

  const avgErrorRate = errorRates.length > 0 
    ? errorRates.reduce((a: number, b: number) => a + b, 0) / errorRates.length 
    : 0;

  // Determine overall status
  let overallStatus: 'healthy' | 'degraded' | 'critical' = 'healthy';
  
  if (avgResponseTime > 2000 || avgErrorRate > 5) {
    overallStatus = 'degraded';
  }
  if (avgResponseTime > 5000 || avgErrorRate > 10) {
    overallStatus = 'critical';
  }

  const healthData: SystemHealth = {
    overall_status: overallStatus,
    api_response_time: Math.round(avgResponseTime),
    database_health: true, // Simplified - could add actual DB health check
    error_rate: Math.round(avgErrorRate * 100) / 100,
    active_users: Math.floor(Math.random() * 100) + 50, // Mock data
    last_updated: now.toISOString()
  };

  console.log('System health calculated:', healthData);

  return new Response(
    JSON.stringify(healthData),
    { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
  );
}

async function getPerformanceMetrics(supabase: any, params: any) {
  const { timeRange = '1h', metricNames } = params;
  
  let startTime: Date;
  const now = new Date();
  
  switch (timeRange) {
    case '1h':
      startTime = new Date(now.getTime() - 60 * 60 * 1000);
      break;
    case '24h':
      startTime = new Date(now.getTime() - 24 * 60 * 60 * 1000);
      break;
    case '7d':
      startTime = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
      break;
    default:
      startTime = new Date(now.getTime() - 60 * 60 * 1000);
  }

  let query = supabase
    .from('performance_metrics')
    .select('*')
    .gte('timestamp', startTime.toISOString())
    .order('timestamp', { ascending: true });

  if (metricNames && metricNames.length > 0) {
    query = query.in('metric_name', metricNames);
  }

  const { data, error } = await query;
  if (error) throw error;

  // Group by metric name
  const groupedMetrics = data.reduce((acc: any, metric: any) => {
    if (!acc[metric.metric_name]) {
      acc[metric.metric_name] = [];
    }
    acc[metric.metric_name].push(metric);
    return acc;
  }, {});

  console.log(`Retrieved ${data.length} performance metrics`);

  return new Response(
    JSON.stringify({
      timeRange,
      startTime: startTime.toISOString(),
      endTime: now.toISOString(),
      metrics: groupedMetrics,
      totalCount: data.length
    }),
    { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
  );
}

async function logError(supabase: any, errorData: any) {
  const { data, error } = await supabase
    .from('error_logs')
    .insert({
      error_message: errorData.message,
      error_stack: errorData.stack,
      user_id: errorData.userId,
      url: errorData.url,
      user_agent: errorData.userAgent,
      severity: errorData.severity || 'error',
      metadata: errorData.metadata || {},
      timestamp: new Date().toISOString()
    });

  if (error) throw error;

  console.log(`Error logged: ${errorData.message}`);

  return new Response(
    JSON.stringify({ success: true, data }),
    { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
  );
}

async function getErrorSummary(supabase: any, params: any) {
  const { timeRange = '24h' } = params;
  
  let startTime: Date;
  const now = new Date();
  
  switch (timeRange) {
    case '1h':
      startTime = new Date(now.getTime() - 60 * 60 * 1000);
      break;
    case '24h':
      startTime = new Date(now.getTime() - 24 * 60 * 60 * 1000);
      break;
    case '7d':
      startTime = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
      break;
    default:
      startTime = new Date(now.getTime() - 24 * 60 * 60 * 1000);
  }

  const { data, error } = await supabase
    .from('error_logs')
    .select('*')
    .gte('timestamp', startTime.toISOString())
    .order('timestamp', { ascending: false });

  if (error) throw error;

  // Group errors by message for summary
  const errorGroups = data.reduce((acc: any, error: any) => {
    const key = error.error_message.substring(0, 100); // Group by first 100 chars
    if (!acc[key]) {
      acc[key] = {
        message: key,
        count: 0,
        latestOccurrence: error.timestamp,
        severity: error.severity,
        urls: new Set()
      };
    }
    acc[key].count++;
    acc[key].urls.add(error.url);
    if (error.timestamp > acc[key].latestOccurrence) {
      acc[key].latestOccurrence = error.timestamp;
    }
    return acc;
  }, {});

  const summary = Object.values(errorGroups).map((group: any) => ({
    ...group,
    urls: Array.from(group.urls)
  }));

  console.log(`Error summary generated: ${data.length} total errors, ${summary.length} unique error types`);

  return new Response(
    JSON.stringify({
      timeRange,
      totalErrors: data.length,
      uniqueErrorTypes: summary.length,
      errorGroups: summary,
      recentErrors: data.slice(0, 10)
    }),
    { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
  );
}