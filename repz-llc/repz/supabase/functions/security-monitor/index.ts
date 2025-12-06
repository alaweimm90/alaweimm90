import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.7.1';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

interface SecurityEvent {
  userId?: string;
  eventType: string;
  eventCategory: string;
  eventDetails?: any;
  ipAddress?: string;
  userAgent?: string;
  sessionId?: string;
  riskScore?: number;
}

interface ComplianceEvent {
  complianceType: string;
  eventType: string;
  userId?: string;
  dataSubject?: string;
  legalBasis?: string;
  purpose?: string;
  retentionPeriod?: string;
  eventDetails?: any;
}

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? '',
    );

    const { action, data } = await req.json();

    switch (action) {
      case 'log_security_event':
        return await logSecurityEvent(supabase, data);
      
      case 'log_compliance_event':
        return await logComplianceEvent(supabase, data);
      
      case 'analyze_risk':
        return await analyzeRisk(supabase, data);
      
      case 'get_security_dashboard':
        return await getSecurityDashboard(supabase, data);
      
      case 'check_suspicious_activity':
        return await checkSuspiciousActivity(supabase, data);
      
      default:
        return new Response(
          JSON.stringify({ error: 'Invalid action' }),
          { 
            status: 400, 
            headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
          }
        );
    }
  } catch (error) {
    console.error('Security Monitor Error:', error);
    return new Response(
      JSON.stringify({ error: 'Security operation failed. Please try again.' }),
      { 
        status: 500, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    );
  }
});

async function logSecurityEvent(supabase: any, eventData: SecurityEvent) {
  console.log('Logging security event:', eventData);

  const { data, error } = await supabase.rpc('log_security_event', {
    p_user_id: eventData.userId || null,
    p_event_type: eventData.eventType,
    p_event_category: eventData.eventCategory,
    p_event_details: eventData.eventDetails || {},
    p_ip_address: eventData.ipAddress || null,
    p_user_agent: eventData.userAgent || null,
    p_session_id: eventData.sessionId || null,
    p_risk_score: eventData.riskScore || 0
  });

  if (error) {
    throw new Error(`Failed to log security event: ${error.message}`);
  }

  // Check if this is a high-risk event that requires immediate action
  if (eventData.riskScore && eventData.riskScore > 70) {
    await handleHighRiskEvent(supabase, eventData, data);
  }

  return new Response(
    JSON.stringify({ 
      success: true, 
      eventId: data,
      message: 'Security event logged successfully' 
    }),
    { 
      status: 200, 
      headers: { 'Content-Type': 'application/json', ...corsHeaders } 
    }
  );
}

async function logComplianceEvent(supabase: any, eventData: ComplianceEvent) {
  console.log('Logging compliance event:', eventData);

  const { data, error } = await supabase.rpc('log_compliance_event', {
    p_compliance_type: eventData.complianceType,
    p_event_type: eventData.eventType,
    p_user_id: eventData.userId || null,
    p_data_subject: eventData.dataSubject || null,
    p_legal_basis: eventData.legalBasis || null,
    p_purpose: eventData.purpose || null,
    p_retention_period: eventData.retentionPeriod || null,
    p_event_details: eventData.eventDetails || {}
  });

  if (error) {
    throw new Error(`Failed to log compliance event: ${error.message}`);
  }

  return new Response(
    JSON.stringify({ 
      success: true, 
      eventId: data,
      message: 'Compliance event logged successfully' 
    }),
    { 
      status: 200, 
      headers: { 'Content-Type': 'application/json', ...corsHeaders } 
    }
  );
}

async function analyzeRisk(supabase: any, requestData: any) {
  const { userId, sessionData, activityPattern } = requestData;
  
  let riskScore = 0;
  const riskFactors: string[] = [];

  // Analyze IP address changes
  if (sessionData.ipAddress) {
    const { data: recentSessions } = await supabase
      .from('secure_sessions')
      .select('ip_address, location_country')
      .eq('user_id', userId)
      .gte('created_at', new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString())
      .order('created_at', { ascending: false })
      .limit(5);

    if (recentSessions && recentSessions.length > 0) {
      const uniqueIPs = new Set(recentSessions.map((s: any) => s.ip_address));
      const uniqueCountries = new Set(recentSessions.map((s: any) => s.location_country));
      
      if (uniqueIPs.size > 3) {
        riskScore += 30;
        riskFactors.push('Multiple IP addresses detected');
      }
      
      if (uniqueCountries.size > 1) {
        riskScore += 40;
        riskFactors.push('Multiple countries detected');
      }
    }
  }

  // Analyze login frequency
  if (activityPattern.loginAttempts > 10) {
    riskScore += 25;
    riskFactors.push('Excessive login attempts');
  }

  // Analyze time patterns
  const hour = new Date().getHours();
  if (hour < 6 || hour > 22) {
    riskScore += 10;
    riskFactors.push('Unusual time activity');
  }

  // Analyze failed attempts
  if (activityPattern.failedAttempts > 3) {
    riskScore += 20;
    riskFactors.push('Multiple failed attempts');
  }

  const riskLevel = riskScore > 70 ? 'HIGH' : riskScore > 40 ? 'MEDIUM' : 'LOW';

  return new Response(
    JSON.stringify({
      success: true,
      riskScore,
      riskLevel,
      riskFactors,
      recommendations: generateSecurityRecommendations(riskScore, riskFactors)
    }),
    { 
      status: 200, 
      headers: { 'Content-Type': 'application/json', ...corsHeaders } 
    }
  );
}

async function getSecurityDashboard(supabase: any, requestData: any) {
  const { timeframe = '24h' } = requestData;
  
  const timeframeHours = timeframe === '24h' ? 24 : timeframe === '7d' ? 168 : 720;
  const since = new Date(Date.now() - timeframeHours * 60 * 60 * 1000).toISOString();

  // Get security events summary
  const { data: securityEvents } = await supabase
    .from('security_audit_logs')
    .select('event_type, event_category, risk_score, requires_action, created_at')
    .gte('created_at', since);

  // Get compliance events
  const { data: complianceEvents } = await supabase
    .from('compliance_events')
    .select('compliance_type, event_type, status, created_at')
    .gte('created_at', since);

  // Get active sessions
  const { data: activeSessions } = await supabase
    .from('secure_sessions')
    .select('*')
    .gte('last_activity', new Date(Date.now() - 30 * 60 * 1000).toISOString()); // Active in last 30 minutes

  // Generate dashboard metrics
  const metrics = {
    totalSecurityEvents: securityEvents?.length || 0,
    highRiskEvents: securityEvents?.filter((e: any) => e.risk_score > 70).length || 0,
    requiresAction: securityEvents?.filter((e: any) => e.requires_action).length || 0,
    complianceEvents: complianceEvents?.length || 0,
    activeSessions: activeSessions?.length || 0,
    suspiciousSessions: activeSessions?.filter((s: any) => s.is_suspicious).length || 0
  };

  return new Response(
    JSON.stringify({
      success: true,
      metrics,
      securityEvents: securityEvents?.slice(0, 50) || [],
      complianceEvents: complianceEvents?.slice(0, 50) || [],
      activeSessions: activeSessions?.slice(0, 100) || []
    }),
    { 
      status: 200, 
      headers: { 'Content-Type': 'application/json', ...corsHeaders } 
    }
  );
}

async function checkSuspiciousActivity(supabase: any, requestData: any) {
  const { userId, sessionId } = requestData;

  // Check for concurrent sessions
  const { data: sessions } = await supabase
    .from('secure_sessions')
    .select('*')
    .eq('user_id', userId)
    .gte('last_activity', new Date(Date.now() - 15 * 60 * 1000).toISOString());

  const isSuspicious = sessions && sessions.length > 3; // More than 3 active sessions

  if (isSuspicious) {
    // Log suspicious activity
    await supabase.rpc('log_security_event', {
      p_user_id: userId,
      p_event_type: 'SUSPICIOUS_ACTIVITY',
      p_event_category: 'SESSION_SECURITY',
      p_event_details: { 
        reason: 'Multiple concurrent sessions',
        sessionCount: sessions.length,
        sessionId
      },
      p_risk_score: 60
    });
  }

  return new Response(
    JSON.stringify({
      success: true,
      isSuspicious,
      sessionCount: sessions?.length || 0,
      activeSessions: sessions || []
    }),
    { 
      status: 200, 
      headers: { 'Content-Type': 'application/json', ...corsHeaders } 
    }
  );
}

async function handleHighRiskEvent(supabase: any, eventData: SecurityEvent, eventId: string) {
  console.log('Handling high-risk security event:', eventId);
  
  // Log additional compliance event for high-risk situations
  await supabase.rpc('log_compliance_event', {
    p_compliance_type: 'SECURITY',
    p_event_type: 'HIGH_RISK_DETECTED',
    p_user_id: eventData.userId,
    p_event_details: {
      originalEventId: eventId,
      riskScore: eventData.riskScore,
      eventType: eventData.eventType
    }
  });

  // In a real implementation, you might:
  // - Send notifications to security team
  // - Trigger automated responses
  // - Update user security settings
  // - Require additional authentication
}

function generateSecurityRecommendations(riskScore: number, riskFactors: string[]): string[] {
  const recommendations: string[] = [];

  if (riskScore > 70) {
    recommendations.push('Immediate security review required');
    recommendations.push('Consider temporarily suspending account');
    recommendations.push('Require additional authentication factors');
  } else if (riskScore > 40) {
    recommendations.push('Enable additional security monitoring');
    recommendations.push('Review recent account activity');
    recommendations.push('Consider enabling 2FA if not already active');
  }

  if (riskFactors.includes('Multiple IP addresses detected')) {
    recommendations.push('Verify legitimate access from multiple locations');
  }

  if (riskFactors.includes('Multiple countries detected')) {
    recommendations.push('Confirm international access is authorized');
  }

  if (riskFactors.includes('Excessive login attempts')) {
    recommendations.push('Implement rate limiting');
    recommendations.push('Check for credential stuffing attacks');
  }

  return recommendations;
}