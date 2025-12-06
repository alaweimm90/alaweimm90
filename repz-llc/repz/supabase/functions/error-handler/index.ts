import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

interface ErrorReport {
  error: string;
  context?: string;
  userId?: string;
  url?: string;
  userAgent?: string;
  timestamp: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  metadata?: Record<string, unknown>;
}

const logError = (step: string, details?: Record<string, unknown>) => {
  const detailsStr = details ? ` - ${JSON.stringify(details)}` : '';
  console.log(`[ERROR-HANDLER] ${step}${detailsStr}`);
};

serve(async (req) => {
  // Set secure search path
  Deno.env.set('PGSEARCH_PATH', '');
  
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  if (req.method !== "POST") {
    return new Response(JSON.stringify({ error: "Method not allowed" }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 405,
    });
  }

  try {
    logError("Error report received");

    const supabase = createClient(
      Deno.env.get("SUPABASE_URL") ?? "",
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "",
      { auth: { persistSession: false } }
    );

    const authHeader = req.headers.get("Authorization");
    let userId: string | null = null;

    // Try to get user info if authenticated
    if (authHeader) {
      try {
        const token = authHeader.replace("Bearer ", "");
        const { data } = await supabase.auth.getUser(token);
        userId = data.user?.id || null;
      } catch (e) {
        logError("Failed to get user from token", { error: e.message });
      }
    }

    const errorReport: ErrorReport = await req.json();
    
    // Validate error report
    if (!errorReport.error || typeof errorReport.error !== 'string') {
      throw new Error("Invalid error report format");
    }

    // Sanitize error message (remove sensitive data)
    const sanitizedError = sanitizeErrorMessage(errorReport.error);
    
    // Determine severity based on error type
    const severity = determineSeverity(sanitizedError, errorReport.context);
    
    // Log to console with appropriate level
    const logLevel = severity === 'critical' ? 'error' : 
                    severity === 'high' ? 'warn' : 'info';
    
    console[logLevel]('[CLIENT-ERROR]', {
      error: sanitizedError,
      context: errorReport.context,
      userId,
      url: errorReport.url,
      userAgent: errorReport.userAgent,
      severity,
      timestamp: errorReport.timestamp || new Date().toISOString()
    });

    // Store error in database for analysis (in production, you might want to use a dedicated error tracking service)
    try {
      await supabase.from('error_logs').insert({
        error_message: sanitizedError,
        context: errorReport.context,
        user_id: userId,
        url: errorReport.url,
        user_agent: errorReport.userAgent,
        severity,
        metadata: errorReport.metadata || {},
        created_at: new Date().toISOString()
      });
    } catch (dbError) {
      logError("Failed to store error in database", { error: dbError.message });
      // Don't fail the entire request if database storage fails
    }

    // For critical errors, you might want to send alerts
    if (severity === 'critical') {
      logError("CRITICAL ERROR DETECTED", {
        error: sanitizedError,
        userId,
        context: errorReport.context
      });
      
      // In production, integrate with alerting service (PagerDuty, Slack, etc.)
      // await sendCriticalAlert(errorReport);
    }

    return new Response(JSON.stringify({ 
      received: true, 
      severity,
      timestamp: new Date().toISOString()
    }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 200,
    });

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    logError("ERROR processing error report", { message: errorMessage });
    
    return new Response(JSON.stringify({ 
      error: "Failed to process error report",
      timestamp: new Date().toISOString()
    }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 500,
    });
  }
});

/**
 * Sanitize error messages to remove sensitive information
 */
function sanitizeErrorMessage(error: string): string {
  // Remove potential sensitive data patterns
  return error
    .replace(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g, '[EMAIL]') // Email addresses
    .replace(/\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b/g, '[CARD]') // Credit card numbers
    .replace(/\b\d{3}-\d{2}-\d{4}\b/g, '[SSN]') // SSN
    .replace(/password[=:]\s*[^\s]+/gi, 'password=[REDACTED]') // Passwords
    .replace(/token[=:]\s*[^\s]+/gi, 'token=[REDACTED]') // Tokens
    .replace(/key[=:]\s*[^\s]+/gi, 'key=[REDACTED]') // API keys
    .slice(0, 1000); // Limit length
}

/**
 * Determine error severity based on error content and context
 */
function determineSeverity(error: string, context?: string): 'low' | 'medium' | 'high' | 'critical' {
  const errorLower = error.toLowerCase();
  const contextLower = context?.toLowerCase() || '';

  // Critical errors
  if (errorLower.includes('security') || 
      errorLower.includes('unauthorized') ||
      errorLower.includes('payment failed') ||
      errorLower.includes('database connection') ||
      contextLower.includes('payment') ||
      contextLower.includes('stripe')) {
    return 'critical';
  }

  // High severity errors
  if (errorLower.includes('authentication') ||
      errorLower.includes('permission denied') ||
      errorLower.includes('network error') ||
      errorLower.includes('timeout') ||
      contextLower.includes('auth') ||
      contextLower.includes('checkout')) {
    return 'high';
  }

  // Medium severity errors
  if (errorLower.includes('validation') ||
      errorLower.includes('not found') ||
      errorLower.includes('invalid') ||
      contextLower.includes('form') ||
      contextLower.includes('upload')) {
    return 'medium';
  }

  // Default to low severity
  return 'low';
}