import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

interface AuthRequest {
  action: 'signin' | 'signup' | 'check-user' | 'verify-otp' | 'reset-password';
  email?: string;
  password?: string;
  phone?: string;
  otp?: string;
  fullName?: string;
  termsAccepted?: boolean;
  liabilityAccepted?: boolean;
  medicalClearance?: boolean;
}

const handler = async (req: Request): Promise<Response> => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  if (req.method !== "POST") {
    return new Response(
      JSON.stringify({ error: "Method not allowed" }),
      { status: 405, headers: { "Content-Type": "application/json", ...corsHeaders } }
    );
  }

  try {
    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseServiceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    
    const supabase = createClient(supabaseUrl, supabaseServiceKey);
    
    const { action, email, password, phone, otp, fullName, termsAccepted, liabilityAccepted, medicalClearance }: AuthRequest = await req.json();

    // Get client IP for rate limiting
    const clientIP = req.headers.get("x-forwarded-for") || req.headers.get("x-real-ip") || "unknown";

    switch (action) {
      case 'check-user': {
        if (!email && !phone) {
          return new Response(
            JSON.stringify({ error: "Email or phone required" }),
            { status: 400, headers: { "Content-Type": "application/json", ...corsHeaders } }
          );
        }

        const { data, error } = await supabase.rpc('check_user_exists', {
          check_email: email || null,
          check_phone: phone || null
        });

        if (error) {
          console.error("Error checking user exists:", error);
          return new Response(
            JSON.stringify({ error: "Failed to check user existence" }),
            { status: 500, headers: { "Content-Type": "application/json", ...corsHeaders } }
          );
        }

        const result = data[0];
        const exists = result?.exists_with_email || result?.exists_with_phone || result?.exists_with_google || false;
        const methods = result?.suggested_methods || [];

        return new Response(
          JSON.stringify({
            exists,
            methods,
            suggestedAction: exists ? 'login' : 'signup',
            message: exists 
              ? `Account found. Try signing in with: ${methods.join(', ')}`
              : 'No account found. You can create a new account.'
          }),
          { status: 200, headers: { "Content-Type": "application/json", ...corsHeaders } }
        );
      }

      case 'signin': {
        if (!email || !password) {
          return new Response(
            JSON.stringify({ error: "Email and password required" }),
            { status: 400, headers: { "Content-Type": "application/json", ...corsHeaders } }
          );
        }

        // Check rate limiting
        const { data: canAttempt } = await supabase.rpc('check_rate_limit', {
          check_email: email,
          time_window_minutes: 15,
          max_attempts: 5
        });

        if (!canAttempt) {
          await supabase.rpc('log_auth_attempt', {
            attempt_email: email,
            auth_method: 'email',
            is_success: false,
            error_message: 'Rate limited'
          });

          return new Response(
            JSON.stringify({ 
              error: "Too many failed attempts. Please wait 15 minutes before trying again.",
              type: "RATE_LIMITED",
              retryAfter: 900
            }),
            { status: 429, headers: { "Content-Type": "application/json", ...corsHeaders } }
          );
        }

        // Attempt sign in
        const { data, error } = await supabase.auth.signInWithPassword({
          email: email.trim(),
          password: password
        });

        // Log attempt
        await supabase.rpc('log_auth_attempt', {
          attempt_email: email,
          auth_method: 'email',
          is_success: !error,
          error_message: error?.message || null
        });

        if (error) {
          let errorType = 'INVALID_CREDENTIALS';
          let message = error.message;

          if (error.message.includes('Email not confirmed')) {
            errorType = 'EMAIL_NOT_VERIFIED';
            message = 'Please check your email and click the confirmation link before signing in.';
          } else if (error.message.includes('Invalid login credentials')) {
            errorType = 'INVALID_CREDENTIALS';
            message = 'Invalid email or password. Please check your credentials and try again.';
          }

          return new Response(
            JSON.stringify({ error: message, type: errorType }),
            { status: 401, headers: { "Content-Type": "application/json", ...corsHeaders } }
          );
        }

        return new Response(
          JSON.stringify({ 
            success: true,
            user: data.user,
            session: data.session
          }),
          { status: 200, headers: { "Content-Type": "application/json", ...corsHeaders } }
        );
      }

      case 'signup': {
        if (!email || !password || !fullName) {
          return new Response(
            JSON.stringify({ error: "Email, password, and full name required" }),
            { status: 400, headers: { "Content-Type": "application/json", ...corsHeaders } }
          );
        }

        if (!termsAccepted || !liabilityAccepted || !medicalClearance) {
          return new Response(
            JSON.stringify({ error: "All legal agreements must be accepted" }),
            { status: 400, headers: { "Content-Type": "application/json", ...corsHeaders } }
          );
        }

        // Check rate limiting
        const { data: canAttempt } = await supabase.rpc('check_rate_limit', {
          check_email: email,
          check_phone: phone || null,
          time_window_minutes: 15,
          max_attempts: 5
        });

        if (!canAttempt) {
          return new Response(
            JSON.stringify({ 
              error: "Too many failed attempts. Please wait 15 minutes before trying again.",
              type: "RATE_LIMITED",
              retryAfter: 900
            }),
            { status: 429, headers: { "Content-Type": "application/json", ...corsHeaders } }
          );
        }

        // Check if user already exists
        const { data: existingUser } = await supabase.rpc('check_user_exists', {
          check_email: email,
          check_phone: phone || null
        });

        const userExists = existingUser?.[0];
        if (userExists?.exists_with_email || userExists?.exists_with_phone || userExists?.exists_with_google) {
          await supabase.rpc('log_auth_attempt', {
            attempt_email: email,
            attempt_phone: phone || null,
            auth_method: 'email',
            is_success: false,
            error_message: 'Account already exists'
          });

          return new Response(
            JSON.stringify({ 
              error: "An account with this email or phone already exists. Please sign in instead.",
              type: "ACCOUNT_EXISTS",
              suggestedMethods: userExists.suggested_methods || ['email']
            }),
            { status: 409, headers: { "Content-Type": "application/json", ...corsHeaders } }
          );
        }

        // Attempt sign up
        const { data, error } = await supabase.auth.signUp({
          email: email.trim(),
          password: password,
          options: {
            data: {
              full_name: fullName,
              terms_accepted: termsAccepted,
              liability_waiver_signed: liabilityAccepted,
              medical_clearance: medicalClearance,
              phone: phone
            }
          }
        });

        // Log attempt
        await supabase.rpc('log_auth_attempt', {
          attempt_email: email,
          attempt_phone: phone || null,
          auth_method: 'email',
          is_success: !error,
          error_message: error?.message || null
        });

        if (error) {
          console.error('Signup error:', error);
          return new Response(
            JSON.stringify({ error: 'Failed to create account. Please try again.', type: "SIGNUP_ERROR" }),
            { status: 400, headers: { "Content-Type": "application/json", ...corsHeaders } }
          );
        }

        return new Response(
          JSON.stringify({ 
            success: true,
            user: data.user,
            session: data.session,
            needsVerification: !data.session
          }),
          { status: 200, headers: { "Content-Type": "application/json", ...corsHeaders } }
        );
      }

      case 'verify-otp': {
        if (!phone || !otp) {
          return new Response(
            JSON.stringify({ error: "Phone and OTP required" }),
            { status: 400, headers: { "Content-Type": "application/json", ...corsHeaders } }
          );
        }

        const { data, error } = await supabase.auth.verifyOtp({
          phone: phone.trim(),
          token: otp.trim(),
          type: 'sms'
        });

        // Log attempt
        await supabase.rpc('log_auth_attempt', {
          attempt_phone: phone,
          auth_method: 'phone',
          is_success: !error,
          error_message: error?.message || null
        });

        if (error) {
          return new Response(
            JSON.stringify({ 
              error: "Invalid verification code. Please check the code and try again.",
              type: "INVALID_CREDENTIALS"
            }),
            { status: 401, headers: { "Content-Type": "application/json", ...corsHeaders } }
          );
        }

        return new Response(
          JSON.stringify({ 
            success: true,
            user: data.user,
            session: data.session
          }),
          { status: 200, headers: { "Content-Type": "application/json", ...corsHeaders } }
        );
      }

      case 'reset-password': {
        if (!email) {
          return new Response(
            JSON.stringify({ error: "Email required" }),
            { status: 400, headers: { "Content-Type": "application/json", ...corsHeaders } }
          );
        }

        const { error } = await supabase.auth.resetPasswordForEmail(email.trim());

        if (error) {
          console.error('Password reset error:', error);
          return new Response(
            JSON.stringify({ error: 'Failed to send password reset email. Please try again.', type: "RESET_ERROR" }),
            { status: 400, headers: { "Content-Type": "application/json", ...corsHeaders } }
          );
        }

        return new Response(
          JSON.stringify({ success: true, message: "Password reset email sent" }),
          { status: 200, headers: { "Content-Type": "application/json", ...corsHeaders } }
        );
      }

      default:
        return new Response(
          JSON.stringify({ error: "Invalid action" }),
          { status: 400, headers: { "Content-Type": "application/json", ...corsHeaders } }
        );
    }

  } catch (error) {
    console.error("Auth handler error:", error);
    return new Response(
      JSON.stringify({ error: "Internal server error" }),
      { status: 500, headers: { "Content-Type": "application/json", ...corsHeaders } }
    );
  }
};

serve(handler);