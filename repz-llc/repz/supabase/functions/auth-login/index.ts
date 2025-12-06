import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';
import { z } from 'https://deno.land/x/zod@v3.22.4/mod.ts';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

// Validation schema
const loginSchema = z.object({
  email: z.string()
    .email('Invalid email format')
    .max(254, 'Email must be less than 254 characters')
    .transform(val => val.toLowerCase().trim()),
  password: z.string()
    .min(1, 'Password is required')
    .max(128, 'Password must be less than 128 characters'),
});

Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? ''
    );

    // Parse and validate input
    const body = await req.json();
    const validation = loginSchema.safeParse(body);

    if (!validation.success) {
      const errors = validation.error.errors.map(e => e.message).join(', ');
      console.log('Validation failed:', errors);
      return new Response(
        JSON.stringify({ error: 'Invalid email or password' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    const { email, password } = validation.data;
    console.log('Login attempt for:', email);

    // Sign in with password
    const { data: authData, error: authError } = await supabase.auth.signInWithPassword({
      email,
      password,
    });

    if (authError) {
      console.error('Login error:', authError);
      return new Response(
        JSON.stringify({ error: 'Invalid email or password' }),
        { status: 401, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Update last login
    const { error: updateError } = await supabase
      .from('profiles')
      .update({ last_login: new Date().toISOString() })
      .eq('id', authData.user.id);

    if (updateError) {
      console.error('Failed to update last login:', updateError);
    }

    // Get user profile and role
    const { data: profile } = await supabase
      .from('profiles')
      .select('*, user_roles(role)')
      .eq('id', authData.user.id)
      .single();

    console.log('Login successful for:', email);

    return new Response(
      JSON.stringify({
        success: true,
        user: {
          id: authData.user.id,
          email: authData.user.email,
          tier: profile?.subscription_tier || 'core',
          role: profile?.user_roles?.[0]?.role || 'member',
        },
        session: {
          access_token: authData.session.access_token,
          refresh_token: authData.session.refresh_token,
          expires_in: authData.session.expires_in,
        },
      }),
      { status: 200, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  } catch (error) {
    console.error('Unexpected error:', error);
    // Return generic error message, log details server-side
    return new Response(
      JSON.stringify({ error: 'An unexpected error occurred. Please try again.' }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});
