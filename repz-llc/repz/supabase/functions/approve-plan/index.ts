// FIXED: supabase/functions/create-plan-approval/index.ts
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.38.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

const logStep = (step: string, details?: Record<string, unknown>) => {
  const detailsStr = details ? ` - ${JSON.stringify(details)}` : '';
  console.log(`[CREATE-PLAN-APPROVAL] ${step}${detailsStr}`);
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    logStep("Function started");

    const supabaseClient = createClient(
      Deno.env.get("SUPABASE_URL") ?? "",
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "",
      { auth: { persistSession: false } }
    );

    // Get authentication for admin check
    const authHeader = req.headers.get("Authorization");
    if (!authHeader) throw new Error("No authorization header provided");

    const authToken = authHeader.replace("Bearer ", "");
    const { data: userData, error: userError } = await supabaseClient.auth.getUser(authToken);
    if (userError) throw new Error(`Authentication error: ${userError.message}`);
    
    const user = userData.user;
    if (!user?.email) throw new Error("User not authenticated");

    // Check if user is admin
    const { data: adminCheck } = await supabaseClient
      .rpc('is_admin', { check_email: user.email });
    
    if (!adminCheck) {
      throw new Error("Unauthorized: Admin access required");
    }

    logStep("Admin user authenticated", { email: user.email });

    const { subscriberId, planDescription, coachNotes } = await req.json();

    if (!subscriberId || !planDescription) {
      throw new Error("Subscriber ID and plan description are required");
    }

    // Verify subscriber exists and is in correct status
    const { data: subscriber, error: subError } = await supabaseClient
      .from('subscribers')
      .select('*')
      .eq('id', subscriberId)
      .single();

    if (subError || !subscriber) {
      throw new Error('Subscriber not found');
    }

    // Accept multiple valid statuses for plan creation
    if (!['pending', 'payment_setup', 'method_attached'].includes(subscriber.plan_status)) {
      throw new Error(`Invalid subscriber status: ${subscriber.plan_status}. Expected 'pending', 'payment_setup', or 'method_attached'`);
    }

    logStep("Subscriber verified", { 
      subscriberId, 
      status: subscriber.plan_status,
      name: subscriber.full_name 
    });

    // Generate unique approval token
    const approvalToken = crypto.randomUUID();
    const expiresAt = new Date();
    expiresAt.setDate(expiresAt.getDate() + 7); // Expires in 7 days

    // Create plan approval record
    const { data: approval, error: approvalError } = await supabaseClient
      .from('plan_approvals')
      .insert({
        subscriber_id: subscriberId,
        approval_token: approvalToken,
        plan_description: planDescription,
        coach_notes: coachNotes || '',
        status: 'pending',
        expires_at: expiresAt.toISOString(),
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      })
      .select()
      .single();

    if (approvalError) {
      logStep("Failed to create plan approval", { error: approvalError });
      throw new Error(`Failed to create plan approval: ${approvalError.message}`);
    }

    logStep("Plan approval created", { approvalId: approval.id, token: approvalToken });

    // Update subscriber status to 'plan_ready'
    const { error: updateError } = await supabaseClient
      .from('subscribers')
      .update({
        plan_status: 'plan_ready',
        plan_ready_date: new Date().toISOString(),
        updated_at: new Date().toISOString()
      })
      .eq('id', subscriberId);

    if (updateError) {
      logStep("Failed to update subscriber status", { error: updateError });
      throw new Error(`Failed to update subscriber status: ${updateError.message}`);
    }

    // Create coach notification
    await supabaseClient
      .from('coach_notifications')
      .insert({
        subscriber_id: subscriberId,
        notification_type: 'plan_ready',
        title: 'Plan Created and Sent for Approval',
        message: `Plan for ${subscriber.full_name} has been created and sent for client approval.`,
        priority: 'normal',
        is_read: false
      });

    logStep("Subscriber status updated to plan_ready");

    // Email sending - temporarily simplified to avoid template issues
    try {
      const tierPrices = {
        'Core Program': 96,
        'Adaptive Engine': 178,
        'Performance Suite': 298,
        'Longevity Concierge': 396
      };

      const monthlyRate = tierPrices[subscriber.selected_tier as keyof typeof tierPrices] || 97;
      const approvalUrl = `${Deno.env.get('FRONTEND_URL') || 'http://localhost:3000'}/plan-approval/${approvalToken}`;

      // Use existing email template that works
      await supabaseClient.functions.invoke('send-notifications', {
        body: {
          type: 'admin_notification', // Use existing template for now
          subscriberData: {
            full_name: subscriber.full_name,
            email: subscriber.email,
            selected_tier: subscriber.selected_tier,
            tier_price: monthlyRate,
            goals: subscriber.goals,
            fitness_level: subscriber.fitness_level,
            special_requirements: subscriber.special_requirements,
            stripe_customer_id: subscriber.stripe_customer_id,
            // Add approval URL to the data
            approval_url: approvalUrl,
            plan_description: planDescription
          }
        }
      });

      logStep("Notification email sent", { email: subscriber.email });
    } catch (emailError) {
      logStep("Failed to send approval email", { error: emailError });
      // Don't fail the main flow if email fails
      console.warn("Email sending failed but plan approval was created successfully");
    }

    return new Response(JSON.stringify({ 
      success: true, 
      approval_id: approval.id,
      approval_token: approvalToken,
      expires_at: expiresAt.toISOString(),
      approval_url: `${Deno.env.get('FRONTEND_URL') || 'http://localhost:3000'}/plan-approval/${approvalToken}`
    }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 200,
    });

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    logStep("ERROR in create-plan-approval", { message: errorMessage });
    return new Response(JSON.stringify({ error: errorMessage }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 400,
    });
  }
});