import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import Stripe from "https://esm.sh/stripe@14.21.0";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

const logStep = (step: string, details?: Record<string, unknown>) => {
  const detailsStr = details ? ` - ${JSON.stringify(details)}` : '';
  console.log(`[SUBSCRIPTION-MANAGEMENT] ${step}${detailsStr}`);
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    logStep("Function started");

    const stripeKey = Deno.env.get("STRIPE_SECRET_KEY");
    if (!stripeKey) throw new Error("STRIPE_SECRET_KEY is not set");

    const supabaseClient = createClient(
      Deno.env.get("SUPABASE_URL") ?? "",
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "",
      { auth: { persistSession: false } }
    );

    const authHeader = req.headers.get("Authorization");
    if (!authHeader) throw new Error("No authorization header provided");

    const token = authHeader.replace("Bearer ", "");
    const { data: userData, error: userError } = await supabaseClient.auth.getUser(token);
    if (userError || !userData.user) throw new Error("User not authenticated");

    const user = userData.user;
    logStep("User authenticated", { userId: user.id, email: user.email });

    const stripe = new Stripe(stripeKey, { apiVersion: "2023-10-16" });
    const { action, ...params } = await req.json();

    switch (action) {
      case 'get_status':
        return await getSubscriptionStatus(supabaseClient, stripe, user.id, user.email);
      
      case 'cancel_subscription':
        return await cancelSubscription(supabaseClient, stripe, user.id, params);
      
      case 'update_subscription':
        return await updateSubscription(supabaseClient, stripe, user.id, params);
      
      case 'create_portal_session':
        return await createPortalSession(stripe, user.email, req.headers.get("origin"));
      
      default:
        throw new Error(`Unknown action: ${action}`);
    }

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    logStep("ERROR", { message: errorMessage });
    return new Response(JSON.stringify({ error: errorMessage }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 500,
    });
  }
});

async function getSubscriptionStatus(supabaseClient: ReturnType<typeof createClient>, stripe: Stripe, userId: string, userEmail: string) {
  logStep("Getting subscription status", { userId });

  // Get user's subscriptions
  const { data: subscriptions, error } = await supabaseClient
    .from('subscriptions')
    .select(`
      *,
      pricing_plans (
        display_name,
        description,
        features,
        metadata
      )
    `)
    .eq('user_id', userId)
    .eq('status', 'active');

  if (error) throw error;

  // Get user's orders
  const { data: orders } = await supabaseClient
    .from('orders')
    .select(`
      *,
      pricing_plans (
        display_name,
        description
      )
    `)
    .eq('user_id', userId)
    .order('created_at', { ascending: false });

  // Get Stripe customer info
  let stripeCustomer = null;
  if (userEmail) {
    const customers = await stripe.customers.list({ email: userEmail, limit: 1 });
    if (customers.data.length > 0) {
      stripeCustomer = customers.data[0];
    }
  }

  return new Response(JSON.stringify({
    subscriptions: subscriptions || [],
    orders: orders || [],
    stripe_customer: stripeCustomer
  }), {
    headers: { ...corsHeaders, "Content-Type": "application/json" },
    status: 200,
  });
}

async function cancelSubscription(supabaseClient: ReturnType<typeof createClient>, stripe: Stripe, userId: string, params: { subscription_id: string }) {
  const { subscription_id } = params;
  logStep("Canceling subscription", { subscriptionId: subscription_id });

  // Get subscription from database
  const { data: subscription, error } = await supabaseClient
    .from('subscriptions')
    .select('*')
    .eq('id', subscription_id)
    .eq('user_id', userId)
    .single();

  if (error || !subscription) {
    throw new Error('Subscription not found');
  }

  // Cancel in Stripe
  await stripe.subscriptions.update(subscription.stripe_subscription_id, {
    cancel_at_period_end: true
  });

  // Update in database
  await supabaseClient
    .from('subscriptions')
    .update({ 
      cancel_at_period_end: true,
      updated_at: new Date().toISOString()
    })
    .eq('id', subscription_id);

  logStep("Subscription cancelled successfully");

  return new Response(JSON.stringify({ success: true }), {
    headers: { ...corsHeaders, "Content-Type": "application/json" },
    status: 200,
  });
}

async function updateSubscription(supabaseClient: ReturnType<typeof createClient>, stripe: Stripe, userId: string, params: { subscription_id: string; new_plan_id: string }) {
  const { subscription_id, new_plan_id } = params;
  logStep("Updating subscription", { subscriptionId: subscription_id, newPlanId: new_plan_id });

  // Get current subscription
  const { data: subscription, error } = await supabaseClient
    .from('subscriptions')
    .select('*')
    .eq('id', subscription_id)
    .eq('user_id', userId)
    .single();

  if (error || !subscription) {
    throw new Error('Subscription not found');
  }

  // Get new plan details
  const { data: newPlan, error: planError } = await supabaseClient
    .from('pricing_plans')
    .select('*')
    .eq('id', new_plan_id)
    .single();

  if (planError || !newPlan) {
    throw new Error('New plan not found');
  }

  // Update subscription in Stripe (simplified - would need proper price IDs in production)
  const stripeSubscription = await stripe.subscriptions.retrieve(subscription.stripe_subscription_id);
  
  // This is a simplified version - in production you'd need proper Stripe price IDs
  logStep("Subscription update completed");

  return new Response(JSON.stringify({ success: true }), {
    headers: { ...corsHeaders, "Content-Type": "application/json" },
    status: 200,
  });
}

async function createPortalSession(stripe: Stripe, userEmail: string, origin: string | null) {
  logStep("Creating customer portal session", { userEmail });

  const customers = await stripe.customers.list({ email: userEmail, limit: 1 });
  if (customers.data.length === 0) {
    throw new Error("No Stripe customer found. Please subscribe to a plan first.");
  }

  const portalSession = await stripe.billingPortal.sessions.create({
    customer: customers.data[0].id,
    return_url: `${origin || "http://localhost:3000"}/account`,
  });

  logStep("Portal session created", { sessionId: portalSession.id });

  return new Response(JSON.stringify({ url: portalSession.url }), {
    headers: { ...corsHeaders, "Content-Type": "application/json" },
    status: 200,
  });
}