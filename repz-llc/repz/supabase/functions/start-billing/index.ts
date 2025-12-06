import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import Stripe from "https://esm.sh/stripe@14.21.0";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

const logStep = (step: string, details?: Record<string, unknown>) => {
  const detailsStr = details ? ` - ${JSON.stringify(details)}` : '';
  console.log(`[START-BILLING] ${step}${detailsStr}`);
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    logStep("Function started");

    const stripeKey = Deno.env.get("STRIPE_SECRET_KEY");
    if (!stripeKey) throw new Error("STRIPE_SECRET_KEY is not set");

    // Use service role key for secure operations
    const supabaseClient = createClient(
      Deno.env.get("SUPABASE_URL") ?? "",
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "",
      { auth: { persistSession: false } }
    );

    const authHeader = req.headers.get("Authorization");
    if (!authHeader) throw new Error("No authorization header provided");

    const token = authHeader.replace("Bearer ", "");
    const { data: userData, error: userError } = await supabaseClient.auth.getUser(token);
    if (userError) throw new Error(`Authentication error: ${userError.message}`);
    const user = userData.user;
    if (!user?.email) throw new Error("User not authenticated");
    logStep("User authenticated", { userId: user.id, email: user.email });

    // Get client data from database
    const { data: clientData, error: dbError } = await supabaseClient
      .from('subscribers')
      .select('*')
      .eq('user_id', user.id)
      .single();

    if (dbError || !clientData) {
      throw new Error('Client data not found');
    }

    logStep("Client data retrieved", { 
      clientId: clientData.id, 
      tier: clientData.selected_tier,
      stripeCustomerId: clientData.stripe_customer_id 
    });

    const stripe = new Stripe(stripeKey, { apiVersion: "2023-10-16" });

    // Get the customer's payment methods
    const paymentMethods = await stripe.paymentMethods.list({
      customer: clientData.stripe_customer_id,
      type: 'card',
    });

    if (paymentMethods.data.length === 0) {
      throw new Error('No payment method found. Please add a payment method first.');
    }

    const paymentMethod = paymentMethods.data[0];
    logStep("Payment method found", { paymentMethodId: paymentMethod.id });

    // Get tier pricing
    const tierPrices = {
      'core': 96,
      'adaptive': 178,
      'performance': 298,
      'longevity': 396
    };

    const monthlyPrice = tierPrices[clientData.selected_tier as keyof typeof tierPrices] || 298;
    logStep("Pricing determined", { tier: clientData.selected_tier, monthlyPrice });

    // Create subscription
    const subscription = await stripe.subscriptions.create({
      customer: clientData.stripe_customer_id,
      items: [{
        price_data: {
          currency: 'usd',
          product_data: {
            name: `${clientData.selected_tier} Coaching Plan`,
            description: `Monthly coaching subscription for ${clientData.full_name}`,
          },
          unit_amount: monthlyPrice * 100, // Convert to cents
          recurring: {
            interval: 'month',
          },
        },
      }],
      default_payment_method: paymentMethod.id,
      metadata: {
        user_id: user.id,
        client_id: clientData.id,
        tier: clientData.selected_tier
      }
    });

    logStep("Subscription created", { subscriptionId: subscription.id });

    // Update client status in database
    const { error: updateError } = await supabaseClient
      .from('subscribers')
      .update({ 
        plan_status: 'active',
        subscription_status: 'active',
        stripe_subscription_id: subscription.id,
        stripe_payment_method_id: paymentMethod.id,
        billing_start_date: new Date().toISOString()
      })
      .eq('id', clientData.id);

    if (updateError) {
      logStep("Database update error", { error: updateError });
      // Try to cancel the subscription if database update fails
      await stripe.subscriptions.cancel(subscription.id);
      throw new Error('Failed to update client status');
    }

    logStep("Client status updated successfully");

    return new Response(JSON.stringify({
      success: true,
      subscription_id: subscription.id,
      status: subscription.status
    }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 200,
    });

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    logStep("ERROR in start-billing", { message: errorMessage });
    return new Response(JSON.stringify({ error: errorMessage }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 500,
    });
  }
});