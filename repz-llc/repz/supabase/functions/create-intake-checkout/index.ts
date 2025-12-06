import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import Stripe from "https://esm.sh/stripe@14.21.0";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

const logStep = (step: string, details?: Record<string, unknown>) => {
  const detailsStr = details ? ` - ${JSON.stringify(details)}` : '';
  console.log(`[CREATE-INTAKE-CHECKOUT] ${step}${detailsStr}`);
};

// REPZ Canonical Tier Pricing
const TIER_PRICING = {
  core: {
    name: "Core Program",
    amount: 8900, // $89/mo
    description: "Build Your Foundation - Personalized training program with nutrition guidance",
  },
  adaptive: {
    name: "Adaptive Engine",
    amount: 14900, // $149/mo
    description: "Intelligent Optimization - Weekly check-ins, biomarker integration, wearable sync",
  },
  performance: {
    name: "Prime Suite",
    amount: 22900, // $229/mo
    description: "Elite Optimization - AI assistant, form analysis, PEDs protocols",
  },
  longevity: {
    name: "Elite Concierge",
    amount: 34900, // $349/mo
    description: "Ultimate Optimization - In-person training 2x/week, unlimited Q&A, concierge service",
  }
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    logStep("Function started");

    const stripeKey = Deno.env.get("STRIPE_SECRET_KEY");
    if (!stripeKey) throw new Error("STRIPE_SECRET_KEY is not set");
    logStep("Stripe key verified");

    const supabaseClient = createClient(
      Deno.env.get("SUPABASE_URL") ?? "",
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "",
      { auth: { persistSession: false } }
    );

    // Get request body
    const requestData = await req.json();
    const { clientId, paymentType, email, name } = requestData;

    if (!clientId) throw new Error("Client ID is required");
    if (!paymentType) throw new Error("Payment type is required");
    if (!email) throw new Error("Email is required");

    logStep("Request data received", { clientId, paymentType, email, name });

    // Validate tier
    const selectedTier = TIER_PRICING[paymentType as keyof typeof TIER_PRICING];
    if (!selectedTier) throw new Error(`Invalid payment type: ${paymentType}`);

    logStep("Tier selected", { tier: paymentType, amount: selectedTier.amount });

    const stripe = new Stripe(stripeKey, { apiVersion: "2023-10-16" });

    // Check if customer exists by email
    const customers = await stripe.customers.list({ email, limit: 1 });
    let customerId;

    if (customers.data.length > 0) {
      customerId = customers.data[0].id;
      logStep("Existing customer found", { customerId });
    } else {
      // Create new customer
      const customer = await stripe.customers.create({
        email,
        name,
        metadata: {
          client_id: clientId,
          tier: paymentType,
          source: 'intake_form'
        }
      });
      customerId = customer.id;
      logStep("New customer created", { customerId });
    }

    // Create checkout session with one-time payment for initial subscription
    const session = await stripe.checkout.sessions.create({
      customer: customerId,
      line_items: [
        {
          price_data: {
            currency: 'usd',
            product_data: {
              name: `REPZ ${selectedTier.name}`,
              description: selectedTier.description,
              images: ['https://repz.com/lovable-uploads/009ebfb9-7a81-4a04-b2b7-370b51704874.png'],
            },
            unit_amount: selectedTier.amount,
            recurring: {
              interval: 'month',
            },
          },
          quantity: 1,
        },
      ],
      mode: "subscription",
      success_url: `${req.headers.get("origin")}/intake-success?client_id=${clientId}&tier=${paymentType}`,
      cancel_url: `${req.headers.get("origin")}/intake-email?cancelled=true`,
      metadata: {
        client_id: clientId,
        tier: paymentType,
        source: 'intake_form',
        customer_name: name,
        customer_email: email
      },
      payment_method_types: ['card'],
      billing_address_collection: 'required',
      customer_update: {
        address: 'auto',
        name: 'auto',
      },
    });

    logStep("Checkout session created", {
      sessionId: session.id,
      url: session.url,
      tier: paymentType,
      amount: selectedTier.amount
    });

    // Update client record with Stripe session ID
    const { error: updateError } = await supabaseClient
      .from('non_portal_clients')
      .update({
        stripe_session_id: session.id,
        stripe_customer_id: customerId,
        updated_at: new Date().toISOString()
      })
      .eq('id', clientId);

    if (updateError) {
      logStep("Warning: Could not update client record", { error: updateError.message });
    }

    return new Response(JSON.stringify({
      url: session.url,
      session_id: session.id,
      tier: paymentType,
      tier_name: selectedTier.name,
      amount: selectedTier.amount
    }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 200,
    });

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    logStep("ERROR in create-intake-checkout", { message: errorMessage });
    return new Response(JSON.stringify({ error: errorMessage }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 500,
    });
  }
});
