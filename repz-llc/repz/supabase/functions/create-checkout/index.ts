import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import Stripe from "https://esm.sh/stripe@14.21.0";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

const logStep = (step: string, details?: Record<string, unknown>) => {
  const detailsStr = details ? ` - ${JSON.stringify(details)}` : '';
  console.log(`[CREATE-CHECKOUT] ${step}${detailsStr}`);
};

serve(async (req) => {
  // Set secure search path
  Deno.env.set('PGSEARCH_PATH', '');
  
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    logStep("Function started");

    const stripeKey = Deno.env.get("STRIPE_SECRET_KEY");
    if (!stripeKey) throw new Error("STRIPE_SECRET_KEY is not set");
    logStep("Stripe key verified");

    // Use service role key for secure operations
    const supabaseClient = createClient(
      Deno.env.get("SUPABASE_URL") ?? "",
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "",
      { auth: { persistSession: false } }
    );

    const authHeader = req.headers.get("Authorization");
    if (!authHeader) throw new Error("No authorization header provided");
    logStep("Authorization header found");

    const token = authHeader.replace("Bearer ", "");
    const { data: userData, error: userError } = await supabaseClient.auth.getUser(token);
    if (userError) throw new Error(`Authentication error: ${userError.message}`);
    const user = userData.user;
    if (!user?.email) throw new Error("User not authenticated or email not available");
    logStep("User authenticated", { userId: user.id, email: user.email });

    // Get request body
    const requestData = await req.json();
    const { tier, billing_period = 'monthly', trace_id } = requestData;
    if (!tier) throw new Error("Subscription tier is required");
    const traceId = trace_id || crypto.randomUUID();
    logStep("Request data received", { tier, billing_period, traceId });

    // Get tier pricing from environment-based Price IDs - supports monthly/annual billing
    const getTierPricing = () => {
      const isTestMode = stripeKey?.includes("sk_test_");
      const suffix = isTestMode ? "_TEST" : "_PROD";
      const period = billing_period === 'annual' ? '_ANNUAL' : '_MONTHLY';
      
      return {
        core: { 
          priceId: Deno.env.get(`STRIPE_PRICE_CORE${period}${suffix}`),
          name: "Core Program",
          amount: billing_period === 'annual' ? 7120 : 8900, // Annual: $71.20/mo (20% off), Monthly: $89/mo
          period: billing_period
        },
        adaptive: { 
          priceId: Deno.env.get(`STRIPE_PRICE_ADAPTIVE${period}${suffix}`),
          name: "Adaptive Engine",
          amount: billing_period === 'annual' ? 11920 : 14900, // Annual: $119.20/mo (20% off), Monthly: $149/mo
          period: billing_period
        },
        performance: { 
          priceId: Deno.env.get(`STRIPE_PRICE_PERFORMANCE${period}${suffix}`),
          name: "Performance Suite", 
          amount: billing_period === 'annual' ? 18320 : 22900, // Annual: $183.20/mo (20% off), Monthly: $229/mo
          period: billing_period
        },
        longevity: { 
          priceId: Deno.env.get(`STRIPE_PRICE_LONGEVITY${period}${suffix}`),
          name: "Longevity Concierge",
          amount: billing_period === 'annual' ? 27920 : 34900, // Annual: $279.20/mo (20% off), Monthly: $349/mo
          period: billing_period
        }
      };
    };

    const tierPricing = getTierPricing();
    const selectedTier = tierPricing[tier as keyof typeof tierPricing];
    if (!selectedTier) throw new Error("Invalid tier selected");

    const stripe = new Stripe(stripeKey, { apiVersion: "2023-10-16" });

    // Check if customer exists
    const customers = await stripe.customers.list({ email: user.email, limit: 1 });
    let customerId;
    
    if (customers.data.length > 0) {
      customerId = customers.data[0].id;
      logStep("Existing customer found", { customerId });
    } else {
      // Create new customer
      const customer = await stripe.customers.create({
        email: user.email,
        metadata: {
          user_id: user.id,
          tier: tier
        }
      });
      customerId = customer.id;
      logStep("New customer created", { customerId });
    }

    // Handle core tier - still paid but lowest tier
    if (tier === 'core') {
      // Validate required Price ID for core tier
      if (!selectedTier.priceId) {
        throw new Error(`Missing Stripe Price ID for core tier. Please configure STRIPE_PRICE_CORE_TEST or STRIPE_PRICE_CORE_PROD in environment.`);
      }
    }

    // Validate required Price ID for all paid tiers
    if (!selectedTier.priceId) {
      throw new Error(`Missing Stripe Price ID for tier: ${tier}. Please configure STRIPE_PRICE_${tier.toUpperCase()}_TEST or STRIPE_PRICE_${tier.toUpperCase()}_PROD in environment.`);
    }

    // Create checkout session for all paid tiers using Stripe Price IDs
    const session = await stripe.checkout.sessions.create({
      customer: customerId,
      line_items: [
        {
          price: selectedTier.priceId,
          quantity: 1,
        },
      ],
      mode: "subscription",
      success_url: `${req.headers.get("origin")}/dashboard?tier=${tier}&payment=success&trace_id=${traceId}&tab=analytics`,
      cancel_url: `${req.headers.get("origin")}/?payment=cancelled`,
      metadata: {
        user_id: user.id,
        tier: tier,
        trace_id: traceId
      }
    }, { idempotencyKey: `checkout:${user.id}:${tier}:${billing_period}:${traceId}` });

    logStep("Checkout session created", { sessionId: session.id, url: session.url, tier, amount: selectedTier.amount, priceId: selectedTier.priceId });

    return new Response(JSON.stringify({ 
      url: session.url,
      session_id: session.id,
      tier: tier 
    }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 200,
    });

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    logStep("ERROR in create-checkout", { message: errorMessage });
    return new Response(JSON.stringify({ error: errorMessage }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 500,
    });
  }
});
