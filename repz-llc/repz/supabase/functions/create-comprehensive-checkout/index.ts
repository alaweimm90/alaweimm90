import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import Stripe from "https://esm.sh/stripe@14.21.0";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

const logStep = (step: string, details?: Record<string, unknown>) => {
  const detailsStr = details ? ` - ${JSON.stringify(details)}` : '';
  console.log(`[CREATE-COMPREHENSIVE-CHECKOUT] ${step}${detailsStr}`);
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
    const { tier, billingCycle = 'monthly', returnUrl } = requestData;
    if (!tier) throw new Error("Subscription tier is required");
    logStep("Request data received", { tier, billingCycle, returnUrl });

    // Canonical tier pricing structure - matches constants/tiers.ts
    const getTierPricing = () => {
      const isTestMode = stripeKey?.includes("sk_test_");
      const suffix = isTestMode ? "_TEST" : "_PROD";
      
      // Base pricing for all billing cycles
      const basePricing = {
        core: {
          monthly: 8900, // $89
          quarterly: 24000, // $80/month (10% off)
          semiannual: 45360, // $75.60/month (15% off)
          annual: 85440, // $71.20/month (20% off)
        },
        adaptive: {
          monthly: 14900, // $149
          quarterly: 40230, // $134.10/month (10% off)
          semiannual: 76140, // $126.90/month (15% off)
          annual: 143040, // $119.20/month (20% off)
        },
        performance: {
          monthly: 22900, // $229
          quarterly: 61830, // $206.10/month (10% off)
          semiannual: 116730, // $194.55/month (15% off)
          annual: 219360, // $183.20/month (20% off)
        },
        longevity: {
          monthly: 34900, // $349
          quarterly: 94230, // $314.10/month (10% off)
          semiannual: 178065, // $296.78/month (15% off)
          annual: 334080, // $278.40/month (20% off)
        }
      };

      return {
        core: {
          priceId: Deno.env.get(`STRIPE_PRICE_CORE_${billingCycle.toUpperCase()}${suffix}`),
          name: "Core Program",
          amount: basePricing.core[billingCycle as keyof typeof basePricing.core],
          period: billingCycle
        },
        adaptive: {
          priceId: Deno.env.get(`STRIPE_PRICE_ADAPTIVE_${billingCycle.toUpperCase()}${suffix}`),
          name: "Adaptive Engine",
          amount: basePricing.adaptive[billingCycle as keyof typeof basePricing.adaptive],
          period: billingCycle
        },
        performance: {
          priceId: Deno.env.get(`STRIPE_PRICE_PERFORMANCE_${billingCycle.toUpperCase()}${suffix}`),
          name: "Performance Suite",
          amount: basePricing.performance[billingCycle as keyof typeof basePricing.performance],
          period: billingCycle
        },
        longevity: {
          priceId: Deno.env.get(`STRIPE_PRICE_LONGEVITY_${billingCycle.toUpperCase()}${suffix}`),
          name: "Longevity Concierge",
          amount: basePricing.longevity[billingCycle as keyof typeof basePricing.longevity],
          period: billingCycle
        }
      };
    };

    const tierPricing = getTierPricing();
    const selectedTier = tierPricing[tier as keyof typeof tierPricing];
    if (!selectedTier) throw new Error(`Invalid tier selected: ${tier}`);

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
          tier: tier,
          billing_cycle: billingCycle
        }
      });
      customerId = customer.id;
      logStep("New customer created", { customerId });
    }

    // Fallback: If no Price ID configured, create checkout with manual pricing
    if (!selectedTier.priceId) {
      logStep("No Price ID found, creating manual checkout", { tier, billingCycle, amount: selectedTier.amount });
      
      // Get interval for Stripe
      const interval = billingCycle === 'monthly' ? 'month' : 
                     billingCycle === 'quarterly' ? 'month' : 
                     billingCycle === 'semiannual' ? 'month' : 'year';
      
      const intervalCount = billingCycle === 'quarterly' ? 3 : 
                           billingCycle === 'semiannual' ? 6 : 1;

      const session = await stripe.checkout.sessions.create({
        customer: customerId,
        line_items: [
          {
            price_data: {
              currency: 'usd',
              product_data: {
                name: selectedTier.name,
                description: `${selectedTier.name} - ${billingCycle} billing`,
              },
              unit_amount: selectedTier.amount,
              recurring: {
                interval,
                interval_count: intervalCount,
              },
            },
            quantity: 1,
          },
        ],
        mode: "subscription",
        success_url: returnUrl ? `${returnUrl}?tier=${tier}&payment=success` : `${req.headers.get("origin")}/payment-success?tier=${tier}`,
        cancel_url: returnUrl ? `${returnUrl}?payment=cancelled` : `${req.headers.get("origin")}/?payment=cancelled`,
        metadata: {
          user_id: user.id,
          tier: tier,
          billing_cycle: billingCycle
        }
      });

      logStep("Manual checkout session created", { 
        sessionId: session.id, 
        url: session.url, 
        tier, 
        billingCycle, 
        amount: selectedTier.amount 
      });

      return new Response(JSON.stringify({ 
        url: session.url,
        session_id: session.id,
        tier: tier,
        billing_cycle: billingCycle
      }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
        status: 200,
      });
    }

    // Use pre-configured Price ID
    const session = await stripe.checkout.sessions.create({
      customer: customerId,
      line_items: [
        {
          price: selectedTier.priceId,
          quantity: 1,
        },
      ],
      mode: "subscription",
      success_url: returnUrl ? `${returnUrl}?tier=${tier}&payment=success` : `${req.headers.get("origin")}/payment-success?tier=${tier}`,
      cancel_url: returnUrl ? `${returnUrl}?payment=cancelled` : `${req.headers.get("origin")}/?payment=cancelled`,
      metadata: {
        user_id: user.id,
        tier: tier,
        billing_cycle: billingCycle
      }
    });

    logStep("Checkout session created with Price ID", { 
      sessionId: session.id, 
      url: session.url, 
      tier, 
      billingCycle, 
      priceId: selectedTier.priceId 
    });

    return new Response(JSON.stringify({ 
      url: session.url,
      session_id: session.id,
      tier: tier,
      billing_cycle: billingCycle
    }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 200,
    });

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    logStep("ERROR in create-comprehensive-checkout", { message: errorMessage, stack: error instanceof Error ? error.stack : undefined });
    return new Response(JSON.stringify({ error: errorMessage }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 500,
    });
  }
});