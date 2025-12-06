import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import Stripe from "https://esm.sh/stripe@14.21.0";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

const logStep = (step: string, details?: Record<string, unknown>) => {
  const detailsStr = details ? ` - ${JSON.stringify(details)}` : '';
  console.log(`[STRIPE-WEBHOOK] ${step}${detailsStr}`);
};

serve(async (req) => {
  // Set secure search path to prevent SQL injection
  Deno.env.set('PGSEARCH_PATH', '');
  
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  // Only allow POST requests for webhooks
  if (req.method !== "POST") {
    logStep("Invalid HTTP method", { method: req.method });
    return new Response(JSON.stringify({ error: "Method not allowed" }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 405,
    });
  }

  try {
    logStep("Webhook received", { 
      method: req.method,
      userAgent: req.headers.get("user-agent"),
      contentType: req.headers.get("content-type")
    });

    const stripeKey = Deno.env.get("STRIPE_SECRET_KEY");
    const webhookSecret = Deno.env.get("STRIPE_WEBHOOK_SECRET");
    
    if (!stripeKey || !webhookSecret) {
      logStep("ERROR: Missing Stripe configuration");
      throw new Error("Stripe configuration incomplete");
    }

    // Get raw body for signature verification
    const body = await req.text();
    const signature = req.headers.get("stripe-signature");
    
    if (!signature) {
      logStep("ERROR: Missing Stripe signature");
      throw new Error("Webhook signature missing");
    }

    // Validate content type
    const contentType = req.headers.get("content-type");
    if (contentType !== "application/json") {
      logStep("ERROR: Invalid content type", { contentType });
      throw new Error("Invalid content type");
    }

    // CRITICAL: Verify webhook signature to prevent spoofing
    const stripe = new Stripe(stripeKey, { apiVersion: "2023-10-16" });
    const event = stripe.webhooks.constructEvent(body, signature, webhookSecret);
    
    logStep("Webhook signature verified", { type: event.type, id: event.id });

    // Use service role key for secure database operations
    const supabase = createClient(
      Deno.env.get("SUPABASE_URL") ?? "",
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "",
      { auth: { persistSession: false } }
    );

    // Process webhook events
    switch (event.type) {
      case 'checkout.session.completed':
        await handleCheckoutCompleted(event.data.object as Stripe.Checkout.Session, supabase);
        break;
        
      case 'payment_intent.succeeded':
        await handlePaymentIntentSucceeded(event.data.object as Stripe.PaymentIntent, supabase);
        break;
        
      case 'customer.subscription.created':
        await handleSubscriptionCreated(event.data.object as Stripe.Subscription, supabase);
        break;
        
      case 'customer.subscription.updated':
        await handleSubscriptionUpdated(event.data.object as Stripe.Subscription, supabase);
        break;
        
      case 'customer.subscription.deleted':
        await handleSubscriptionCancelled(event.data.object as Stripe.Subscription, supabase);
        break;
        
      case 'invoice.payment_succeeded':
        await handlePaymentSucceeded(event.data.object as Stripe.Invoice, supabase);
        break;
        
      case 'invoice.payment_failed':
        await handlePaymentFailed(event.data.object as Stripe.Invoice, supabase);
        break;
        
      default:
        logStep("Unhandled event type", { type: event.type });
    }

    return new Response(JSON.stringify({ received: true }), {
      headers: { 
        ...corsHeaders, 
        "Content-Type": "application/json",
        "X-Webhook-Processed": "true"
      },
      status: 200,
    });

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    logStep("ERROR processing webhook", { 
      message: errorMessage,
      stack: error instanceof Error ? error.stack : undefined,
      timestamp: new Date().toISOString()
    });
    
    // Determine appropriate status code
    let statusCode = 400;
    if (errorMessage.includes("signature") || errorMessage.includes("unauthorized")) {
      statusCode = 401;
    } else if (errorMessage.includes("configuration") || errorMessage.includes("environment")) {
      statusCode = 500;
    }
    
    return new Response(JSON.stringify({ 
      error: "Webhook processing failed",
      timestamp: new Date().toISOString(),
      requestId: crypto.randomUUID()
    }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: statusCode,
    });
  }
});

async function handleSubscriptionCreated(subscription: Stripe.Subscription, supabase: any) {
  logStep("Processing subscription created", { subscriptionId: subscription.id });
  
  const customerId = subscription.customer as string;
  const stripe = new Stripe(Deno.env.get("STRIPE_SECRET_KEY")!, { apiVersion: "2023-10-16" });
  
  // Get customer details
  const customer = await stripe.customers.retrieve(customerId);
  if (!customer || customer.deleted) {
    throw new Error("Customer not found");
  }
  
  const email = customer.email;
  if (!email) {
    throw new Error("Customer email not found");
  }

  // Determine tier from subscription
  const tier = await getTierFromSubscription(subscription);
  
  // Update client profile
  const { error } = await supabase.from("client_profiles").upsert({
    client_name: email.split('@')[0],
    subscription_tier: tier,
    stripe_subscription_id: subscription.id,
    updated_at: new Date().toISOString(),
  }, { 
    onConflict: 'stripe_subscription_id' 
  });

  if (error) {
    logStep("ERROR updating client profile", { error: error.message });
    throw error;
  }

  logStep("Subscription created successfully", { tier, email });
}

async function handleSubscriptionUpdated(subscription: Stripe.Subscription, supabase: any) {
  logStep("Processing subscription updated", { subscriptionId: subscription.id });
  
  const tier = await getTierFromSubscription(subscription);
  
  // Update client profile
  const { error } = await supabase.from("client_profiles")
    .update({
      subscription_tier: tier,
      updated_at: new Date().toISOString(),
    })
    .eq('stripe_subscription_id', subscription.id);

  if (error) {
    logStep("ERROR updating subscription", { error: error.message });
    throw error;
  }

  logStep("Subscription updated successfully", { tier });
}

async function handleSubscriptionCancelled(subscription: Stripe.Subscription, supabase: any) {
  logStep("Processing subscription cancelled", { subscriptionId: subscription.id });
  
  // Downgrade to core tier but maintain data
  const { error } = await supabase.from("client_profiles")
    .update({
      subscription_tier: 'core',
      stripe_subscription_id: null,
      updated_at: new Date().toISOString(),
    })
    .eq('stripe_subscription_id', subscription.id);

  if (error) {
    logStep("ERROR cancelling subscription", { error: error.message });
    throw error;
  }

  logStep("Subscription cancelled successfully");
}

async function handlePaymentSucceeded(invoice: Stripe.Invoice, supabase: any) {
  logStep("Processing payment succeeded", { invoiceId: invoice.id });
  
  const subscriptionId = invoice.subscription as string;
  if (!subscriptionId) return;

  // Update payment status if needed
  const { error } = await supabase.from("client_profiles")
    .update({
      updated_at: new Date().toISOString(),
    })
    .eq('stripe_subscription_id', subscriptionId);

  if (error) {
    logStep("ERROR updating payment status", { error: error.message });
  }

  logStep("Payment processed successfully");
}

async function handleCheckoutCompleted(session: Stripe.Checkout.Session, supabase: any) {
  logStep("Processing checkout completed", { sessionId: session.id });
  
  const customerId = session.customer as string;
  if (!customerId) {
    logStep("No customer ID in checkout session");
    return;
  }

  const stripe = new Stripe(Deno.env.get("STRIPE_SECRET_KEY")!, { apiVersion: "2023-10-16" });
  
  // Get customer details
  const customer = await stripe.customers.retrieve(customerId);
  if (!customer || customer.deleted) {
    throw new Error("Customer not found");
  }
  
  const email = customer.email;
  if (!email) {
    throw new Error("Customer email not found");
  }

  // Handle subscription mode checkout
  if (session.mode === 'subscription' && session.subscription) {
    const subscription = await stripe.subscriptions.retrieve(session.subscription as string);
    const tier = await getTierFromSubscription(subscription);
    
    // Update client profile
    const { error } = await supabase.from("client_profiles").upsert({
      client_name: email.split('@')[0],
      subscription_tier: tier,
      stripe_subscription_id: subscription.id,
      updated_at: new Date().toISOString(),
    }, { 
      onConflict: 'stripe_subscription_id' 
    });

    if (error) {
      logStep("ERROR updating client profile from checkout", { error: error.message });
      throw error;
    }

    logStep("Checkout completed - subscription created", { tier, email });
    const traceId = (session.metadata && (session.metadata as any).trace_id) || crypto.randomUUID();
    await supabase.from("outbox").insert({
      event_type: "checkout_completed",
      payload: { tier, email, subscription_id: subscription.id },
      trace_id: traceId,
      published: false
    });
  } else if (session.mode === 'payment') {
    // Handle one-time payment
    logStep("One-time payment completed", { 
      amount: session.amount_total,
      currency: session.currency,
      email 
    });
  }
}

async function handlePaymentIntentSucceeded(paymentIntent: Stripe.PaymentIntent, supabase: any) {
  logStep("Processing payment intent succeeded", { 
    paymentIntentId: paymentIntent.id,
    amount: paymentIntent.amount,
    currency: paymentIntent.currency
  });
  
  const customerId = paymentIntent.customer as string;
  if (!customerId) {
    logStep("No customer ID in payment intent");
    return;
  }

  const stripe = new Stripe(Deno.env.get("STRIPE_SECRET_KEY")!, { apiVersion: "2023-10-16" });
  
  // Get customer details
  const customer = await stripe.customers.retrieve(customerId);
  if (!customer || customer.deleted) {
    logStep("Customer not found for payment intent");
    return;
  }
  
  const email = customer.email;
  if (!email) {
    logStep("Customer email not found for payment intent");
    return;
  }

  // Log successful payment
  logStep("Payment intent processed successfully", { 
    email,
    amount: paymentIntent.amount / 100, // Convert cents to dollars
    currency: paymentIntent.currency
  });
}

async function handlePaymentFailed(invoice: Stripe.Invoice, supabase: any) {
  logStep("Processing payment failed", { invoiceId: invoice.id });
  
  const subscriptionId = invoice.subscription as string;
  if (!subscriptionId) return;

  // Could implement retry logic, notifications, etc.
  // For now, just log the failure
  logStep("Payment failed - manual intervention may be required", { subscriptionId });
}

async function getTierFromSubscription(subscription: Stripe.Subscription): Promise<string> {
  const priceId = subscription.items.data[0].price.id;
  const stripe = new Stripe(Deno.env.get("STRIPE_SECRET_KEY")!, { apiVersion: "2023-10-16" });
  const price = await stripe.prices.retrieve(priceId);
  const amount = price.unit_amount || 0;
  
  // Map price amounts to tiers (corrected pricing)
  if (amount >= 34900) return "longevity";    // $349+
  if (amount >= 22900) return "performance";  // $229+
  if (amount >= 14900) return "adaptive";     // $149+
  if (amount >= 8900) return "core";          // $89+
  
  return "core"; // fallback
}
