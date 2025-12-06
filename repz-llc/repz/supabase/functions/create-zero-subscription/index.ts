// UPDATED: supabase/functions/create-zero-subscription/index.ts
import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import Stripe from "https://esm.sh/stripe@14.21.0";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

const logStep = (step: string, details?: Record<string, unknown>) => {
  const detailsStr = details ? ` - ${JSON.stringify(details)}` : '';
  console.log(`[CREATE-ZERO-SUBSCRIPTION] ${step}${detailsStr}`);
};

const TIER_MAPPING = {
  'Core Program': {
    productId: 'core-program',
    realPrice: 8900, // $89.00 in cents
    displayPrice: 89
  },
  'Adaptive Engine': {
    productId: 'adaptive-engine', 
    realPrice: 14900, // $149.00 in cents
    displayPrice: 149
  },
  'Performance Suite': {
    productId: 'performance-suite',
    realPrice: 22900, // $229.00 in cents
    displayPrice: 229
  },
  'Longevity Concierge': {
    productId: 'longevity-concierge',
    realPrice: 34900, // $349.00 in cents
    displayPrice: 349
  }
};

serve(async (req) => {
  // Set secure search path
  Deno.env.set('PGSEARCH_PATH', '');
  
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  const supabaseClient = createClient(
    Deno.env.get("SUPABASE_URL") ?? "",
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "",
    { auth: { persistSession: false } }
  );

  try {
    logStep("Function started");

    const formData = await req.json();
    const { 
      fullName, 
      email, 
      phone, 
      dateOfBirth,
      gender,
      location,
      currentWeight,
      height,
      goals, 
      fitnessLevel, 
      specialRequirements, 
      selectedTier,
      paymentMethodId,
      userId // Should be passed from frontend
    } = formData;
    
    logStep("Received form data", { 
      email, 
      selectedTier, 
      fitnessLevel,
      hasPaymentMethod: !!paymentMethodId 
    });

    const stripe = new Stripe(Deno.env.get("STRIPE_SECRET_KEY") || "", { 
      apiVersion: "2023-10-16" 
    });

    // Check if customer exists
    const existingCustomers = await stripe.customers.list({ email, limit: 1 });
    let customer;
    
    if (existingCustomers.data.length > 0) {
      customer = existingCustomers.data[0];
      logStep("Found existing customer", { customerId: customer.id });
    } else {
      // Create new customer
      customer = await stripe.customers.create({
        email,
        name: fullName,
        phone,
        metadata: {
          goals,
          fitness_level: fitnessLevel,
          special_requirements: specialRequirements || '',
          selected_tier: selectedTier,
          form_submitted_at: new Date().toISOString()
        }
      });
      logStep("Created new customer", { customerId: customer.id });
    }

    // Attach payment method to customer if provided
    if (paymentMethodId) {
      await stripe.paymentMethods.attach(paymentMethodId, {
        customer: customer.id,
      });
      
      // Set as default payment method
      await stripe.customers.update(customer.id, {
        invoice_settings: {
          default_payment_method: paymentMethodId,
        },
      });
      logStep("Attached payment method", { paymentMethodId });
    }

    // Get tier mapping
    const tierInfo = TIER_MAPPING[selectedTier as keyof typeof TIER_MAPPING];
    if (!tierInfo) {
      throw new Error(`Invalid tier: ${selectedTier}`);
    }

    // Store in Supabase with NEW SCHEMA
    const subscriberRecord = {
      user_id: userId,
      email,
      full_name: fullName,
      phone,
      date_of_birth: dateOfBirth,
      gender,
      location,
      current_weight: currentWeight,
      height,
      stripe_customer_id: customer.id,
      stripe_payment_method_id: paymentMethodId,
      selected_tier: selectedTier,
      tier_price: tierInfo.displayPrice,
      payment_status: 'method_attached', // NEW SCHEMA
      plan_status: 'pending', // NEW SCHEMA  
      goals,
      fitness_level: fitnessLevel,
      special_requirements: specialRequirements,
      payment_setup_at: new Date().toISOString(), // NEW SCHEMA
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    };

    const { data: insertedRecord, error: insertError } = await supabaseClient
      .from('subscribers')
      .insert(subscriberRecord)
      .select()
      .single();

    if (insertError) {
      logStep("Database insert error", { error: insertError.message });
      throw new Error(`Database error: ${insertError.message}`);
    }

    logStep("Stored subscription in database");

    // Create coach notification using NEW SCHEMA
    await supabaseClient
      .from('coach_notifications')
      .insert({
        subscriber_id: insertedRecord.id,
        notification_type: 'new_submission',
        title: 'New Client Submission',
        message: `${fullName} has completed their intake form and payment setup for ${selectedTier}.`,
        priority: 'normal'
      });

    // Send notification emails
    try {
      logStep("Sending notification emails");
      const notificationData = {
        full_name: fullName,
        email: email,
        phone: phone,
        selected_tier: selectedTier,
        tier_price: tierInfo.displayPrice,
        goals: goals,
        fitness_level: fitnessLevel,
        special_requirements: specialRequirements,
        stripe_customer_id: customer.id,
      };

      // Send client confirmation email
      await supabaseClient.functions.invoke('send-notifications', {
        body: {
          type: 'client_confirmation',
          subscriberData: notificationData
        }
      });

      // Send admin notification email
      await supabaseClient.functions.invoke('send-notifications', {
        body: {
          type: 'admin_notification',
          subscriberData: notificationData
        }
      });

      logStep("Notification emails sent successfully");
    } catch (emailError) {
      logStep("Email notification failed", { error: emailError });
      // Don't fail the main flow if emails fail
      console.warn("Email notifications failed but subscription was created successfully");
    }

    return new Response(JSON.stringify({
      success: true,
      customer_id: customer.id,
      subscriber_id: insertedRecord.id,
      tier_name: selectedTier,
      tier_price: tierInfo.displayPrice,
      status: 'payment_setup_complete'
    }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 200,
    });

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    logStep("ERROR in create-zero-subscription", { message: errorMessage });
    return new Response(JSON.stringify({ error: errorMessage }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 500,
    });
  }
});

// UPDATED: supabase/functions/start-billing/index.ts
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

    const supabaseClient = createClient(
      Deno.env.get("SUPABASE_URL") ?? "",
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "",
      { auth: { persistSession: false } }
    );

    const { subscriberId } = await req.json();
    if (!subscriberId) {
      throw new Error("Subscriber ID is required");
    }

    // Get client data from database using NEW SCHEMA
    const { data: clientData, error: dbError } = await supabaseClient
      .from('subscribers')
      .select('*')
      .eq('id', subscriberId)
      .single();

    if (dbError || !clientData) {
      throw new Error('Client data not found');
    }

    logStep("Client data retrieved", { 
      clientId: clientData.id, 
      tier: clientData.selected_tier,
      stripeCustomerId: clientData.stripe_customer_id,
      planStatus: clientData.plan_status 
    });

    if (clientData.plan_status !== 'approved') {
      throw new Error('Plan must be approved before starting billing');
    }

    const stripe = new Stripe(stripeKey, { apiVersion: "2023-10-16" });

    // Get tier pricing
    const tierPrices = {
      'Core Program': 96,
      'Adaptive Engine': 178,
      'Performance Suite': 298,
      'Longevity Concierge': 396
    };

    const monthlyPrice = tierPrices[clientData.selected_tier as keyof typeof tierPrices] || 298;
    logStep("Pricing determined", { tier: clientData.selected_tier, monthlyPrice });

    // Create subscription with real pricing
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
      default_payment_method: clientData.stripe_payment_method_id,
      metadata: {
        subscriber_id: subscriberId,
        tier: clientData.selected_tier
      }
    });

    logStep("Subscription created", { subscriptionId: subscription.id });

    // Update client status in database using NEW SCHEMA
    const { error: updateError } = await supabaseClient
      .from('subscribers')
      .update({ 
        plan_status: 'active',
        payment_status: 'active',
        stripe_subscription_id: subscription.id,
        billing_started_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      })
      .eq('id', subscriberId);

    if (updateError) {
      logStep("Database update error", { error: updateError });
      // Try to cancel the subscription if database update fails
      await stripe.subscriptions.cancel(subscription.id);
      throw new Error('Failed to update client status');
    }

    // Log payment event using NEW SCHEMA
    await supabaseClient
      .from('payment_events')
      .insert({
        subscriber_id: subscriberId,
        event_type: 'subscription_created',
        stripe_event_id: subscription.id,
        amount_cents: monthlyPrice * 100,
        status: 'active',
        metadata: {
          subscription_id: subscription.id,
          tier: clientData.selected_tier
        }
      });

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

// UPDATED: supabase/functions/check-subscription/index.ts
import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import Stripe from "https://esm.sh/stripe@14.21.0";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

const logStep = (step: string, details?: Record<string, unknown>) => {
  const detailsStr = details ? ` - ${JSON.stringify(details)}` : '';
  console.log(`[CHECK-SUBSCRIPTION] ${step}${detailsStr}`);
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  const supabaseClient = createClient(
    Deno.env.get("SUPABASE_URL") ?? "",
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "",
    { auth: { persistSession: false } }
  );

  try {
    logStep("Function started");

    const stripeKey = Deno.env.get("STRIPE_SECRET_KEY");
    if (!stripeKey) throw new Error("STRIPE_SECRET_KEY is not set");
    logStep("Stripe key verified");

    const authHeader = req.headers.get("Authorization");
    if (!authHeader) throw new Error("No authorization header provided");
    logStep("Authorization header found");

    const token = authHeader.replace("Bearer ", "");
    logStep("Authenticating user with token");
    
    const { data: userData, error: userError } = await supabaseClient.auth.getUser(token);
    if (userError) throw new Error(`Authentication error: ${userError.message}`);
    const user = userData.user;
    if (!user?.email) throw new Error("User not authenticated or email not available");
    logStep("User authenticated", { userId: user.id, email: user.email });

    // Check database first using NEW SCHEMA
    const { data: subscriberData, error: dbError } = await supabaseClient
      .from('subscribers')
      .select('*')
      .eq('user_id', user.id)
      .single();

    if (dbError && dbError.code !== 'PGRST116') { // PGRST116 = no rows found
      throw new Error(`Database error: ${dbError.message}`);
    }

    if (!subscriberData) {
      logStep("No subscriber record found");
      return new Response(JSON.stringify({ 
        subscribed: false,
        plan_status: 'none',
        payment_status: 'none'
      }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
        status: 200,
      });
    }

    const stripe = new Stripe(stripeKey, { apiVersion: "2023-10-16" });
    
    // Sync with Stripe if we have a customer ID
    if (subscriberData.stripe_customer_id) {
      const subscriptions = await stripe.subscriptions.list({
        customer: subscriberData.stripe_customer_id,
        status: "active",
        limit: 1,
      });
      
      const hasActiveSub = subscriptions.data.length > 0;
      let subscriptionEnd = null;

      if (hasActiveSub) {
        const subscription = subscriptions.data[0];
        subscriptionEnd = new Date(subscription.current_period_end * 1000).toISOString();
        logStep("Active subscription found in Stripe", { subscriptionId: subscription.id });
        
        // Update database if status doesn't match
        if (subscriberData.payment_status !== 'active') {
          await supabaseClient
            .from('subscribers')
            .update({ 
              payment_status: 'active',
              plan_status: 'active',
              updated_at: new Date().toISOString()
            })
            .eq('id', subscriberData.id);
        }
      } else {
        logStep("No active subscription found in Stripe");
        // Update database if needed
        if (subscriberData.payment_status === 'active') {
          await supabaseClient
            .from('subscribers')
            .update({ 
              payment_status: 'inactive',
              updated_at: new Date().toISOString()
            })
            .eq('id', subscriberData.id);
        }
      }
    }

    // Return current status using NEW SCHEMA
    return new Response(JSON.stringify({
      subscribed: subscriberData.payment_status === 'active',
      plan_status: subscriberData.plan_status,
      payment_status: subscriberData.payment_status,
      selected_tier: subscriberData.selected_tier,
      tier_price: subscriberData.tier_price,
      created_at: subscriberData.created_at
    }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 200,
    });

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    logStep("ERROR in check-subscription", { message: errorMessage });
    return new Response(JSON.stringify({ error: errorMessage }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 500,
    });
  }
});

// NEW: supabase/functions/create-plan-approval/index.ts
import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const supabaseClient = createClient(
      Deno.env.get("SUPABASE_URL") ?? "",
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "",
      { auth: { persistSession: false } }
    );

    const { subscriberId, planDescription, coachNotes } = await req.json();

    // Generate unique approval token
    const token = crypto.randomUUID();
    const expiresAt = new Date();
    expiresAt.setDate(expiresAt.getDate() + 7); // Expires in 7 days

    // Create plan approval record
    const { data: approval, error } = await supabaseClient
      .from('plan_approvals')
      .insert({
        subscriber_id: subscriberId,
        approval_token: token,
        plan_description: planDescription,
        coach_notes: coachNotes || '',
        expires_at: expiresAt.toISOString()
      })
      .select()
      .single();

    if (error) {
      throw new Error('Failed to create plan approval');
    }

    // Update subscriber status
    await supabaseClient
      .from('subscribers')
      .update({
        plan_status: 'plan_ready',
        updated_at: new Date().toISOString()
      })
      .eq('id', subscriberId);

    // Get subscriber details for email
    const { data: subscriber } = await supabaseClient
      .from('subscribers')
      .select('*')
      .eq('id', subscriberId)
      .single();

    if (subscriber) {
      // Send plan approval email
      await supabaseClient.functions.invoke('send-notifications', {
        body: {
          type: 'plan_approval_request',
          subscriberData: {
            ...subscriber,
            approval_token: token,
            plan_description: planDescription,
            coach_notes: coachNotes
          }
        }
      });
    }

    return new Response(JSON.stringify({ 
      success: true, 
      approval_token: token,
      expires_at: expiresAt.toISOString()
    }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 200,
    });

  } catch (error) {
    return new Response(JSON.stringify({ error: error.message }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 400,
    });
  }
});

// NEW: supabase/functions/approve-plan/index.ts  
import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const supabaseClient = createClient(
      Deno.env.get("SUPABASE_URL") ?? "",
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "",
      { auth: { persistSession: false } }
    );

    const { approvalToken, approved } = await req.json();

    // Get plan approval
    const { data: planApproval, error: fetchError } = await supabaseClient
      .from('plan_approvals')
      .select('*')
      .eq('approval_token', approvalToken)
      .eq('status', 'pending')
      .single();

    if (fetchError || !planApproval) {
      throw new Error('Plan approval not found or expired');
    }

    // Check if expired
    if (new Date(planApproval.expires_at) < new Date()) {
      throw new Error('Approval link has expired');
    }

    // Update approval status
    await supabaseClient
      .from('plan_approvals')
      .update({
        status: approved ? 'approved' : 'rejected',
        approved_at: approved ? new Date().toISOString() : null,
        updated_at: new Date().toISOString()
      })
      .eq('id', planApproval.id);

    // Update subscriber status
    await supabaseClient
      .from('subscribers')
      .update({
        plan_status: approved ? 'approved' : 'rejected',
        plan_approved_at: approved ? new Date().toISOString() : null,
        updated_at: new Date().toISOString()
      })
      .eq('id', planApproval.subscriber_id);

    if (approved) {
      // Start billing by calling start-billing function
      await supabaseClient.functions.invoke('start-billing', {
        body: { subscriberId: planApproval.subscriber_id }
      });
    }

    return new Response(JSON.stringify({ success: true }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 200,
    });

  } catch (error) {
    return new Response(JSON.stringify({ error: error.message }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 400,
    });
  }
});