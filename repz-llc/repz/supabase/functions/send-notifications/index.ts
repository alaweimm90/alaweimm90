import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { Resend } from "npm:resend@2.0.0";

const resend = new Resend(Deno.env.get("RESEND_API_KEY"));

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

interface NotificationRequest {
  type: 'client_confirmation' | 'admin_notification';
  subscriberData: {
    full_name: string;
    email: string;
    phone: string;
    selected_tier: string;
    tier_price: string;
    goals: string;
    fitness_level?: string;
    preferred_start_date?: string;
    special_requirements?: string;
    stripe_customer_id: string;
    stripe_subscription_id: string;
  };
}

const logStep = (step: string, details?: Record<string, unknown>) => {
  const detailsStr = details ? ` - ${JSON.stringify(details)}` : '';
  console.log(`[SEND-NOTIFICATIONS] ${step}${detailsStr}`);
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    logStep("Function started");

    const { type, subscriberData }: NotificationRequest = await req.json();
    logStep("Request received", { type, email: subscriberData.email });

    if (type === 'client_confirmation') {
      // Send confirmation email to client
      const clientEmail = await resend.emails.send({
        from: "RepzCoach <onboarding@resend.dev>",
        to: [subscriberData.email],
        subject: `Payment Method Secured - ${subscriberData.selected_tier} Plan Design Starting`,
        html: `
          <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
            <h1 style="color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px;">
              ✅ Payment Method Secured Successfully
            </h1>
            
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0;">
              <h2 style="color: #007bff; margin-top: 0;">PLAN DETAILS:</h2>
              <p><strong>Selected Plan:</strong> ${subscriberData.selected_tier} (${subscriberData.tier_price}/month)</p>
              <p><strong>Your Information:</strong> ${subscriberData.full_name}, ${subscriberData.email}, ${subscriberData.phone}</p>
              <p><strong>Status:</strong> Plan design in progress</p>
              ${subscriberData.preferred_start_date ? `<p><strong>Preferred Start:</strong> ${subscriberData.preferred_start_date}</p>` : ''}
            </div>

            <div style="background: #e3f2fd; padding: 20px; border-radius: 8px; margin: 20px 0;">
              <h2 style="color: #1976d2; margin-top: 0;">WHAT HAPPENS NEXT:</h2>
              <ol style="line-height: 1.6;">
                <li><strong>Plan Design (2-3 business days)</strong><br>
                    → Our team will create your personalized program</li>
                <li><strong>Plan Presentation</strong><br>
                    → You'll receive detailed plan via email/call</li>
                <li><strong>Your Approval Required</strong><br>
                    → Review and approve your custom plan</li>
                <li><strong>Billing Activation</strong><br>
                    → Monthly billing begins only after your approval<br>
                    → First charge: ${subscriberData.tier_price}<br>
                    → Billing date: Will be provided after approval</li>
              </ol>
            </div>

            <div style="background: #fff3cd; padding: 15px; border-radius: 8px; margin: 20px 0;">
              <h3 style="color: #856404; margin-top: 0;">IMPORTANT NOTES:</h3>
              <ul style="line-height: 1.6; color: #856404;">
                <li>No charges will occur until you approve your plan</li>
                <li>You can modify requests during the design phase</li>
                <li>Full refund available if you're not satisfied with the plan</li>
                <li>Questions? Reply to this email</li>
              </ul>
            </div>

            <div style="background: #d4edda; padding: 15px; border-radius: 8px; margin: 20px 0;">
              <p style="margin: 0; color: #155724;"><strong>Your Goals:</strong> ${subscriberData.goals}</p>
              ${subscriberData.fitness_level ? `<p style="margin: 10px 0 0 0; color: #155724;"><strong>Fitness Level:</strong> ${subscriberData.fitness_level}</p>` : ''}
              ${subscriberData.special_requirements ? `<p style="margin: 10px 0 0 0; color: #155724;"><strong>Special Requirements:</strong> ${subscriberData.special_requirements}</p>` : ''}
            </div>

            <div style="text-align: center; margin: 30px 0; padding: 20px; background: #f8f9fa; border-radius: 8px;">
              <h3 style="color: #333;">Your plan designer will contact you within 24 hours!</h3>
              <p style="color: #666;">Questions? Reply to this email or contact us directly.</p>
            </div>

            <hr style="margin: 30px 0; border: none; border-top: 1px solid #ddd;">
            
            <div style="font-size: 14px; color: #666;">
              <h4>BILLING DETAILS:</h4>
              <p><strong>Current Status:</strong> $0.00 charged (payment method secured)</p>
              <p><strong>Future Billing:</strong> ${subscriberData.tier_price}/month after plan approval</p>
              <p><strong>Payment Method:</strong> Card ending in **** (secured with Stripe)</p>
            </div>

            <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd;">
              <p style="color: #666; font-size: 14px;">
                Best regards,<br>
                <strong>RepzCoach Team</strong>
              </p>
            </div>
          </div>
        `,
      });

      logStep("Client confirmation email sent", { messageId: clientEmail.data?.id });
    }

    if (type === 'admin_notification') {
      // Send notification to admin
      const adminEmail = await resend.emails.send({
        from: "RepzCoach System <onboarding@resend.dev>",
        to: ["admin@repzcoach.com"], // Replace with actual admin email
        subject: `🔔 New Plan Reservation - ${subscriberData.selected_tier} - ${subscriberData.full_name}`,
        html: `
          <div style="font-family: Arial, sans-serif; max-width: 700px; margin: 0 auto; padding: 20px;">
            <div style="background: #007bff; color: white; padding: 20px; border-radius: 8px 8px 0 0;">
              <h1 style="margin: 0; font-size: 24px;">🎯 New Client Reservation</h1>
              <p style="margin: 10px 0 0 0; font-size: 18px;">${subscriberData.selected_tier} Plan</p>
            </div>

            <div style="background: #f8f9fa; padding: 20px; border: 1px solid #dee2e6;">
              <h2 style="color: #495057; margin-top: 0;">CLIENT DETAILS:</h2>
              <table style="width: 100%; border-collapse: collapse;">
                <tr>
                  <td style="padding: 8px 0; font-weight: bold; width: 150px;">Name:</td>
                  <td style="padding: 8px 0;">${subscriberData.full_name}</td>
                </tr>
                <tr>
                  <td style="padding: 8px 0; font-weight: bold;">Email:</td>
                  <td style="padding: 8px 0;">${subscriberData.email}</td>
                </tr>
                <tr>
                  <td style="padding: 8px 0; font-weight: bold;">Phone:</td>
                  <td style="padding: 8px 0;">${subscriberData.phone}</td>
                </tr>
                <tr>
                  <td style="padding: 8px 0; font-weight: bold;">Tier:</td>
                  <td style="padding: 8px 0;">${subscriberData.selected_tier} - ${subscriberData.tier_price}/month</td>
                </tr>
                ${subscriberData.preferred_start_date ? `
                <tr>
                  <td style="padding: 8px 0; font-weight: bold;">Preferred Start:</td>
                  <td style="padding: 8px 0;">${subscriberData.preferred_start_date}</td>
                </tr>` : ''}
                ${subscriberData.fitness_level ? `
                <tr>
                  <td style="padding: 8px 0; font-weight: bold;">Fitness Level:</td>
                  <td style="padding: 8px 0;">${subscriberData.fitness_level}</td>
                </tr>` : ''}
              </table>
            </div>

            <div style="background: #e3f2fd; padding: 20px; border: 1px solid #dee2e6;">
              <h3 style="color: #1976d2; margin-top: 0;">CLIENT GOALS:</h3>
              <p style="background: white; padding: 15px; border-radius: 4px; margin: 0; font-style: italic;">
                "${subscriberData.goals}"
              </p>
              ${subscriberData.special_requirements ? `
                <h4 style="color: #1976d2; margin: 15px 0 5px 0;">SPECIAL REQUIREMENTS:</h4>
                <p style="background: white; padding: 10px; border-radius: 4px; margin: 0;">
                  ${subscriberData.special_requirements}
                </p>
              ` : ''}
            </div>

            <div style="background: #d1ecf1; padding: 20px; border: 1px solid #bee5eb;">
              <h3 style="color: #0c5460; margin-top: 0;">STRIPE DETAILS:</h3>
              <p><strong>Customer ID:</strong> ${subscriberData.stripe_customer_id}</p>
              <p><strong>Subscription ID:</strong> ${subscriberData.stripe_subscription_id}</p>
              <p><strong>Status:</strong> $0 secured, awaiting plan approval</p>
              
              <p style="margin-top: 15px;">
                <a href="https://dashboard.stripe.com/customers/${subscriberData.stripe_customer_id}" 
                   style="background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px;">
                  View in Stripe Dashboard
                </a>
              </p>
            </div>

            <div style="background: #fff3cd; padding: 20px; border: 1px solid #ffeaa7; border-radius: 0 0 8px 8px;">
              <h3 style="color: #856404; margin-top: 0;">⚡ REQUIRED ACTIONS:</h3>
              <ol style="color: #856404; line-height: 1.6;">
                <li><strong>Contact client within 24 hours</strong></li>
                <li><strong>Design personalized plan (2-3 days)</strong></li>
                <li><strong>Present plan for approval</strong></li>
                <li><strong>Update subscription to real amount after approval</strong></li>
              </ol>
              
              <div style="background: #ffc107; color: #212529; padding: 10px; border-radius: 4px; margin-top: 15px;">
                <strong>⏰ Time Sensitive:</strong> Client expects contact within 24 hours
              </div>
            </div>

            <div style="text-align: center; margin: 20px 0; color: #666; font-size: 14px;">
              <p>This is an automated notification from the RepzCoach reservation system.</p>
              <p>Timestamp: ${new Date().toLocaleString()}</p>
            </div>
          </div>
        `,
      });

      logStep("Admin notification email sent", { messageId: adminEmail.data?.id });
    }

    return new Response(JSON.stringify({ success: true }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 200,
    });

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    logStep("ERROR in send-notifications", { message: errorMessage });
    return new Response(JSON.stringify({ error: errorMessage }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 500,
    });
  }
});