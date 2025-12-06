import Stripe from 'stripe';
import { STRIPE_CONFIG } from '@/lib/stripe';

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY || '', {
  apiVersion: STRIPE_CONFIG.API_VERSION,
});

const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET || '';

if (!webhookSecret) {
  console.warn('STRIPE_WEBHOOK_SECRET environment variable is not set');
}

/**
 * Stripe Webhook Handler
 *
 * Handles various Stripe webhook events:
 * - payment_intent.succeeded: Payment completed
 * - payment_intent.payment_failed: Payment failed
 * - charge.refunded: Refund processed
 * - customer.subscription.updated: Subscription changes
 *
 * @swagger
 * /api/webhooks/stripe:
 *   post:
 *     summary: Stripe webhook endpoint
 *     description: Receives and processes Stripe webhook events
 *     tags: [Webhooks]
 *     parameters:
 *       - in: header
 *         name: stripe-signature
 *         required: true
 *         schema:
 *           type: string
 *         description: Stripe signature for webhook verification
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *     responses:
 *       200:
 *         description: Webhook received and processed
 *       400:
 *         description: Invalid webhook signature or format
 */
export default async function handler(req: Request): Promise<Response> {
  if (req.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  const signature = req.headers.get('stripe-signature');

  if (!signature) {
    return new Response(JSON.stringify({ error: 'Missing stripe-signature header' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  try {
    const body = await req.text();
    const event = stripe.webhooks.constructEvent(body, signature, webhookSecret);

    // Log webhook received
    console.log(`Webhook event received: ${event.type}`);

    // Handle different event types
    switch (event.type) {
      case 'payment_intent.succeeded':
        await handlePaymentSuccess(event.data.object as Stripe.PaymentIntent);
        break;

      case 'payment_intent.payment_failed':
        await handlePaymentFailure(event.data.object as Stripe.PaymentIntent);
        break;

      case 'charge.refunded':
        await handleRefund(event.data.object as Stripe.Charge);
        break;

      case 'charge.dispute.created':
        await handleChargeback(event.data.object as Stripe.Dispute);
        break;

      case 'customer.subscription.updated':
        await handleSubscriptionUpdate(event.data.object as Stripe.Subscription);
        break;

      case 'customer.subscription.deleted':
        await handleSubscriptionCancellation(event.data.object as Stripe.Subscription);
        break;

      default:
        console.log(`Unhandled event type: ${event.type}`);
    }

    return new Response(JSON.stringify({ received: true }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    console.error('Webhook error:', error);

    if (error instanceof Stripe.errors.StripeSignatureVerificationError) {
      return new Response(JSON.stringify({ error: 'Invalid signature' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    return new Response(JSON.stringify({ error: 'Webhook processing failed' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}

/**
 * Handle successful payment intent
 * - Update order status to confirmed
 * - Send confirmation email
 * - Update inventory
 * - Trigger fulfillment
 */
async function handlePaymentSuccess(paymentIntent: Stripe.PaymentIntent): Promise<void> {
  try {
    const { metadata, amount, currency } = paymentIntent;

    console.log('Processing successful payment:', {
      paymentIntentId: paymentIntent.id,
      amount,
      currency,
      metadata,
    });

    // TODO: Implement business logic
    // 1. Update order status in database
    // 2. Send order confirmation email
    // 3. Update inventory
    // 4. Trigger fulfillment process
    // 5. Record transaction analytics

    console.log(`Payment ${paymentIntent.id} processed successfully`);
  } catch (error) {
    console.error('Error handling payment success:', error);
    throw error;
  }
}

/**
 * Handle failed payment intent
 * - Log failure details
 * - Send failure notification
 * - Update order status
 */
async function handlePaymentFailure(paymentIntent: Stripe.PaymentIntent): Promise<void> {
  try {
    const { last_payment_error, metadata, amount, id } = paymentIntent;

    console.log('Processing failed payment:', {
      paymentIntentId: id,
      amount,
      error: last_payment_error?.message,
      metadata,
    });

    // TODO: Implement business logic
    // 1. Log payment failure
    // 2. Update order status to failed
    // 3. Send failure notification to customer
    // 4. Alert support team if needed
    // 5. Record failure analytics

    console.log(`Payment ${id} failed: ${last_payment_error?.message}`);
  } catch (error) {
    console.error('Error handling payment failure:', error);
    throw error;
  }
}

/**
 * Handle charge refund
 * - Update order status
 * - Restore inventory
 * - Send refund confirmation email
 */
async function handleRefund(charge: Stripe.Charge): Promise<void> {
  try {
    const { id, amount, refunded, metadata } = charge;

    console.log('Processing refund:', {
      chargeId: id,
      amount: refunded ? amount : 0,
      metadata,
    });

    // TODO: Implement business logic
    // 1. Update order status to refunded
    // 2. Restore inventory quantities
    // 3. Send refund confirmation email
    // 4. Process supplier return if needed
    // 5. Record refund analytics

    console.log(`Charge ${id} refunded`);
  } catch (error) {
    console.error('Error handling refund:', error);
    throw error;
  }
}

/**
 * Handle chargeback/dispute
 * - Log dispute details
 * - Alert support team
 * - Prepare evidence if needed
 */
async function handleChargeback(dispute: Stripe.Dispute): Promise<void> {
  try {
    const { id, amount, reason, status } = dispute;

    console.log('Processing chargeback/dispute:', {
      disputeId: id,
      amount,
      reason,
      status,
    });

    // TODO: Implement business logic
    // 1. Alert support team
    // 2. Log dispute details
    // 3. Prepare response documentation
    // 4. Update order tracking
    // 5. Record dispute analytics

    console.log(`Dispute ${id} created with reason: ${reason}`);
  } catch (error) {
    console.error('Error handling chargeback:', error);
    throw error;
  }
}

/**
 * Handle subscription update
 * - Update subscription status
 * - Handle plan changes
 * - Update billing information
 */
async function handleSubscriptionUpdate(subscription: Stripe.Subscription): Promise<void> {
  try {
    const { id, status, metadata, items } = subscription;

    console.log('Processing subscription update:', {
      subscriptionId: id,
      status,
      metadata,
      itemCount: items.data.length,
    });

    // TODO: Implement business logic
    // 1. Update subscription in database
    // 2. Handle plan changes
    // 3. Adjust invoice amounts if needed
    // 4. Send confirmation if changed
    // 5. Record subscription analytics

    console.log(`Subscription ${id} updated to status: ${status}`);
  } catch (error) {
    console.error('Error handling subscription update:', error);
    throw error;
  }
}

/**
 * Handle subscription cancellation
 * - Update subscription status
 * - Send cancellation email
 * - Handle final billing
 */
async function handleSubscriptionCancellation(subscription: Stripe.Subscription): Promise<void> {
  try {
    const { id, metadata, canceled_at } = subscription;

    console.log('Processing subscription cancellation:', {
      subscriptionId: id,
      canceledAt: canceled_at,
      metadata,
    });

    // TODO: Implement business logic
    // 1. Update subscription status to canceled
    // 2. Send cancellation confirmation email
    // 3. Process final invoice if needed
    // 4. Trigger cleanup/archival
    // 5. Record cancellation analytics

    console.log(`Subscription ${id} cancelled at ${new Date(canceled_at! * 1000).toISOString()}`);
  } catch (error) {
    console.error('Error handling subscription cancellation:', error);
    throw error;
  }
}
