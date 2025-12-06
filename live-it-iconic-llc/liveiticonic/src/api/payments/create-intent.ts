import Stripe from 'stripe';
import { getErrorMessage } from '@/types/api';
import { STRIPE_CONFIG } from '@/lib/stripe';

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY || '', {
  apiVersion: STRIPE_CONFIG.API_VERSION,
});

/**
 * @swagger
 * /api/payments/create-intent:
 *   post:
 *     summary: Create payment intent
 *     description: Creates a Stripe payment intent for processing payments. Supports various payment methods including credit cards, digital wallets, and local payment methods.
 *     tags: [Payments]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - amount
 *             properties:
 *               amount:
 *                 type: number
 *                 description: Amount in USD cents (e.g., 9999 for $99.99)
 *                 example: 9999
 *               currency:
 *                 type: string
 *                 default: usd
 *                 example: usd
 *               metadata:
 *                 type: object
 *                 description: Custom metadata to attach to the payment intent
 *                 example:
 *                   orderId: ORD-1234567890
 *                   customerId: CUST-9876543210
 *               description:
 *                 type: string
 *                 description: Description for the payment intent
 *                 example: "Purchase of merchandise"
 *     responses:
 *       200:
 *         description: Payment intent created successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 clientSecret:
 *                   type: string
 *                   description: Client secret for confirming the payment
 *                 paymentIntentId:
 *                   type: string
 *                   description: Stripe payment intent ID
 *                 status:
 *                   type: string
 *                   description: Current status of the payment intent
 *       400:
 *         description: Validation error
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 error:
 *                   type: string
 *       500:
 *         description: Server error
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 error:
 *                   type: string
 */
export default async function handler(req: Request): Promise<Response> {
  if (req.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  try {
    const { amount, currency = 'usd', metadata, description } = await req.json();

    // Validate amount
    if (!amount || typeof amount !== 'number' || amount <= 0) {
      return new Response(
        JSON.stringify({
          error: 'Invalid amount. Must be a positive number representing cents.',
        }),
        {
          status: 400,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    }

    // Create payment intent with Stripe
    const paymentIntent = await stripe.paymentIntents.create({
      amount: Math.round(amount), // Ensure amount is in cents
      currency,
      automatic_payment_methods: {
        enabled: true,
      },
      metadata: {
        application: 'live-it-iconic',
        timestamp: new Date().toISOString(),
        ...metadata,
      },
      description,
    });

    return new Response(
      JSON.stringify({
        clientSecret: paymentIntent.client_secret,
        paymentIntentId: paymentIntent.id,
        status: paymentIntent.status,
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  } catch (error: unknown) {
    console.error('Payment intent creation error:', error);

    const errorMessage =
      error instanceof Stripe.errors.StripeInvalidRequestError
        ? error.message
        : getErrorMessage(error) || 'Failed to create payment intent';

    return new Response(JSON.stringify({ error: errorMessage }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}
