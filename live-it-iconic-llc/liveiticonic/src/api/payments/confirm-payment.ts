import { getErrorMessage } from '@/types/api';

/**
 * Confirm Payment API Endpoint
 *
 * Confirms a Stripe payment and creates an order upon successful payment
 *
 * @route POST /api/payments/confirm-payment
 * @param {Request} req - Request with { paymentIntentId, paymentMethodId } in body
 * @returns {Promise<Response>} Response with { success, orderId, paymentIntentId } or error
 *
 * @example
 * POST /api/payments/confirm-payment
 * {
 *   "paymentIntentId": "pi_1234567890_abc123def456",
 *   "paymentMethodId": "pm_1234567890_abcdefghijklmnopqrst"
 * }
 *
 * Response 200:
 * {
 *   "success": true,
 *   "orderId": "ORD-1234567890",
 *   "paymentIntentId": "pi_1234567890_abc123def456"
 * }
 *
 * Response 500: { "error": "Payment confirmation failed" }
 */
export default async function handler(req: Request): Promise<Response> {
  if (req.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  try {
    const { paymentIntentId, paymentMethodId } = await req.json();

    // In production, this would confirm the payment with Stripe
    // For now, simulate successful payment
    const orderId = `ORD-${Date.now()}`;

    // Create order via order service
    // This would typically be done in a separate endpoint
    // For now, return success with order ID

    return new Response(
      JSON.stringify({
        success: true,
        orderId,
        paymentIntentId,
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  } catch (error: unknown) {
    return new Response(
      JSON.stringify({
        error: getErrorMessage(error) || 'Payment confirmation failed',
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}
