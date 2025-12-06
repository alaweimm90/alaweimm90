import { getErrorMessage } from '@/types/api';

/**
 * Send Order Confirmation Email API Endpoint
 *
 * Sends a confirmation email to the customer after successful order placement
 *
 * @route POST /api/email/order-confirmation
 * @param {Request} req - Request with { email, order } in body
 * @returns {Promise<Response>} Response with { success, messageId } or error
 *
 * @example
 * POST /api/email/order-confirmation
 * {
 *   "email": "customer@example.com",
 *   "order": {
 *     "id": "ORD-1234567890",
 *     "orderNumber": "LII-1234567890",
 *     "items": [...],
 *     "total": 118.78
 *   }
 * }
 *
 * Response 200:
 * {
 *   "success": true,
 *   "messageId": "email_1234567890"
 * }
 *
 * Response 500: { "error": "Failed to send email" }
 */
export default async function handler(req: Request): Promise<Response> {
  if (req.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  try {
    const { email, order } = await req.json();

    // In production, this would send email via SendGrid, Mailchimp, or similar
    // For now, simulate successful email send
    console.log('Order confirmation email sent to:', email, 'for order:', order.orderNumber);

    return new Response(
      JSON.stringify({
        success: true,
        messageId: `email_${Date.now()}`,
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  } catch (error: unknown) {
    return new Response(
      JSON.stringify({
        error: getErrorMessage(error) || 'Failed to send email',
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}
