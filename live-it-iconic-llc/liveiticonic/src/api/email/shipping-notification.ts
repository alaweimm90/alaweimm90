import { getErrorMessage } from '@/types/api';

/**
 * Send Shipping Notification Email API Endpoint
 *
 * Sends a shipping notification email with tracking information to the customer
 *
 * @route POST /api/email/shipping-notification
 * @param {Request} req - Request with { email, order, trackingNumber } in body
 * @returns {Promise<Response>} Response with { success, messageId } or error
 *
 * @example
 * POST /api/email/shipping-notification
 * {
 *   "email": "customer@example.com",
 *   "order": {
 *     "id": "ORD-1234567890",
 *     "orderNumber": "LII-1234567890"
 *   },
 *   "trackingNumber": "TRACK123456789"
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
    const { email, order, trackingNumber } = await req.json();

    // In production, this would send email via email service
    console.log('Shipping notification email sent to:', email, 'tracking:', trackingNumber);

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
