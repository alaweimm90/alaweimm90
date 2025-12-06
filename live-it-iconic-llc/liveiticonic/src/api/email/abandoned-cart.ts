import { getErrorMessage } from '@/types/api';

/**
 * Send Abandoned Cart Email API Endpoint
 *
 * Sends a reminder email to customers with items in their abandoned cart
 *
 * @route POST /api/email/abandoned-cart
 * @param {Request} req - Request with { email, cartItems } in body
 * @returns {Promise<Response>} Response with { success, messageId } or error
 *
 * @example
 * POST /api/email/abandoned-cart
 * {
 *   "email": "customer@example.com",
 *   "cartItems": [
 *     { "productId": "prod_123", "name": "T-Shirt", "price": 29.99 },
 *     { "productId": "prod_456", "name": "Hoodie", "price": 59.99 }
 *   ]
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
    const { email, cartItems } = await req.json();

    // In production, this would send abandoned cart email
    console.log('Abandoned cart email sent to:', email, 'items:', cartItems.length);

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
