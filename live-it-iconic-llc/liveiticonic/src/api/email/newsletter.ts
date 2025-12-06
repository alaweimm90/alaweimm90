import { getErrorMessage } from '@/types/api';

/**
 * Newsletter Subscription API Endpoint
 *
 * Subscribes a user to the newsletter mailing list
 *
 * @route POST /api/email/newsletter
 * @param {Request} req - Request with { email } in body
 * @returns {Promise<Response>} Response with { success, message } or error
 *
 * @example
 * POST /api/email/newsletter
 * { "email": "subscriber@example.com" }
 *
 * Response 200:
 * {
 *   "success": true,
 *   "message": "Successfully subscribed to newsletter"
 * }
 *
 * Response 400: { "error": "Email required" }
 * Response 500: { "error": "Failed to subscribe" }
 */
export default async function handler(req: Request): Promise<Response> {
  if (req.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  try {
    const { email } = await req.json();

    if (!email) {
      return new Response(JSON.stringify({ error: 'Email required' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // In production, this would add to newsletter list (ConvertKit, Mailchimp, etc.)
    console.log('Newsletter subscription:', email);

    return new Response(
      JSON.stringify({
        success: true,
        message: 'Successfully subscribed to newsletter',
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  } catch (error: unknown) {
    return new Response(
      JSON.stringify({
        error: getErrorMessage(error) || 'Failed to subscribe',
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}
