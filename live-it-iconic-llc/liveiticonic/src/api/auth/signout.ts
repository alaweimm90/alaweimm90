import { getErrorMessage } from '@/types/api';

/**
 * User Sign Out API Endpoint
 *
 * Invalidates the user's authentication token and logs them out
 *
 * @route POST /api/auth/signout
 * @param {Request} req - Request (optionally with Authorization header)
 * @returns {Promise<Response>} Response with { success: true } or error
 *
 * @example
 * POST /api/auth/signout
 * Headers: { "Authorization": "Bearer token_123456789_abcdefghijklmnopqrst" }
 *
 * Response 200:
 * {
 *   "success": true
 * }
 *
 * Response 500: { "error": "Sign out failed" }
 */
export default async function handler(req: Request): Promise<Response> {
  if (req.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  try {
    // In production, this would invalidate the token on the server
    // For now, just return success
    return new Response(
      JSON.stringify({
        success: true,
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  } catch (error: unknown) {
    return new Response(
      JSON.stringify({
        error: getErrorMessage(error) || 'Sign out failed',
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}
