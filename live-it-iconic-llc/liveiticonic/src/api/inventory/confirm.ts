import { getErrorMessage } from '@/types/api';

/**
 * Confirm Inventory Reservation API Endpoint
 *
 * Confirms and finalizes an inventory reservation after successful payment
 *
 * @route POST /api/inventory/confirm
 * @param {Request} req - Request with { productId, quantity, variantId? } in body
 * @returns {Promise<Response>} Response with { success, productId, quantity } or error
 *
 * @example
 * POST /api/inventory/confirm
 * { "productId": "prod_123", "quantity": 5, "variantId": "var_456" }
 *
 * Response 200:
 * {
 *   "success": true,
 *   "productId": "prod_123",
 *   "quantity": 5
 * }
 *
 * Response 500: { "error": "Failed to confirm inventory" }
 */
export default async function handler(req: Request): Promise<Response> {
  if (req.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  try {
    const { productId, quantity, variantId } = await req.json();

    // In production, this would decrement inventory and remove reservation
    // For now, simulate successful confirmation

    return new Response(
      JSON.stringify({
        success: true,
        productId,
        quantity,
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  } catch (error: unknown) {
    return new Response(
      JSON.stringify({
        error: getErrorMessage(error) || 'Failed to confirm inventory',
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}
