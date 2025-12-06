import { getErrorMessage } from '@/types/api';

/**
 * Reserve Inventory API Endpoint
 *
 * Reserves inventory items for a pending order (holds items before payment confirmation)
 *
 * @route POST /api/inventory/reserve
 * @param {Request} req - Request with { productId, quantity, orderId, variantId? } in body
 * @returns {Promise<Response>} Response with { success, productId, quantity, orderId } or error
 *
 * @example
 * POST /api/inventory/reserve
 * {
 *   "productId": "prod_123",
 *   "quantity": 5,
 *   "orderId": "ORD-1234567890",
 *   "variantId": "var_456"
 * }
 *
 * Response 200:
 * {
 *   "success": true,
 *   "productId": "prod_123",
 *   "quantity": 5,
 *   "orderId": "ORD-1234567890"
 * }
 *
 * Response 500: { "error": "Failed to reserve inventory" }
 */
export default async function handler(req: Request): Promise<Response> {
  if (req.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  try {
    const { productId, quantity, orderId, variantId } = await req.json();

    // In production, this would reserve inventory in database
    // For now, simulate successful reservation

    return new Response(
      JSON.stringify({
        success: true,
        productId,
        quantity,
        orderId,
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  } catch (error: unknown) {
    return new Response(
      JSON.stringify({
        error: getErrorMessage(error) || 'Failed to reserve inventory',
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}
