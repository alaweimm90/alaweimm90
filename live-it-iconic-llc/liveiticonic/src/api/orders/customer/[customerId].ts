import { getErrorMessage } from '@/types/api';

/**
 * Get Customer Orders API Endpoint
 *
 * Retrieves all orders for a specific customer
 *
 * @route GET /api/orders/customer/:customerId
 * @param {Request} req - Request with customer ID in URL path
 * @returns {Promise<Response>} Response with { orders, count } or error
 *
 * @example
 * GET /api/orders/customer/cust_123
 *
 * Response 200:
 * {
 *   "orders": [
 *     {
 *       "id": "ORD-1234567890",
 *       "orderNumber": "LII-1234567890",
 *       "total": 118.78,
 *       "status": "pending",
 *       "createdAt": "2024-01-15T10:30:00.000Z"
 *     }
 *   ],
 *   "count": 1
 * }
 *
 * Response 400: { "error": "Customer ID required" }
 * Response 500: { "error": "Failed to get customer orders" }
 */
export default async function handler(req: Request): Promise<Response> {
  if (req.method !== 'GET') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  try {
    const url = new URL(req.url);
    const customerId = url.pathname.split('/').pop();

    if (!customerId) {
      return new Response(JSON.stringify({ error: 'Customer ID required' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // In production, this would fetch customer orders from database
    // For now, return empty array
    const orders:unknown[] = [];

    return new Response(
      JSON.stringify({
        orders,
        count: orders.length,
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  } catch (error: unknown) {
    return new Response(
      JSON.stringify({
        error: getErrorMessage(error) || 'Failed to get customer orders',
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}
