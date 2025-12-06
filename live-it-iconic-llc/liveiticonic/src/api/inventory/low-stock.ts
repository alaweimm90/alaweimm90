import { getErrorMessage } from '@/types/api';

/**
 * Get Low Stock Products API Endpoint
 *
 * Retrieves products with inventory below a specified threshold
 *
 * @route GET /api/inventory/low-stock?threshold=5
 * @param {Request} req - Request with optional threshold query parameter (default: 5)
 * @returns {Promise<Response>} Response with { products, threshold } or error
 *
 * @example
 * GET /api/inventory/low-stock?threshold=10
 *
 * Response 200:
 * {
 *   "products": [
 *     { "productId": "prod_3", "quantity": 8 }
 *   ],
 *   "threshold": 10
 * }
 *
 * Response 500: { "error": "Failed to get low stock products" }
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
    const threshold = parseInt(url.searchParams.get('threshold') || '5', 10);

    // In production, this would query database for products with quantity < threshold
    // For now, return mock low stock products
    const lowStockProducts = [
      { productId: '3', quantity: 8 }, // Hoodie
    ].filter(p => p.quantity < threshold);

    return new Response(
      JSON.stringify({
        products: lowStockProducts,
        threshold,
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  } catch (error: unknown) {
    return new Response(
      JSON.stringify({
        error: getErrorMessage(error) || 'Failed to get low stock products',
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}
