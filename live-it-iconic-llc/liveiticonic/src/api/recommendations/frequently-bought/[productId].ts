import { getErrorMessage } from '@/types/api';

/**
 * Get Frequently Bought Together API Endpoint
 *
 * Retrieves products that are frequently purchased together with a given product
 *
 * @route GET /api/recommendations/frequently-bought/:productId
 * @param {Request} req - Request with product ID in URL path
 * @returns {Promise<Response>} Response with { products } or error
 *
 * @example
 * GET /api/recommendations/frequently-bought/prod_123
 *
 * Response 200:
 * {
 *   "products": [
 *     {
 *       "id": "prod_456",
 *       "name": "Complementary Product",
 *       "price": 39.99,
 *       "frequency": 0.85
 *     },
 *     {
 *       "id": "prod_789",
 *       "name": "Bundle Item",
 *       "price": 24.99,
 *       "frequency": 0.72
 *     }
 *   ]
 * }
 *
 * Response 500: { "error": "Failed to get frequently bought products" }
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
    const productId = url.pathname.split('/').pop();

    // In production, this would analyze order data to find frequently bought together
    // For now, return empty array
    const products:unknown[] = [];

    return new Response(
      JSON.stringify({
        products,
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  } catch (error: unknown) {
    return new Response(
      JSON.stringify({
        error: getErrorMessage(error) || 'Failed to get frequently bought products',
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}
