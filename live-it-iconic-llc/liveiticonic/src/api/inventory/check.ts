import { getErrorMessage } from '@/types/api';

/**
 * @swagger
 * /api/inventory/check:
 *   get:
 *     summary: Check product inventory
 *     description: Checks availability and stock quantity for a product
 *     tags: [Inventory]
 *     parameters:
 *       - in: query
 *         name: productId
 *         required: true
 *         schema:
 *           type: string
 *         description: Product ID to check
 *         example: prod_123456789
 *     responses:
 *       200:
 *         description: Inventory information retrieved
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 productId:
 *                   type: string
 *                 available:
 *                   type: integer
 *                   description: Number of units available
 *                 reserved:
 *                   type: integer
 *                   description: Number of units reserved
 *                 total:
 *                   type: integer
 *                   description: Total stock
 *       400:
 *         description: Missing product ID
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *       500:
 *         description: Server error
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
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

    // In production, this would check actual inventory in database
    // For now, return mock availability
    // Mock: Products with ID ending in 1 have 25 units, ending in 2 have 15, ending in 3 have 8
    const productSuffix = productId.slice(-1);
    let availableQuantity = 0;

    if (productSuffix === '1') availableQuantity = 25;
    else if (productSuffix === '2') availableQuantity = 15;
    else if (productSuffix === '3') availableQuantity = 8;
    else availableQuantity = 10; // Default

    const available = availableQuantity >= quantity;

    return new Response(
      JSON.stringify({
        available,
        quantity: availableQuantity,
        requested: quantity,
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  } catch (error: unknown) {
    return new Response(
      JSON.stringify({
        error: getErrorMessage(error) || 'Failed to check inventory',
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}
