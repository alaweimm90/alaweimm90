import { getErrorMessage } from '@/types/api';

/**
 * @swagger
 * /api/orders/create:
 *   post:
 *     summary: Create a new order
 *     description: Creates a new order with items, shipping address, and payment information
 *     tags: [Orders]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - customerId
 *               - items
 *               - total
 *             properties:
 *               customerId:
 *                 type: string
 *                 example: cust_123456789
 *               items:
 *                 type: array
 *                 items:
 *                   $ref: '#/components/schemas/OrderItem'
 *               subtotal:
 *                 type: number
 *                 example: 99.98
 *               shipping:
 *                 type: number
 *                 example: 10.00
 *               tax:
 *                 type: number
 *                 example: 8.80
 *               total:
 *                 type: number
 *                 example: 118.78
 *               shippingAddress:
 *                 type: object
 *                 properties:
 *                   street: { type: string }
 *                   city: { type: string }
 *                   state: { type: string }
 *                   zip: { type: string }
 *     responses:
 *       200:
 *         description: Order created successfully
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Order'
 *       400:
 *         description: Validation error
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *       500:
 *         description: Server error during order creation
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
    const orderData = await req.json();

    // Generate order number
    const orderNumber = `LII-${Date.now()}`;
    const orderId = `ORD-${Date.now()}`;

    // In production, this would save to database
    // For now, return the created order

    const order = {
      id: orderId,
      orderNumber,
      ...orderData,
      status: 'pending',
      paymentStatus: 'paid',
      createdAt: new Date().toISOString(),
    };

    return new Response(JSON.stringify(order), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error: unknown) {
    return new Response(
      JSON.stringify({
        error: getErrorMessage(error) || 'Failed to create order',
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}
