import { getErrorMessage } from '@/types/api';

/**
 * @swagger
 * /api/orders/{id}:
 *   get:
 *     summary: Get order details
 *     description: Retrieves detailed information about a specific order by ID
 *     tags: [Orders]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 *         description: Order ID
 *         example: ORD-1234567890
 *     responses:
 *       200:
 *         description: Order retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Order'
 *       400:
 *         description: Missing order ID
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
 *   patch:
 *     summary: Update order status
 *     description: Updates order status and tracking information (admin only)
 *     tags: [Orders]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 *         description: Order ID
 *         example: ORD-1234567890
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               status:
 *                 type: string
 *                 enum: [pending, processing, shipped, delivered, cancelled]
 *                 example: shipped
 *               trackingNumber:
 *                 type: string
 *                 example: TRACK123456789
 *     responses:
 *       200:
 *         description: Order updated successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 id:
 *                   type: string
 *                 status:
 *                   type: string
 *                 trackingNumber:
 *                   type: string
 *                 updatedAt:
 *                   type: string
 *                   format: date-time
 *       400:
 *         description: Missing order ID
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
  const url = new URL(req.url);
  const orderId = url.pathname.split('/').pop();

  if (!orderId) {
    return new Response(JSON.stringify({ error: 'Order ID required' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  if (req.method === 'GET') {
    try {
      // In production, this would fetch from database
      // For now, return mock order
      const order = {
        id: orderId,
        orderNumber: `LII-${orderId.slice(-10)}`,
        items: [],
        subtotal: 0,
        shipping: 0,
        tax: 0,
        total: 0,
        shippingAddress: {},
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
          error: getErrorMessage(error) || 'Failed to get order',
        }),
        {
          status: 500,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    }
  }

  if (req.method === 'PATCH') {
    try {
      const { status, trackingNumber } = await req.json();

      // In production, this would update order in database
      const updatedOrder = {
        id: orderId,
        status,
        trackingNumber,
        updatedAt: new Date().toISOString(),
      };

      return new Response(JSON.stringify(updatedOrder), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    } catch (error: unknown) {
      return new Response(
        JSON.stringify({
          error: getErrorMessage(error) || 'Failed to update order',
        }),
        {
          status: 500,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    }
  }

  return new Response(JSON.stringify({ error: 'Method not allowed' }), {
    status: 405,
    headers: { 'Content-Type': 'application/json' },
  });
}
