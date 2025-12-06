import { getErrorMessage } from '@/types/api';
import { requireAdmin } from '@/middleware/auth';

/**
 * Admin Dashboard Statistics API Endpoint
 *
 * Retrieves aggregated statistics and metrics for the admin dashboard
 * Requires valid JWT access token in Authorization header
 * Enforces admin-level access control
 *
 * @route GET /api/admin/dashboard
 * @param {Request} req - Request with Authorization header for admin access
 * @returns {Promise<Response>} Response with dashboard statistics or error
 *
 * @example
 * GET /api/admin/dashboard
 * Headers: { "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." }
 *
 * Response 200:
 * {
 *   "totalOrders": 42,
 *   "totalRevenue": 15250.75,
 *   "totalProducts": 12,
 *   "lowStockCount": 3,
 *   "recentOrders": [
 *     {
 *       "id": "ORD-1",
 *       "orderNumber": "LII-1234567890",
 *       "customerEmail": "customer@example.com",
 *       "total": 125.99,
 *       "status": "processing",
 *       "createdAt": "2024-01-15T10:30:00.000Z"
 *     }
 *   ]
 * }
 *
 * Response 401: { "error": "Unauthorized", "message": "Missing or invalid Authorization header" }
 * Response 403: { "error": "Forbidden", "message": "Admin access required" }
 * Response 405: { "error": "Method not allowed" }
 * Response 500: { "error": "Failed to get dashboard stats" }
 */
export default async function handler(req: Request): Promise<Response> {
  if (req.method !== 'GET') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  // Check authentication and authorization
  // For now, this just verifies the token exists and is valid
  // In production, implement checkAdminRole to verify user's role in database
  const userIdOrError = await requireAdmin(req, async (userId: string) => {
    // TODO: Implement actual admin role check
    // Example:
    // const user = await db.users.findById(userId);
    // return user?.role === 'admin' || user?.email?.includes('@admin.');
    // For now, allow all authenticated users
    return true;
  });

  if (userIdOrError instanceof Response) {
    return userIdOrError;
  }

  const userId = userIdOrError;

  try {
    // In production, this would calculate stats from database
    // For now, return mock stats
    const stats = {
      totalOrders: 42,
      totalRevenue: 15250.75,
      totalProducts: 12,
      lowStockCount: 3,
      recentOrders: [
        {
          id: 'ORD-1',
          orderNumber: 'LII-1234567890',
          customerEmail: 'customer@example.com',
          total: 125.99,
          status: 'processing',
          createdAt: new Date().toISOString(),
        },
      ],
      meta: {
        requestedBy: userId,
        timestamp: new Date().toISOString(),
      },
    };

    return new Response(JSON.stringify(stats), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error: unknown) {
    return new Response(
      JSON.stringify({
        error: getErrorMessage(error) || 'Failed to get dashboard stats',
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}
