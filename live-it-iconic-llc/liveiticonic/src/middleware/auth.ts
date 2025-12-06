import { verifyToken } from '@/lib/jwt';

/**
 * Extended Request interface with optional userId
 */
export interface AuthenticatedRequest extends Request {
  userId?: string;
}

/**
 * Require authentication middleware
 * Extracts and validates JWT token from Authorization header
 *
 * @param request - The incoming HTTP request
 * @returns User ID if authenticated, or error Response if not
 *
 * @example
 * const userIdOrError = await requireAuth(request);
 * if (userIdOrError instanceof Response) {
 *   return userIdOrError;
 * }
 * const userId = userIdOrError;
 */
export async function requireAuth(
  request: Request
): Promise<string | Response> {
  const authHeader = request.headers.get('Authorization');

  if (!authHeader?.startsWith('Bearer ')) {
    return new Response(
      JSON.stringify({
        error: 'Unauthorized',
        message: 'Missing or invalid Authorization header',
      }),
      {
        status: 401,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }

  const token = authHeader.slice(7); // Remove 'Bearer ' prefix
  const payload = verifyToken(token);

  if (!payload || payload.type !== 'access') {
    return new Response(
      JSON.stringify({
        error: 'Invalid or expired token',
        message: 'Please sign in again',
      }),
      {
        status: 401,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }

  return payload.userId;
}

/**
 * Require admin authentication middleware
 * Validates that user is authenticated and has admin privileges
 *
 * @param request - The incoming HTTP request
 * @param checkAdminRole - Optional function to check if user is admin
 * @returns User ID if authenticated and is admin, or error Response
 *
 * @example
 * const userIdOrError = await requireAdmin(request, async (userId) => {
 *   const user = await db.users.findById(userId);
 *   return user?.role === 'admin';
 * });
 * if (userIdOrError instanceof Response) {
 *   return userIdOrError;
 * }
 */
export async function requireAdmin(
  request: Request,
  checkAdminRole?: (userId: string) => Promise<boolean>
): Promise<string | Response> {
  const userIdOrError = await requireAuth(request);

  if (userIdOrError instanceof Response) {
    return userIdOrError;
  }

  // If admin role check function is provided, use it
  if (checkAdminRole) {
    const isAdmin = await checkAdminRole(userIdOrError);
    if (!isAdmin) {
      return new Response(
        JSON.stringify({
          error: 'Forbidden',
          message: 'Admin access required',
        }),
        {
          status: 403,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    }
  }

  return userIdOrError;
}

/**
 * Optional authentication middleware
 * Extracts userId if token is present and valid, but doesn't require it
 *
 * @param request - The incoming HTTP request
 * @returns User ID if authenticated, undefined if not authenticated
 *
 * @example
 * const userId = await optionalAuth(request);
 * if (userId) {
 *   // User is authenticated
 * }
 */
export async function optionalAuth(request: Request): Promise<string | undefined> {
  const authHeader = request.headers.get('Authorization');

  if (!authHeader?.startsWith('Bearer ')) {
    return undefined;
  }

  const token = authHeader.slice(7);
  const payload = verifyToken(token);

  if (!payload || payload.type !== 'access') {
    return undefined;
  }

  return payload.userId;
}
