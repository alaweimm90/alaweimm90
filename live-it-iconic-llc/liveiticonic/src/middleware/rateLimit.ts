/**
 * Rate limiting store entry
 */
interface RateLimitEntry {
  count: number;
  resetTime: number;
}

/**
 * In-memory store for rate limit tracking
 * In production, consider using Redis or similar persistent store
 */
const rateLimitStore = new Map<string, RateLimitEntry>();

/**
 * Configuration for rate limiting
 */
interface RateLimitConfig {
  limit: number; // Number of requests allowed
  windowMs: number; // Time window in milliseconds
}

const DEFAULT_CONFIG: RateLimitConfig = {
  limit: 5, // 5 requests
  windowMs: 15 * 60 * 1000, // 15 minutes
};

/**
 * Extract client IP from request headers
 * Handles various proxy headers and fallback scenarios
 *
 * @param request - The incoming HTTP request
 * @returns Client IP address or 'unknown'
 */
function getClientIp(request: Request): string {
  const headers = request.headers;
  return (
    (headers.get('x-forwarded-for') as string)?.split(',')[0].trim() ||
    headers.get('x-real-ip') ||
    headers.get('cf-connecting-ip') ||
    'unknown'
  );
}

/**
 * Check if request exceeds rate limit
 * Returns null if within limit, or error Response if limit exceeded
 *
 * @param request - The incoming HTTP request
 * @param config - Rate limit configuration
 * @returns null if within limit, error Response if exceeded
 *
 * @example
 * export async function POST(request: Request) {
 *   const rateLimitResponse = rateLimit(request);
 *   if (rateLimitResponse) {
 *     return rateLimitResponse;
 *   }
 *   // Continue with normal flow
 * }
 */
export function rateLimit(
  request: Request,
  config: Partial<RateLimitConfig> = {}
): Response | null {
  const finalConfig = { ...DEFAULT_CONFIG, ...config };
  const ip = getClientIp(request);
  const now = Date.now();
  const entry = rateLimitStore.get(ip);

  // No entry or window has expired, create new entry
  if (!entry || now > entry.resetTime) {
    rateLimitStore.set(ip, {
      count: 1,
      resetTime: now + finalConfig.windowMs,
    });
    return null;
  }

  // Check if limit exceeded
  if (entry.count >= finalConfig.limit) {
    const retryAfterSeconds = Math.ceil((entry.resetTime - now) / 1000);
    return new Response(
      JSON.stringify({
        error: 'Too many requests',
        message: 'Please try again later.',
        retryAfter: retryAfterSeconds,
      }),
      {
        status: 429,
        headers: {
          'Content-Type': 'application/json',
          'Retry-After': String(retryAfterSeconds),
        },
      }
    );
  }

  // Increment counter and allow request
  entry.count++;
  return null;
}

/**
 * Create a custom rate limiter with specific configuration
 * Useful for different endpoints with different rate limits
 *
 * @param config - Custom rate limit configuration
 * @returns Rate limit middleware function
 *
 * @example
 * const strictLimiter = createRateLimiter({ limit: 3, windowMs: 60000 });
 * export async function POST(request: Request) {
 *   const response = strictLimiter(request);
 *   if (response) return response;
 *   // ...
 * }
 */
export function createRateLimiter(
  config: Partial<RateLimitConfig>
): (request: Request) => Response | null {
  const finalConfig = { ...DEFAULT_CONFIG, ...config };
  return (request: Request) => rateLimit(request, finalConfig);
}

/**
 * Reset rate limit for specific IP
 * Useful for testing or admin operations
 *
 * @param ip - The IP address to reset
 */
export function resetRateLimit(ip: string): void {
  rateLimitStore.delete(ip);
}

/**
 * Reset all rate limits
 * Useful for testing or server restart
 */
export function resetAllRateLimits(): void {
  rateLimitStore.clear();
}

/**
 * Get current rate limit status for an IP
 *
 * @param ip - The IP address to check
 * @returns Rate limit entry or null if not found
 */
export function getRateLimitStatus(
  ip: string
): (RateLimitEntry & { remaining: number }) | null {
  const entry = rateLimitStore.get(ip);
  if (!entry) return null;

  const { limit, windowMs } = DEFAULT_CONFIG;
  const remaining = Math.max(0, limit - entry.count);
  const resetTime = new Date(entry.resetTime);

  return {
    ...entry,
    remaining,
  };
}

/**
 * Cleanup expired entries periodically
 * Prevents memory leaks from accumulated IP entries
 * Runs automatically every minute
 */
const cleanupInterval = setInterval(() => {
  const now = Date.now();
  let cleanedCount = 0;

  for (const [ip, entry] of rateLimitStore.entries()) {
    if (now > entry.resetTime) {
      rateLimitStore.delete(ip);
      cleanedCount++;
    }
  }

  if (cleanedCount > 0) {
    console.debug(`Rate limit cleanup: removed ${cleanedCount} expired entries`);
  }
}, 60000); // Every minute

/**
 * Stop the cleanup interval (useful for testing)
 */
export function stopCleanupInterval(): void {
  clearInterval(cleanupInterval);
}
