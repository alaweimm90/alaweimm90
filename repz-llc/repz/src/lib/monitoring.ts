/**
 * Monitoring stub - replace with actual implementation when needed
 */

export function trackError(
  error: Error,
  context?: Record<string, unknown>
): void {
  if (import.meta.env.DEV) {
    console.error('[Monitoring] Error:', error.message, context);
  }
  // In production, send to Sentry or similar
}

export function trackMetric(
  name: string,
  value: number,
  tags?: Record<string, string>
): void {
  if (import.meta.env.DEV) {
    console.log('[Monitoring] Metric:', name, value, tags);
  }
}

export function trackPerformance(
  name: string,
  duration: number
): void {
  if (import.meta.env.DEV) {
    console.log('[Monitoring] Performance:', name, `${duration}ms`);
  }
}
