/**
 * Analytics stub - replace with actual implementation when needed
 */

export class Analytics {
  static instance: Analytics | null = null;

  static getInstance(): Analytics {
    if (!Analytics.instance) {
      Analytics.instance = new Analytics();
    }
    return Analytics.instance;
  }

  track(event: string, properties?: Record<string, unknown>): void {
    if (import.meta.env.DEV) {
      console.log('[Analytics]', event, properties);
    }
  }

  identify(userId: string, traits?: Record<string, unknown>): void {
    if (import.meta.env.DEV) {
      console.log('[Analytics] Identify:', userId, traits);
    }
  }

  page(name: string, properties?: Record<string, unknown>): void {
    if (import.meta.env.DEV) {
      console.log('[Analytics] Page:', name, properties);
    }
  }
}

export const analytics = Analytics.getInstance();
