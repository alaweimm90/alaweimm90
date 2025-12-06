/**
 * Type definitions for non-standard browser APIs
 * These APIs are not included in standard TypeScript DOM definitions
 */

// Chrome Performance Memory API
declare global {
  interface Performance {
    memory?: {
      usedJSHeapSize: number;
      totalJSHeapSize: number;
      jsHeapSizeLimit: number;
    };
  }

  // Network Information API
  interface Navigator {
    connection?: {
      effectiveType?: '4g' | '3g' | '2g' | 'slow-2g';
      downlink?: number;
      rtt?: number;
      saveData?: boolean;
      type?: 'bluetooth' | 'cellular' | 'ethernet' | 'wifi' | 'wimax' | 'none' | 'other' | 'unknown';
    };
    deviceMemory?: number; // Device Memory API (in GB)
    standalone?: boolean; // iOS PWA standalone mode
  }

  // Scheduler API (experimental)
  interface Window {
    scheduler?: {
      postTask: (callback: () => void, options?: { priority?: 'user-blocking' | 'user-visible' | 'background' }) => Promise<void>;
    };
    dataLayer?: Array<Record<string, unknown>>; // Google Tag Manager dataLayer
    serviceWorkerManager?: unknown; // Service worker manager global
  }
}

export {};
