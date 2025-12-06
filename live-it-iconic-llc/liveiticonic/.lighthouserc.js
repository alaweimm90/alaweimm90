/**
 * Lighthouse CI Configuration
 * Automated performance testing and monitoring
 * Runs on every build to ensure Core Web Vitals targets are met
 *
 * Targets:
 * - Performance: 90+
 * - Accessibility: 95+
 * - Best Practices: 95+
 * - SEO: 100
 *
 * Core Web Vitals:
 * - LCP: ≤2.5s (2500ms)
 * - INP: <200ms
 * - CLS: <0.1
 */

module.exports = {
  ci: {
    collect: {
      // Run Lighthouse against the built application
      startServerCommand: 'npm run build && npm run preview',
      url: ['http://localhost:4173/'],
      numberOfRuns: 3,
      settings: {
        chromeFlags: ['--no-sandbox', '--disable-gpu'],
        // Simulate slow 4G network
        emulatedFormFactor: 'mobile',
        throttle: {
          rttMs: 150,
          downstreamThroughputKbps: 1.6 * 1024,
          upstreamThroughputKbps: 750,
          requestLatencyMs: 150,
          downloadThroughputKbps: 1.6 * 1024,
        },
      },
    },
    assert: {
      // Audits that must pass
      assertions: {
        // Overall scores
        'categories:performance': ['error', { minScore: 0.9 }],
        'categories:accessibility': ['error', { minScore: 0.95 }],
        'categories:best-practices': ['error', { minScore: 0.95 }],
        'categories:seo': ['error', { minScore: 1.0 }],

        // Core Web Vitals - Largest Contentful Paint (LCP)
        'largest-contentful-paint': ['error', { maxNumericValue: 2500 }],

        // Core Web Vitals - Interaction to Next Paint (INP)
        'interaction-to-next-paint': ['error', { maxNumericValue: 200 }],

        // Core Web Vitals - Cumulative Layout Shift (CLS)
        'cumulative-layout-shift': ['error', { maxNumericValue: 0.1 }],

        // Additional performance metrics
        'first-contentful-paint': ['error', { maxNumericValue: 1800 }],
        'first-meaningful-paint': ['error', { maxNumericValue: 2300 }],
        'speed-index': ['error', { maxNumericValue: 3400 }],
        'total-blocking-time': ['error', { maxNumericValue: 300 }],

        // JavaScript metrics
        'unused-javascript': ['warn', { maxLength: 0 }],
        'unused-css': ['warn', { maxLength: 0 }],

        // Network metrics
        'network-requests': ['warn'],
        'network-rtt': ['warn'],
        'network-server-latency': ['warn'],

        // Best practices
        'image-alt-text': ['error'],
        'label': ['error'],
        'color-contrast': ['error'],
      },
    },
    upload: {
      // Optional: Upload results to Lighthouse CI server
      // target: 'lhci',
      // serverBaseUrl: 'https://lhci.example.com',
      // token: process.env.LHCI_TOKEN,
      // githubToken: process.env.GITHUB_TOKEN,
    },
  },
};
