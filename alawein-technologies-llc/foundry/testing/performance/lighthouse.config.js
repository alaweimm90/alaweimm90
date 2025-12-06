module.exports = {
  ci: {
    collect: {
      startServerCommand: 'npm run preview',
      startServerReadyPattern: 'Local:',
      url: ['http://localhost:4173'],
      numberOfRuns: 3,
      settings: {
        preset: 'desktop',
        throttling: {
          cpuSlowdownMultiplier: 1,
        },
        screenEmulation: {
          mobile: false,
          width: 1920,
          height: 1080,
          deviceScaleFactor: 1,
        },
      },
    },
    assert: {
      assertions: {
        // Performance metrics
        'first-contentful-paint': ['error', { maxNumericValue: 1500 }],
        'largest-contentful-paint': ['error', { maxNumericValue: 2500 }],
        'first-meaningful-paint': ['error', { maxNumericValue: 2000 }],
        'speed-index': ['error', { maxNumericValue: 3000 }],
        'total-blocking-time': ['error', { maxNumericValue: 300 }],
        'max-potential-fid': ['error', { maxNumericValue: 100 }],
        'cumulative-layout-shift': ['error', { maxNumericValue: 0.1 }],
        'interactive': ['error', { maxNumericValue: 3500 }],

        // Resource optimization
        'uses-webp-images': 'error',
        'uses-optimized-images': 'error',
        'uses-text-compression': 'error',
        'uses-responsive-images': 'warn',
        'offscreen-images': 'warn',
        'unminified-css': 'error',
        'unminified-javascript': 'error',
        'unused-css-rules': 'warn',
        'unused-javascript': 'warn',
        'modern-image-formats': 'warn',

        // Caching
        'uses-long-cache-ttl': 'warn',
        'efficient-animated-content': 'warn',

        // JavaScript optimization
        'no-unload-listeners': 'error',
        'no-document-write': 'error',
        'uses-passive-listeners': 'warn',

        // Accessibility scores
        'accessibility': ['error', { minScore: 0.9 }],

        // Best practices
        'best-practices': ['error', { minScore: 0.9 }],

        // SEO
        'seo': ['warn', { minScore: 0.9 }],

        // Bundle size checks
        'total-byte-weight': ['error', { maxNumericValue: 500000 }],
        'dom-size': ['warn', { maxNumericValue: 1500 }],

        // Network optimization
        'network-requests': ['warn', { maxNumericValue: 50 }],
        'network-rtt': ['warn', { maxNumericValue: 50 }],
        'network-server-latency': ['warn', { maxNumericValue: 50 }],
      },
    },
    upload: {
      target: 'temporary-public-storage',
    },
  },
};