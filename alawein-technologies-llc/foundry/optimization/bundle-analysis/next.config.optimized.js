/**
 * Optimized Next.js Configuration
 * Performance-focused configuration for all applications
 */

const withBundleAnalyzer = require('@next/bundle-analyzer')({
  enabled: process.env.ANALYZE === 'true',
  openAnalyzer: false
});

const withPWA = require('next-pwa')({
  dest: 'public',
  disable: process.env.NODE_ENV === 'development',
  register: true,
  skipWaiting: true,
  runtimeCaching: [
    {
      urlPattern: /^https?.*/,
      handler: 'NetworkFirst',
      options: {
        cacheName: 'api-cache',
        expiration: {
          maxEntries: 200,
          maxAgeSeconds: 3600 // 1 hour
        },
        networkTimeoutSeconds: 10
      }
    },
    {
      urlPattern: /\.(png|jpg|jpeg|svg|gif|webp|avif)$/,
      handler: 'CacheFirst',
      options: {
        cacheName: 'image-cache',
        expiration: {
          maxEntries: 100,
          maxAgeSeconds: 86400 * 30 // 30 days
        }
      }
    },
    {
      urlPattern: /\.(woff|woff2|ttf|otf|eot)$/,
      handler: 'CacheFirst',
      options: {
        cacheName: 'font-cache',
        expiration: {
          maxEntries: 10,
          maxAgeSeconds: 86400 * 365 // 1 year
        }
      }
    }
  ]
});

/** @type {import('next').NextConfig} */
const nextConfig = {
  // React Strict Mode
  reactStrictMode: true,

  // SWC Minification
  swcMinify: true,

  // Compiler Options
  compiler: {
    // Remove console logs in production
    removeConsole: process.env.NODE_ENV === 'production' ? {
      exclude: ['error', 'warn']
    } : false,

    // React Remove Properties
    reactRemoveProperties: process.env.NODE_ENV === 'production' ? {
      properties: ['^data-test']
    } : false,

    // Emotion (if using emotion)
    emotion: false,

    // Styled Components (if using styled-components)
    styledComponents: false
  },

  // Image Optimization
  images: {
    domains: ['localhost', 'api.crazyideas.com'],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
    formats: ['image/avif', 'image/webp'],
    minimumCacheTTL: 60 * 60 * 24 * 30, // 30 days
    dangerouslyAllowSVG: true,
    contentSecurityPolicy: "default-src 'self'; script-src 'none'; sandbox;",
  },

  // Experimental Features
  experimental: {
    optimizeCss: true,
    scrollRestoration: true,
    optimizePackageImports: [
      '@radix-ui/react-accordion',
      '@radix-ui/react-alert-dialog',
      '@radix-ui/react-avatar',
      '@radix-ui/react-checkbox',
      '@radix-ui/react-dialog',
      '@radix-ui/react-dropdown-menu',
      '@radix-ui/react-label',
      '@radix-ui/react-popover',
      '@radix-ui/react-progress',
      '@radix-ui/react-select',
      '@radix-ui/react-separator',
      '@radix-ui/react-slider',
      '@radix-ui/react-switch',
      '@radix-ui/react-tabs',
      '@radix-ui/react-toast',
      '@radix-ui/react-tooltip',
      'lucide-react',
      'recharts',
      'framer-motion',
      '@tanstack/react-query',
      'date-fns'
    ]
  },

  // Webpack Configuration
  webpack: (config, { isServer, dev }) => {
    // Tree shaking
    if (!dev && !isServer) {
      config.optimization.usedExports = true;
      config.optimization.sideEffects = false;
    }

    // Module Resolution
    config.resolve.alias = {
      ...config.resolve.alias,
      '@': './src',
      'lodash': 'lodash-es'
    };

    // Ignore unnecessary files
    config.module.rules.push({
      test: /\.(test|spec)\.(js|jsx|ts|tsx)$/,
      loader: 'ignore-loader'
    });

    // Source Maps in production
    if (!dev && !isServer) {
      config.devtool = 'hidden-source-map';
    }

    return config;
  },

  // Headers for caching and security
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'X-DNS-Prefetch-Control',
            value: 'on'
          },
          {
            key: 'X-Frame-Options',
            value: 'SAMEORIGIN'
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff'
          },
          {
            key: 'Referrer-Policy',
            value: 'origin-when-cross-origin'
          }
        ]
      },
      {
        source: '/fonts/:path*',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, immutable, max-age=31536000'
          }
        ]
      },
      {
        source: '/_next/static/:path*',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, immutable, max-age=31536000'
          }
        ]
      },
      {
        source: '/images/:path*',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=31536000, s-maxage=31536000'
          }
        ]
      }
    ];
  },

  // Redirects (if needed)
  async redirects() {
    return [];
  },

  // Rewrites for API optimization
  async rewrites() {
    return {
      beforeFiles: [],
      afterFiles: [],
      fallback: []
    };
  },

  // Environment Variables
  env: {
    // Add production optimizations
    NODE_ENV: process.env.NODE_ENV
  },

  // Production Source Maps
  productionBrowserSourceMaps: false,

  // Output Configuration
  output: 'standalone',

  // Trailing Slash
  trailingSlash: false,

  // Powered By Header
  poweredByHeader: false
};

// Export with bundle analyzer and PWA
module.exports = withBundleAnalyzer(withPWA(nextConfig));