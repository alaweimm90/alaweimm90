import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';
import path from 'path';
import { componentTagger } from 'lovable-tagger';
import { visualizer } from 'rollup-plugin-visualizer';

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: '::',
    port: 8080,
  },
  plugins: [
    react(),
    mode === 'development' && componentTagger(),
    mode === 'production' &&
      visualizer({
        open: false,
        gzipSize: true,
        brotliSize: true,
        filename: 'dist/stats.html',
      }),
  ].filter(Boolean),
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    // Optimize chunk splitting for better caching and LCP
    rollupOptions: {
      output: {
        manualChunks: id => {
          // Vendor chunks for better long-term caching
          if (id.includes('node_modules')) {
            // React and routing (core to app)
            if (id.includes('react') || id.includes('react-dom')) {
              return 'react-core';
            }
            if (id.includes('react-router')) {
              return 'react-router';
            }

            // UI components
            if (id.includes('@radix-ui')) {
              return 'radix-ui';
            }

            // Icons (often large)
            if (id.includes('lucide-react')) {
              return 'icons';
            }

            // Styling utilities
            if (
              id.includes('clsx') ||
              id.includes('tailwind-merge') ||
              id.includes('class-variance-authority')
            ) {
              return 'style-utils';
            }

            // API and data fetching
            if (id.includes('@tanstack/react-query') || id.includes('@supabase')) {
              return 'api-vendor';
            }

            // Form handling
            if (id.includes('@hookform') || id.includes('react-hook-form') || id.includes('zod')) {
              return 'forms';
            }

            // Other vendors
            return 'vendor';
          }

          // Feature chunks - split by route/feature
          if (id.includes('/contexts/CartContext')) {
            return 'cart';
          }
          if (id.includes('/CartIcon') || id.includes('/CartDrawer')) {
            return 'cart-ui';
          }
          if (id.includes('/pages/Checkout')) {
            return 'checkout';
          }
          if (id.includes('/pages/admin')) {
            return 'admin';
          }
          if (id.includes('/pages/Shop') || id.includes('/pages/Collection')) {
            return 'shop';
          }
        },
        // Optimize chunk file names for long-term caching
        chunkFileNames: 'assets/[name]-[hash].js',
        entryFileNames: 'assets/[name]-[hash].js',
        assetFileNames: assetInfo => {
          const info = assetInfo.name?.split('.') ?? [];
          const extType = info[info.length - 1];
          if (/\.(png|jpe?g|svg|gif|tiff|bmp|ico|webp|avif)$/i.test(assetInfo.name ?? '')) {
            return `assets/images/[name]-[hash].${extType}`;
          }
          if (/\.(woff2?|eot|ttf|otf)$/i.test(assetInfo.name ?? '')) {
            return `assets/fonts/[name]-[hash].${extType}`;
          }
          return `assets/[name]-[hash].${extType}`;
        },
      },
      onwarn(warning, warn) {
        // Skip certain warnings
        if (warning.code === 'THIS_IS_UNDEFINED') return;
        if (warning.code === 'CIRCULAR_DEPENDENCY') return;
        // Use default for everything else
        warn(warning);
      },
    },

    // Optimize build for performance
    target: 'es2015', // Support older browsers but with modern features
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true, // Remove console.log in production
        drop_debugger: true,
        pure_funcs: ['console.log', 'console.info'],
      },
      mangle: true,
      format: {
        comments: false, // Remove comments in production
      },
    },
    cssMinify: true,
    cssCodeSplit: true, // Split CSS into separate files

    // Reduce bundle size
    sourcemap: false, // Disable sourcemaps in production
    reportCompressedSize: false,

    // Optimize dependencies
    commonjsOptions: {
      include: [/node_modules/],
    },

    // Chunk size warnings
    chunkSizeWarningLimit: 500, // Warn if chunks exceed 500KB

    // Build performance
    write: true,
    emptyOutDir: true,
  },

  // Optimize development server
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'react-router-dom',
      'lucide-react',
      '@radix-ui/react-dialog',
      '@radix-ui/react-dropdown-menu',
      '@tanstack/react-query',
      '@hookform/resolvers',
      'react-hook-form',
      'clsx',
      'tailwind-merge',
      'class-variance-authority',
    ],
    exclude: ['node_modules/.vite'],
  },

  // Performance
  ssr: false,
}));
