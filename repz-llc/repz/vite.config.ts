/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
  },
  plugins: [
    react(),
    mode === 'development' &&
    componentTagger(),
  ].filter(Boolean),
  
  // Performance optimizations
  build: {
    // Enable code splitting
    rollupOptions: {
      output: {
        manualChunks: {
          // Vendor chunk for stable dependencies
          vendor: ['react', 'react-dom', 'react-router-dom'],
          // UI chunk for component libraries
          ui: ['@radix-ui/react-dialog', '@radix-ui/react-dropdown-menu', '@radix-ui/react-select'],
          // Charts chunk for data visualization
          charts: ['recharts'],
          // Utils chunk for utilities
          utils: ['date-fns', 'clsx', 'tailwind-merge', 'zod'],
          // Supabase chunk for backend services
          supabase: ['@supabase/supabase-js'],
          // Stripe chunk for payments
          stripe: ['@stripe/stripe-js', '@stripe/react-stripe-js']
        }
      }
    },
    // Optimize bundle size
    minify: mode === 'production' ? 'esbuild' : false,
    // Increase chunk size warning limit
    chunkSizeWarningLimit: 1000,
    // Source maps for production debugging
    sourcemap: mode !== 'production'
  },
  
  // Development server optimizations
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'react-router-dom',
      '@tanstack/react-query',
      'lucide-react'
    ],
    exclude: ['@vite/client', '@vite/env']
  },
  
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    include: ['**/*.test.{ts,tsx}'],
    exclude: ['src/test/e2e/**', 'tests/e2e/**', '**/*.spec.{ts,tsx}']
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
