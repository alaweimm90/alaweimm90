import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";
import viteCompression from "vite-plugin-compression";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
  },
  plugins: [
    react(),
    mode === 'development' && componentTagger(),
    // Enable gzip and brotli compression
    mode === 'production' && viteCompression({ 
      algorithm: 'gzip',
      threshold: 1024,
      ext: '.gz'
    }),
    mode === 'production' && viteCompression({ 
      algorithm: 'brotliCompress',
      threshold: 1024,
      ext: '.br'
    }),
  ].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: (id) => {
          // Vendor splitting strategy
          if (id.includes('node_modules')) {
            // React core - critical, keep small
            if (id.includes('react/') || id.includes('react-dom/')) {
              return 'react-vendor';
            }
            // Three.js - lazy loaded
            if (id.includes('three')) {
              return 'three-vendor';
            }
            // Charts - lazy loaded
            if (id.includes('recharts') || id.includes('d3')) {
              return 'charts-vendor';
            }
            // UI components - frequently used
            if (id.includes('@radix-ui')) {
              return 'ui-vendor';
            }
            // Router
            if (id.includes('react-router')) {
              return 'router-vendor';
            }
            // Icons - lazy loaded
            if (id.includes('lucide-react')) {
              return 'icons-vendor';
            }
            // Query/state management
            if (id.includes('@tanstack/react-query')) {
              return 'query-vendor';
            }
            // All other vendor deps
            return 'vendor';
          }
        },
        chunkFileNames: (chunkInfo) => {
          const facadeModuleId = chunkInfo.facadeModuleId ? chunkInfo.facadeModuleId.split('/').pop() : 'chunk';
          return `assets/${facadeModuleId}-[hash].js`;
        }
      }
    },
    // Performance optimizations
    chunkSizeWarningLimit: 600,
    sourcemap: mode === 'development',
    minify: mode === 'production' ? 'terser' : 'esbuild',
    target: 'es2020',
    terserOptions: mode === 'production' ? {
      compress: {
        drop_console: true,
        drop_debugger: true,
        pure_funcs: ['console.log', 'console.info', 'console.debug', 'console.trace'],
      },
      mangle: true,
      format: {
        comments: false,
      },
    } : undefined,
    cssMinify: true,
    cssCodeSplit: true,
    assetsInlineLimit: 4096
  },
  // Performance optimizations
  optimizeDeps: {
    include: ['react', 'react-dom'],
    exclude: ['three', 'lucide-react'] // Let these be dynamically imported
  },
  esbuild: {
    logOverride: { 'this-is-undefined-in-esm': 'silent' }
  }
}));
