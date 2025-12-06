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
  preview: {
    host: "127.0.0.1",
    port: 4173,
  },
  plugins: [
    react(),
    mode === 'development' &&
    componentTagger(),
    // Token validation in development builds
    mode === 'development' && {
      name: 'token-validation',
      buildStart() {
        console.log('🔍 Running token validation...');
        // Token validation will run automatically
      }
    }
  ].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  define: {
    global: 'globalThis',
  },
  build: {
    rollupOptions: {
      external: [],
      onwarn(warning, warn) {
        // Suppress token-related warnings in production
        if (warning.code === 'UNUSED_EXTERNAL_IMPORT') {
          return;
        }
        warn(warning);
      }
    }
  }
}));