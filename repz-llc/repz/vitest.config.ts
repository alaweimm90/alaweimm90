/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['**/*.test.{ts,tsx}'],
    exclude: [
      'node_modules/**',
      'src/test/e2e/**',
      'src/test/integration/**',
      'src/test/performance/**',
      'src/test/accessibility/**',
      'tests/e2e/**',
      '**/*.spec.{ts,tsx}',
      'tests/accessibility/**',
      'tests/performance/**',
      'tests/fixtures/**',
      'playwright.config.ts',
      'src/test/global-setup.ts',
      'src/test/global-teardown.ts',
      'src/__tests__/components/EdgeCaseScenarios.test.tsx'
    ],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/test/',
        '**/*.d.ts',
        '**/*.config.*',
        '**/coverage/**',
        '**/dist/**',
        '**/build/**',
        'tests/e2e/',
        'tests/performance/',
        'tests/accessibility/',
        'tests/fixtures/'
      ],
      thresholds: {
        global: {
          branches: 85,
          functions: 85,
          lines: 85,
          statements: 85
        }
      }
    },
    // Enterprise testing timeouts
    testTimeout: 10000,
    hookTimeout: 10000,
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
})
