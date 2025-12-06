import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react-swc';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.{test,spec}.{js,ts,jsx,tsx}'],
    exclude: ['node_modules', 'dist', '.idea', '.git', '.cache'],
    testTimeout: 10000,
    mockReset: true,
    restoreMocks: true,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html', 'lcov', 'text-summary'],
      exclude: [
        'node_modules/',
        'src/test/',
        '**/*.d.ts',
        '**/*.config.*',
        '**/mockData',
        '**/*.stories.*',
        '**/*.test.*',
        '**/*.spec.*',
        'dist/',
        'coverage/',
        'src/main.tsx',
        'src/vite-env.d.ts',
        'src/launch-platform/**',
        'src/api/**',
      ],
      all: true,
      lines: 80,
      functions: 80,
      branches: 75,
      statements: 80,
      skipFull: false,
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
