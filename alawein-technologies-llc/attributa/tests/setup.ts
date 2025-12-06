// Jest test setup file
import { jest } from '@jest/globals';

// Increase timeout for async operations
jest.setTimeout(30000);

// Mock console methods to reduce noise in tests
global.console = {
  ...console,
  error: jest.fn(),
  warn: jest.fn(),
  log: jest.fn(),
};

// Mock browser APIs that may not be available in Node.js
if (typeof window !== 'undefined') {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: jest.fn().mockImplementation(query => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: jest.fn(), // deprecated
      removeListener: jest.fn(), // deprecated
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    })),
  });
} else {
  // Create window mock for Node.js environment
  global.window = {
    matchMedia: jest.fn().mockImplementation(query => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: jest.fn(),
      removeListener: jest.fn(),
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    })),
  } as unknown as Window & typeof globalThis;
}

// Mock ResizeObserver
global.ResizeObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Mock IntersectionObserver
global.IntersectionObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Suppress specific warnings
const originalError = console.error;
console.error = (...args) => {
  if (
    args[0]?.includes?.('DetectGPT analysis failed:') ||
    args[0]?.includes?.('ts-jest')
  ) {
    return;
  }
  originalError.call(console, ...args);
};