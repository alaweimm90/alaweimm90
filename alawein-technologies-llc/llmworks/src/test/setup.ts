import { beforeAll, afterEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';
import '@testing-library/jest-dom';

// Extend Jest matchers with custom accessibility matchers
import 'jest-axe/extend-expect';

// Clean up after each test
afterEach(() => {
  cleanup();
});

// Mock implementations for browser APIs
beforeAll(() => {
  // Mock IntersectionObserver
  global.IntersectionObserver = vi.fn(() => ({
    observe: vi.fn(),
    unobserve: vi.fn(),
    disconnect: vi.fn(),
    root: null,
    rootMargin: '',
    thresholds: [],
    takeRecords: vi.fn(),
  })) as any;

  // Mock ResizeObserver
  global.ResizeObserver = vi.fn(() => ({
    observe: vi.fn(),
    unobserve: vi.fn(),
    disconnect: vi.fn(),
  }));

  // Mock window.matchMedia
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation(query => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(), // deprecated
      removeListener: vi.fn(), // deprecated
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  });

  // Mock localStorage
  const localStorageMock = (() => {
    let store: Record<string, string> = {};
    return {
      getItem: vi.fn((key: string) => store[key] || null),
      setItem: vi.fn((key: string, value: string) => {
        store[key] = value.toString();
      }),
      removeItem: vi.fn((key: string) => {
        delete store[key];
      }),
      clear: vi.fn(() => {
        store = {};
      }),
      length: Object.keys(store).length,
      key: vi.fn((index: number) => Object.keys(store)[index] || null),
    };
  })();
  Object.defineProperty(window, 'localStorage', { value: localStorageMock });

  // Mock sessionStorage
  const sessionStorageMock = (() => {
    let store: Record<string, string> = {};
    return {
      getItem: vi.fn((key: string) => store[key] || null),
      setItem: vi.fn((key: string, value: string) => {
        store[key] = value.toString();
      }),
      removeItem: vi.fn((key: string) => {
        delete store[key];
      }),
      clear: vi.fn(() => {
        store = {};
      }),
      length: Object.keys(store).length,
      key: vi.fn((index: number) => Object.keys(store)[index] || null),
    };
  })();
  Object.defineProperty(window, 'sessionStorage', { value: sessionStorageMock });

  // Mock Web APIs for accessibility testing
  Object.defineProperty(window, 'speechSynthesis', {
    writable: true,
    value: {
      speak: vi.fn(),
      cancel: vi.fn(),
      pause: vi.fn(),
      resume: vi.fn(),
      getVoices: vi.fn(() => []),
    },
  });

  // Mock fetch for API calls
  global.fetch = vi.fn();

  // Mock console methods to reduce noise in tests
  const originalError = console.error;
  beforeAll(() => {
    console.error = vi.fn((...args) => {
      // Only suppress React warnings, show actual errors
      if (typeof args[0] === 'string' && args[0].includes('Warning:')) {
        return;
      }
      originalError(...args);
    });
  });

  // Mock performance APIs for performance testing
  Object.defineProperty(window, 'performance', {
    value: {
      mark: vi.fn(),
      measure: vi.fn(),
      getEntriesByName: vi.fn(() => []),
      getEntriesByType: vi.fn(() => []),
      now: vi.fn(() => Date.now()),
      navigation: {
        type: 0,
      },
      timing: {
        navigationStart: Date.now(),
        loadEventEnd: Date.now() + 1000,
      },
    },
  });

  // Mock requestAnimationFrame
  global.requestAnimationFrame = vi.fn(cb => setTimeout(cb, 16)) as any;
  global.cancelAnimationFrame = vi.fn(id => clearTimeout(id));

  // Mock HTMLElement methods for accessibility
  HTMLElement.prototype.focus = vi.fn();
  HTMLElement.prototype.blur = vi.fn();
  HTMLElement.prototype.click = vi.fn();

  // Mock scrollIntoView
  HTMLElement.prototype.scrollIntoView = vi.fn();

  // Mock clipboard API
  Object.defineProperty(navigator, 'clipboard', {
    value: {
      writeText: vi.fn(() => Promise.resolve()),
      readText: vi.fn(() => Promise.resolve('')),
    },
  });

  // Mock getUserMedia for any media-related tests
  Object.defineProperty(navigator, 'mediaDevices', {
    value: {
      getUserMedia: vi.fn(() => Promise.resolve({
        getTracks: () => [],
      })),
    },
  });
});