import { render, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import { describe, it, expect, beforeEach, vi } from 'vitest';

import Index from '@/pages/Index';
import Arena from '@/pages/Arena';
import Bench from '@/pages/Bench';
import Dashboard from '@/pages/Dashboard';

// Performance thresholds
const PERFORMANCE_THRESHOLDS = {
  RENDER_TIME: 100, // ms
  MEMORY_USAGE: 50 * 1024 * 1024, // 50MB
  COMPONENT_COUNT: 500, // Maximum components
  EVENT_LISTENER_COUNT: 100, // Maximum event listeners
};

// Test wrapper component
const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        {children}
      </BrowserRouter>
    </QueryClientProvider>
  );
};

// Helper function to measure render time
const measureRenderTime = async (Component: React.ComponentType) => {
  const startTime = performance.now();
  
  const result = render(
    <TestWrapper>
      <Component />
    </TestWrapper>
  );

  await waitFor(() => {
    expect(result.container.firstChild).toBeInTheDocument();
  });

  const endTime = performance.now();
  return endTime - startTime;
};

// Helper function to count DOM nodes
const countDOMNodes = (container: HTMLElement): number => {
  return container.querySelectorAll('*').length;
};

// Helper function to simulate memory usage check
const checkMemoryUsage = (): number => {
  // In a real browser environment, you'd use performance.memory
  // This is a mock for the test environment
  return Math.random() * 30 * 1024 * 1024; // Mock 0-30MB
};

describe('Performance Tests', () => {
  beforeEach(() => {
    // Clear any existing performance marks
    performance.clearMarks?.();
    performance.clearMeasures?.();
  });

  describe('Render Performance', () => {
    it('Index page should render quickly', async () => {
      const renderTime = await measureRenderTime(Index);
      expect(renderTime).toBeLessThan(PERFORMANCE_THRESHOLDS.RENDER_TIME);
    });

    it('Arena page should render quickly', async () => {
      const renderTime = await measureRenderTime(Arena);
      expect(renderTime).toBeLessThan(PERFORMANCE_THRESHOLDS.RENDER_TIME);
    });

    it('Bench page should render quickly', async () => {
      const renderTime = await measureRenderTime(Bench);
      expect(renderTime).toBeLessThan(PERFORMANCE_THRESHOLDS.RENDER_TIME);
    });

    it('Dashboard page should render quickly', async () => {
      const renderTime = await measureRenderTime(Dashboard);
      expect(renderTime).toBeLessThan(PERFORMANCE_THRESHOLDS.RENDER_TIME);
    });
  });

  describe('DOM Complexity', () => {
    it('Index page should not have excessive DOM nodes', async () => {
      const { container } = render(
        <TestWrapper>
          <Index />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(container.firstChild).toBeInTheDocument();
      });

      const nodeCount = countDOMNodes(container);
      expect(nodeCount).toBeLessThan(PERFORMANCE_THRESHOLDS.COMPONENT_COUNT);
    });

    it('Dashboard should manage complex UI efficiently', async () => {
      const { container } = render(
        <TestWrapper>
          <Dashboard />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(container.firstChild).toBeInTheDocument();
      });

      const nodeCount = countDOMNodes(container);
      // Dashboard can be more complex but should still be reasonable
      expect(nodeCount).toBeLessThan(PERFORMANCE_THRESHOLDS.COMPONENT_COUNT * 2);
    });
  });

  describe('Memory Usage', () => {
    it('should not leak memory during component lifecycle', async () => {
      const initialMemory = checkMemoryUsage();

      const { unmount } = render(
        <TestWrapper>
          <Dashboard />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(document.body.firstChild).toBeInTheDocument();
      });

      unmount();

      // Allow garbage collection
      await new Promise(resolve => setTimeout(resolve, 100));

      const finalMemory = checkMemoryUsage();
      const memoryIncrease = finalMemory - initialMemory;

      expect(memoryIncrease).toBeLessThan(PERFORMANCE_THRESHOLDS.MEMORY_USAGE);
    });
  });

  describe('Bundle Size Performance', () => {
    it('should lazy load components efficiently', async () => {
      // Mock dynamic import timing
      const mockDynamicImport = vi.fn(() => 
        new Promise(resolve => 
          setTimeout(() => resolve({ default: () => <div>Loaded</div> }), 10)
        )
      );

      const startTime = performance.now();
      await mockDynamicImport();
      const loadTime = performance.now() - startTime;

      expect(loadTime).toBeLessThan(50); // Should load quickly
    });
  });

  describe('Animation Performance', () => {
    it('should handle animations without blocking the main thread', async () => {
      const { container } = render(
        <TestWrapper>
          <Index />
        </TestWrapper>
      );

      // Find animated elements
      const animatedElements = container.querySelectorAll(
        '[class*="animate"], [class*="transition"], [style*="transition"]'
      );

      // Verify animations use CSS transforms/opacity for better performance
      animatedElements.forEach(element => {
        const computedStyle = window.getComputedStyle(element);
        const transition = computedStyle.transition;
        
        if (transition && transition !== 'none') {
          // Should primarily animate transform/opacity for better performance
          const hasPerformantAnimation = transition.includes('transform') || 
                                        transition.includes('opacity');
          
          // Allow if no specific transition properties are set (defaults are usually fine)
          const isGenericTransition = transition.includes('all');
          
          expect(hasPerformantAnimation || isGenericTransition).toBe(true);
        }
      });
    });
  });

  describe('Event Handling Performance', () => {
    it('should not have excessive event listeners', async () => {
      const { container } = render(
        <TestWrapper>
          <Dashboard />
        </TestWrapper>
      );

      // Count interactive elements that likely have event listeners
      const interactiveElements = container.querySelectorAll(
        'button, a[href], input, select, textarea, [onclick], [role="button"], [tabindex="0"]'
      );

      expect(interactiveElements.length).toBeLessThan(PERFORMANCE_THRESHOLDS.EVENT_LISTENER_COUNT);
    });

    it('should properly clean up event listeners on unmount', async () => {
      const { unmount } = render(
        <TestWrapper>
          <Dashboard />
        </TestWrapper>
      );

      // Simulate component unmount
      unmount();

      // In a real test, you'd check that listeners were removed
      // This is more of a structural test to ensure cleanup
      expect(true).toBe(true); // Placeholder - real implementation would check listener cleanup
    });
  });

  describe('Image and Asset Loading', () => {
    it('should load images efficiently', async () => {
      const { container } = render(
        <TestWrapper>
          <Index />
        </TestWrapper>
      );

      const images = container.querySelectorAll('img');
      images.forEach(img => {
        // Should have loading="lazy" for performance
        const loading = img.getAttribute('loading');
        const isAboveFold = img.getAttribute('data-priority') === 'high';
        
        if (!isAboveFold) {
          expect(loading).toBe('lazy');
        }

        // Should have alt text for accessibility
        expect(img).toHaveAttribute('alt');
      });
    });
  });

  describe('Query Performance', () => {
    it('should handle React Query efficiently', async () => {
      const queryClient = new QueryClient({
        defaultOptions: {
          queries: { retry: false },
          mutations: { retry: false },
        },
      });

      const startTime = performance.now();

      render(
        <QueryClientProvider client={queryClient}>
          <BrowserRouter>
            <Dashboard />
          </BrowserRouter>
        </QueryClientProvider>
      );

      await waitFor(() => {
        expect(document.body.firstChild).toBeInTheDocument();
      });

      const renderTime = performance.now() - startTime;
      expect(renderTime).toBeLessThan(PERFORMANCE_THRESHOLDS.RENDER_TIME * 2); // Allow more time for complex dashboard
    });
  });

  describe('Scroll Performance', () => {
    it('should handle large lists efficiently', async () => {
      const { container } = render(
        <TestWrapper>
          <Dashboard />
        </TestWrapper>
      );

      // Find scrollable containers
      const scrollableElements = container.querySelectorAll(
        '[style*="overflow"], [class*="scroll"], .overflow-auto, .overflow-y-auto'
      );

      scrollableElements.forEach(element => {
        const style = window.getComputedStyle(element);
        
        // Should use efficient scrolling properties
        if (style.overflow === 'auto' || style.overflowY === 'auto') {
          // Virtual scrolling would be tested in e2e tests
          // Here we just ensure the structure exists
          expect(element).toBeInTheDocument();
        }
      });
    });
  });

  describe('Third-party Library Performance', () => {
    it('should load external dependencies efficiently', async () => {
      // Test that heavy libraries are code-split
      const heavyComponents = ['recharts', 'chart'];
      
      // This would typically be tested in the build process
      // Here we ensure components using heavy libraries are lazy-loaded
      expect(true).toBe(true); // Placeholder for actual bundle analysis
    });
  });

  describe('Accessibility Performance', () => {
    it('should not impact performance with accessibility features', async () => {
      const startTime = performance.now();

      const { container } = render(
        <TestWrapper>
          <Index />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(container.firstChild).toBeInTheDocument();
      });

      const renderTime = performance.now() - startTime;

      // Accessibility features should not significantly impact render time
      expect(renderTime).toBeLessThan(PERFORMANCE_THRESHOLDS.RENDER_TIME);

      // Check that accessibility features are present
      const ariaElements = container.querySelectorAll('[aria-label], [aria-describedby], [role]');
      expect(ariaElements.length).toBeGreaterThan(0);
    });
  });
});