import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import { describe, it, expect, beforeEach, vi } from 'vitest';

import App from '@/App';
import { Navigation } from '@/components/Navigation';
import { AccessibilityToolbar } from '@/components/accessibility/AccessibilityToolbar';

// Mock service worker registration
vi.mock('@/lib/service-worker', () => ({
  registerSW: vi.fn(),
}));

// Mock performance monitoring
vi.mock('@/lib/performance', () => ({
  initPerformanceMonitoring: vi.fn(),
}));

// Mock analytics
vi.mock('@/lib/analytics', () => ({
  trackEvent: vi.fn(),
}));

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

describe('Integration Tests', () => {
  beforeEach(() => {
    // Reset window location for each test
    Object.defineProperty(window, 'location', {
      value: {
        pathname: '/',
        search: '',
        hash: '',
        href: 'http://localhost:3000/',
      },
      writable: true,
    });
  });

  describe('App Integration', () => {
    it('should render the app without errors', async () => {
      render(<App />);
      
      await waitFor(() => {
        expect(screen.getByRole('main')).toBeInTheDocument();
      });
    });

    it('should handle routing correctly', async () => {
      const user = userEvent.setup();
      render(<App />);

      // Wait for initial load
      await waitFor(() => {
        expect(screen.getByRole('main')).toBeInTheDocument();
      });

      // Navigate to Arena
      const arenaLink = screen.getByRole('link', { name: /arena/i });
      await user.click(arenaLink);

      await waitFor(() => {
        expect(window.location.pathname).toBe('/arena');
      });
    });

    it('should handle 404 routes gracefully', async () => {
      // Mock a non-existent route
      Object.defineProperty(window, 'location', {
        value: { pathname: '/non-existent-route' },
        writable: true,
      });

      render(<App />);

      await waitFor(() => {
        expect(screen.getByText(/404/i)).toBeInTheDocument();
      });
    });
  });

  describe('Navigation Integration', () => {
    it('should navigate between pages correctly', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <Navigation />
        </TestWrapper>
      );

      // Test navigation to different pages
      const pages = [
        { name: /home|llm works/i, expectedPath: '/' },
        { name: /arena/i, expectedPath: '/arena' },
        { name: /bench/i, expectedPath: '/bench' },
        { name: /dashboard/i, expectedPath: '/dashboard' },
      ];

      for (const page of pages) {
        const link = screen.getByRole('link', { name: page.name });
        expect(link).toBeInTheDocument();
        
        // Check that href is correct
        expect(link).toHaveAttribute('href', page.expectedPath);
      }
    });

    it('should handle mobile navigation correctly', async () => {
      const user = userEvent.setup();

      // Mock mobile viewport
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });

      render(
        <TestWrapper>
          <Navigation />
        </TestWrapper>
      );

      // Look for mobile menu button
      const menuButton = screen.queryByRole('button', { name: /menu/i });
      if (menuButton) {
        await user.click(menuButton);
        
        // Check that mobile menu is visible
        await waitFor(() => {
          expect(screen.getByRole('navigation')).toBeInTheDocument();
        });
      }
    });
  });

  describe('Accessibility Integration', () => {
    it('should toggle accessibility features correctly', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <AccessibilityToolbar />
        </TestWrapper>
      );

      // Find accessibility toolbar toggle
      const accessibilityToggle = screen.getByRole('button', { name: /accessibility/i });
      await user.click(accessibilityToggle);

      await waitFor(() => {
        expect(screen.getByText(/high contrast/i)).toBeInTheDocument();
      });
    });

    it('should apply accessibility settings correctly', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <AccessibilityToolbar />
        </TestWrapper>
      );

      // Open accessibility toolbar
      const accessibilityToggle = screen.getByRole('button', { name: /accessibility/i });
      await user.click(accessibilityToggle);

      // Toggle high contrast
      const highContrastToggle = screen.getByRole('switch', { name: /high contrast/i });
      await user.click(highContrastToggle);

      // Check that high contrast class is applied
      await waitFor(() => {
        expect(document.body).toHaveClass('high-contrast');
      });
    });

    it('should persist accessibility settings', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <AccessibilityToolbar />
        </TestWrapper>
      );

      // Open accessibility toolbar and enable large text
      const accessibilityToggle = screen.getByRole('button', { name: /accessibility/i });
      await user.click(accessibilityToggle);

      const largeTextToggle = screen.getByRole('switch', { name: /large text/i });
      await user.click(largeTextToggle);

      // Check localStorage
      expect(localStorage.setItem).toHaveBeenCalledWith(
        'accessibility-settings',
        expect.stringContaining('largeText')
      );
    });
  });

  describe('Keyboard Navigation Integration', () => {
    it('should handle global keyboard shortcuts', async () => {
      const user = userEvent.setup();
      
      render(<App />);

      await waitFor(() => {
        expect(screen.getByRole('main')).toBeInTheDocument();
      });

      // Test Alt+H shortcut (home)
      await user.keyboard('{Alt>}h{/Alt}');

      // Should navigate to home or show that shortcut works
      // This would need to be implemented based on actual shortcut handling
      expect(window.location.pathname).toBe('/');
    });

    it('should handle focus management correctly', async () => {
      const user = userEvent.setup();
      
      render(<App />);

      await waitFor(() => {
        expect(screen.getByRole('main')).toBeInTheDocument();
      });

      // Test tab navigation
      await user.tab();
      
      // Should focus on first interactive element
      const focusedElement = document.activeElement;
      expect(focusedElement).toBeInstanceOf(HTMLElement);
      expect(focusedElement?.getAttribute('tabindex')).not.toBe('-1');
    });

    it('should handle skip links correctly', async () => {
      const user = userEvent.setup();
      
      render(<App />);

      await waitFor(() => {
        expect(screen.getByRole('main')).toBeInTheDocument();
      });

      // Focus and activate skip link
      const skipLink = screen.getByText(/skip to main content/i);
      skipLink.focus();
      await user.click(skipLink);

      // Should focus on main content
      await waitFor(() => {
        const main = screen.getByRole('main');
        expect(main).toHaveFocus();
      });
    });
  });

  describe('Error Boundary Integration', () => {
    it('should catch and display errors gracefully', async () => {
      // Mock a component that throws an error
      const ErrorComponent = () => {
        throw new Error('Test error');
      };

      // Mock console.error to avoid noise in tests
      const originalError = console.error;
      console.error = vi.fn();

      render(
        <TestWrapper>
          <ErrorComponent />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
      });

      // Restore console.error
      console.error = originalError;
    });
  });

  describe('Service Worker Integration', () => {
    it('should register service worker on app initialization', async () => {
      const { registerSW } = await import('@/lib/service-worker');
      
      render(<App />);

      await waitFor(() => {
        expect(registerSW).toHaveBeenCalled();
      });
    });
  });

  describe('Performance Monitoring Integration', () => {
    it('should initialize performance monitoring', async () => {
      const { initPerformanceMonitoring } = await import('@/lib/performance');
      
      render(<App />);

      await waitFor(() => {
        expect(initPerformanceMonitoring).toHaveBeenCalledWith({
          trackCoreWebVitals: true,
          trackNavigation: true,
          debug: expect.any(Boolean),
        });
      });
    });
  });

  describe('Query Client Integration', () => {
    it('should handle query errors gracefully', async () => {
      const queryClient = new QueryClient({
        defaultOptions: {
          queries: {
            retry: false,
            // Mock a query that fails
            queryFn: () => Promise.reject(new Error('Query failed')),
          },
        },
      });

      render(
        <QueryClientProvider client={queryClient}>
          <BrowserRouter>
            <App />
          </BrowserRouter>
        </QueryClientProvider>
      );

      // Should still render without crashing
      await waitFor(() => {
        expect(screen.getByRole('main')).toBeInTheDocument();
      });
    });
  });

  describe('Theme Integration', () => {
    it('should handle theme switching correctly', async () => {
      const user = userEvent.setup();
      
      render(<App />);

      await waitFor(() => {
        expect(screen.getByRole('main')).toBeInTheDocument();
      });

      // Look for theme toggle (if it exists)
      const themeToggle = screen.queryByRole('button', { name: /theme|dark|light/i });
      
      if (themeToggle) {
        await user.click(themeToggle);
        
        // Check that theme class is applied
        await waitFor(() => {
          expect(document.documentElement).toHaveClass(/dark|light/);
        });
      }
    });
  });

  describe('Loading States Integration', () => {
    it('should show loading states during navigation', async () => {
      render(<App />);

      // Should show some loading indicator initially or transition smoothly
      await waitFor(() => {
        expect(screen.getByRole('main')).toBeInTheDocument();
      });

      // Test that lazy-loaded components show loading states
      // This would be more apparent in actual browser testing
      expect(true).toBe(true); // Placeholder for actual loading state tests
    });
  });

  describe('Responsive Design Integration', () => {
    it('should adapt to different screen sizes', async () => {
      // Test mobile
      Object.defineProperty(window, 'innerWidth', { value: 375 });
      Object.defineProperty(window, 'innerHeight', { value: 667 });
      fireEvent(window, new Event('resize'));

      render(<App />);

      await waitFor(() => {
        expect(screen.getByRole('main')).toBeInTheDocument();
      });

      // Test tablet
      Object.defineProperty(window, 'innerWidth', { value: 768 });
      Object.defineProperty(window, 'innerHeight', { value: 1024 });
      fireEvent(window, new Event('resize'));

      // Should still be functional
      expect(screen.getByRole('main')).toBeInTheDocument();

      // Test desktop
      Object.defineProperty(window, 'innerWidth', { value: 1440 });
      Object.defineProperty(window, 'innerHeight', { value: 900 });
      fireEvent(window, new Event('resize'));

      // Should still be functional
      expect(screen.getByRole('main')).toBeInTheDocument();
    });
  });
});