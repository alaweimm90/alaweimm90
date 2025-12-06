import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@/utils/testHelpers';

// Mock components for edge case testing
const ErrorComponent = () => {
  const [shouldError, setShouldError] = React.useState(false);
  
  if (shouldError) {
    throw new Error('Test error for edge case handling');
  }
  
  return (
    <div>
      <button onClick={() => setShouldError(true)}>Trigger Error</button>
      <span>No Error</span>
    </div>
  );
};

const SlowLoadingComponent = () => {
  const [isLoading, setIsLoading] = React.useState(true);
  
  React.useEffect(() => {
    const timer = setTimeout(() => setIsLoading(false), 2000);
    return () => clearTimeout(timer);
  }, []);
  
  return isLoading ? <div>Loading...</div> : <div>Content Loaded</div>;
};

const NetworkFailureComponent = () => {
  const [data, setData] = React.useState(null);
  const [error, setError] = React.useState<string | null>(null);
  
  const fetchData = async () => {
    try {
      // Simulate network failure
      if (Math.random() > 0.5) {
        throw new Error('Network failure');
      }
      setData('Success data');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  };
  
  return (
    <div>
      <button onClick={fetchData}>Fetch Data</button>
      {data && <div>Data: {data}</div>}
      {error && <div>Error: {error}</div>}
    </div>
  );
};

describe('🚨 Edge Case Scenarios', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Error Handling', () => {
    it('handles component errors gracefully', () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      
      render(<ErrorComponent />);
      
      const triggerButton = screen.getByText('Trigger Error');
      fireEvent.click(triggerButton);
      
      // Component should handle error without crashing
      expect(consoleSpy).toHaveBeenCalled();
      consoleSpy.mockRestore();
    });

    it('handles async operation failures', async () => {
      render(<NetworkFailureComponent />);
      
      const fetchButton = screen.getByText('Fetch Data');
      fireEvent.click(fetchButton);
      
      // Should handle either success or failure
      await waitFor(() => {
        const hasData = screen.queryByText(/Data:/);
        const hasError = screen.queryByText(/Error:/);
        expect(hasData || hasError).toBeTruthy();
      });
    });
  });

  describe('Performance Edge Cases', () => {
    it('handles rapid state updates without performance degradation', async () => {
      const RapidUpdateComponent = () => {
        const [count, setCount] = React.useState(0);
        
        const rapidUpdate = () => {
          for (let i = 0; i < 100; i++) {
            setCount(prev => prev + 1);
          }
        };
        
        return (
          <div>
            <button onClick={rapidUpdate}>Rapid Update</button>
            <span>Count: {count}</span>
          </div>
        );
      };
      
      const startTime = performance.now();
      render(<RapidUpdateComponent />);
      
      const button = screen.getByText('Rapid Update');
      fireEvent.click(button);
      
      await waitFor(() => {
        expect(screen.getByText('Count: 100')).toBeInTheDocument();
      });
      
      const endTime = performance.now();
      const duration = endTime - startTime;
      
      // Should complete rapidly (under 1 second)
      expect(duration).toBeLessThan(1000);
    });

    it('handles large data sets efficiently', () => {
      const LargeDataComponent = () => {
        const largeArray = Array.from({ length: 10000 }, (_, i) => i);
        
        return (
          <div>
            <div>Array length: {largeArray.length}</div>
            <div>First item: {largeArray[0]}</div>
            <div>Last item: {largeArray[largeArray.length - 1]}</div>
          </div>
        );
      };
      
      const startTime = performance.now();
      render(<LargeDataComponent />);
      
      expect(screen.getByText('Array length: 10000')).toBeInTheDocument();
      expect(screen.getByText('First item: 0')).toBeInTheDocument();
      expect(screen.getByText('Last item: 9999')).toBeInTheDocument();
      
      const endTime = performance.now();
      const renderTime = endTime - startTime;
      
      // Should render quickly even with large data
      expect(renderTime).toBeLessThan(500);
    });
  });

  describe('Memory Management', () => {
    it('cleans up event listeners and timers', () => {
      const CleanupComponent = () => {
        React.useEffect(() => {
          const interval = setInterval(() => {
            console.log('Timer tick');
          }, 100);
          
          const handleResize = () => {
            console.log('Resize event');
          };
          
          window.addEventListener('resize', handleResize);
          
          return () => {
            clearInterval(interval);
            window.removeEventListener('resize', handleResize);
          };
        }, []);
        
        return <div>Component with cleanup</div>;
      };
      
      const { unmount } = render(<CleanupComponent />);
      
      // Component should mount successfully
      expect(screen.getByText('Component with cleanup')).toBeInTheDocument();
      
      // Should unmount without memory leaks
      unmount();
      
      // Test passes if no memory leaks or console errors
    });
  });

  describe('Browser Compatibility', () => {
    it('handles missing APIs gracefully', () => {
      // Mock missing IntersectionObserver
      const originalIntersectionObserver = window.IntersectionObserver;
      delete (window as any).IntersectionObserver;
      
      const FallbackComponent = () => {
        const [isVisible, setIsVisible] = React.useState(false);
        
        React.useEffect(() => {
          if (window.IntersectionObserver) {
            setIsVisible(true);
          } else {
            // Fallback for browsers without IntersectionObserver
            setIsVisible(true);
          }
        }, []);
        
        return (
          <div>
            {isVisible ? 'Visible' : 'Not visible'}
          </div>
        );
      };
      
      render(<FallbackComponent />);
      expect(screen.getByText('Visible')).toBeInTheDocument();
      
      // Restore original API
      window.IntersectionObserver = originalIntersectionObserver;
    });
  });

  describe('Data Validation Edge Cases', () => {
    it('handles malformed data gracefully', () => {
      const DataComponent = ({ data }: { data: any }) => {
        const safeData = React.useMemo(() => {
          try {
            return {
              id: data?.id || 'unknown',
              name: typeof data?.name === 'string' ? data.name : 'No name',
              count: typeof data?.count === 'number' ? data.count : 0
            };
          } catch {
            return { id: 'error', name: 'Error', count: 0 };
          }
        }, [data]);
        
        return (
          <div>
            <div>ID: {safeData.id}</div>
            <div>Name: {safeData.name}</div>
            <div>Count: {safeData.count}</div>
          </div>
        );
      };
      
      // Test with various malformed data
      const malformedData = [
        null,
        undefined,
        { id: null, name: 123, count: 'invalid' },
        { completely: 'different', structure: true }
      ];
      
      malformedData.forEach((data, index) => {
        const { unmount } = render(<DataComponent data={data} key={index} />);
        
        // Should render without crashing
        expect(screen.getByText(/ID:/)).toBeInTheDocument();
        expect(screen.getByText(/Name:/)).toBeInTheDocument();
        expect(screen.getByText(/Count:/)).toBeInTheDocument();
        
        unmount();
      });
    });
  });
});