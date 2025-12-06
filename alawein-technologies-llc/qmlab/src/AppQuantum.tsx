import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter as Router } from 'react-router-dom';
import { QuantumWorkspace } from '@/components/QuantumWorkspace';
import { Toaster } from '@/components/ui/toaster';
import { AccessibilityProvider } from '@/components/AccessibilityProvider';
import { ErrorBoundary } from '@/components/ErrorBoundary';

// Create a client for React Query
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      retry: 1,
    },
  },
});

function AppQuantum() {
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <Router>
          <AccessibilityProvider>
            <div className="App">
              <QuantumWorkspace />
              <Toaster />
            </div>
          </AccessibilityProvider>
        </Router>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}

export { AppQuantum };