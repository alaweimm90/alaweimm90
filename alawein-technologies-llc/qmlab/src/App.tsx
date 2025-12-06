import { useEffect, useState, lazy, Suspense } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { QuantumErrorBoundary } from "@/components/QuantumErrorBoundary";
import { AccessibilityProvider } from "@/components/AccessibilityProvider";
import { ComponentSkeleton } from "@/components/ComponentSkeleton";

// Lazy load non-critical components
const Toaster = lazy(() => import("@/components/ui/toaster").then(m => ({ default: m.Toaster })));
const Sonner = lazy(() => import("@/components/ui/sonner").then(m => ({ default: m.Toaster })));
const TooltipProvider = lazy(() => import("@/components/ui/tooltip").then(m => ({ default: m.TooltipProvider })));
const TutorialProvider = lazy(() => import("@/contexts/TutorialContext").then(m => ({ default: m.TutorialProvider })));
const TutorialOverlay = lazy(() => import("@/components/TutorialOverlay").then(m => ({ default: m.TutorialOverlay })));
const CoreWebVitalsMonitor = lazy(() => import("@/components/CoreWebVitalsMonitor").then(m => ({ default: m.CoreWebVitalsMonitor })));

// Lazy load pages
const Index = lazy(() => import("./pages/Index"));
const NotFound = lazy(() => import("./pages/NotFound"));

const queryClient = new QueryClient();

// Component to track page views
function PageViewTracker() {
  const location = useLocation();
  
  useEffect(() => {
    // Defer analytics tracking
    requestIdleCallback(() => {
      import('@/lib/analytics').then(({ trackEvent }) => {
        trackEvent('page_view', {
          page_path: location.pathname,
          page_location: window.location.href
        });
      });
    });
  }, [location]);
  
  return null;
}

const App = () => {
  // Initialize Core Web Vitals monitoring with state
  const [coreWebVitals, setCoreWebVitals] = useState({});

  const handleMetricsUpdate = (metrics: Record<string, unknown>) => {
    setCoreWebVitals(metrics);
  };

  useEffect(() => {
    // Defer non-critical initialization
    const timer = requestIdleCallback(() => {
      // Lazy load analytics and monitoring
      import('@/lib/monitoring').then(({ initErrorReporting }) => {
        initErrorReporting({
          environment: import.meta.env.MODE,
          release: import.meta.env.VITE_APP_VERSION || 'development',
          enablePerformanceMonitoring: true
        });
      });

      import('@/lib/analytics').then(({ initGA, trackEvent }) => {
        initGA();
        
        // Track initial page load time
        if (window.performance && window.performance.timing) {
          const loadTime = window.performance.timing.loadEventEnd - window.performance.timing.navigationStart;
          if (loadTime > 0) {
            trackEvent('page_load_time', { load_time_ms: loadTime });
          }
        }
      });

      // Load GTM helpers after idle
      import('@/lib/gtm-helpers');
    });

    return () => {
      if (timer) cancelIdleCallback(timer);
    };
  }, []);
  
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <AccessibilityProvider>
          <Suspense fallback={<ComponentSkeleton className="h-screen" />}>
            <TutorialProvider>
              <TooltipProvider>
                <Suspense fallback={null}>
                  <Toaster />
                  <Sonner />
                </Suspense>
                <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
                  <PageViewTracker />
                  <Suspense fallback={null}>
                    <CoreWebVitalsMonitor onMetricsUpdate={handleMetricsUpdate} />
                  </Suspense>
                  <Routes>
                    <Route path="/" element={
                      <QuantumErrorBoundary>
                        <ErrorBoundary>
                          <Suspense fallback={<ComponentSkeleton className="h-screen" />}>
                            <Index />
                          </Suspense>
                        </ErrorBoundary>
                      </QuantumErrorBoundary>
                    } />
                    {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
                    <Route path="*" element={
                      <QuantumErrorBoundary>
                        <ErrorBoundary>
                          <Suspense fallback={<ComponentSkeleton className="h-screen" />}>
                            <NotFound />
                          </Suspense>
                        </ErrorBoundary>
                      </QuantumErrorBoundary>
                    } />
                  </Routes>
                  <Suspense fallback={null}>
                    <TutorialOverlay />
                  </Suspense>
                </BrowserRouter>
              </TooltipProvider>
            </TutorialProvider>
          </Suspense>
        </AccessibilityProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  );
};

export { App };
