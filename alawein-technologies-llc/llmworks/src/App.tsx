import React, { Suspense, lazy, useEffect, useState } from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import AnalyticsListener from "@/components/AnalyticsListener";
import { SkipLink } from "@/components/SkipLink";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { PageLoader } from "@/components/LoadingSpinner";
import { initSmoothScroll } from "@/lib/smooth-scroll";
import { registerSW } from "@/lib/service-worker";
import { initPerformanceMonitoring } from "@/lib/performance";
import { AccessibilityToolbar } from "@/components/accessibility/AccessibilityToolbar";
import { ColorBlindnessFilters } from "@/components/accessibility/ColorBlindnessFilters";
import { KeyboardShortcuts } from "@/components/KeyboardShortcuts";
import { CommandPalette } from "@/components/CommandPalette";
import { useCommandPalette } from "@/hooks/useCommandPalette";
import { NotificationProvider } from "@/components/FloatingNotifications";
import { KeyboardProvider } from "@/components/KeyboardShortcuts";
import { DynamicBackground } from "@/components/DynamicBackground";
import { InteractiveBackground } from "@/components/MagneticElements";
import { ThemeCustomizer } from "@/components/ThemeCustomizer";
import { AchievementSystem } from "@/components/AchievementSystem";
// import { initSecurity } from "@/lib/security";
// import { initAdvancedSEO } from "@/lib/advanced-seo";
// import { initMonitoring } from "@/lib/monitoring";

const Index = lazy(() => import("./pages/Index"));
const Arena = lazy(() => import("./pages/Arena"));
const Bench = lazy(() => import("./pages/Bench"));
const Settings = lazy(() => import("./pages/Settings"));
const Admin = lazy(() => import("./pages/Admin"));
const Privacy = lazy(() => import("./pages/Privacy"));
const Terms = lazy(() => import("./pages/Terms"));
const NotFound = lazy(() => import("./pages/NotFound"));
const DashboardPage = lazy(() => import("./pages/Dashboard"));

// Optimized QueryClient configuration for better performance
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Cache queries for 5 minutes by default
      staleTime: 5 * 60 * 1000,
      // Keep unused data in cache for 10 minutes
      gcTime: 10 * 60 * 1000,
      // Retry failed requests 2 times
      retry: 2,
      // Don't refetch on window focus in production
      refetchOnWindowFocus: import.meta.env.DEV,
      // Don't refetch on reconnect unless in dev
      refetchOnReconnect: import.meta.env.DEV,
    },
    mutations: {
      // Retry failed mutations once
      retry: 1,
    },
  },
});

// Router wrapper component to provide context for hooks
const AppRoutes = () => {
  const { isOpen, setIsOpen } = useCommandPalette();
  const [themeCustomizerOpen, setThemeCustomizerOpen] = useState(false);

  return (
    <>
      <SkipLink />
      <AnalyticsListener />
      <DynamicBackground intensity="medium" theme="tactical" />
      <InteractiveBackground />
      <KeyboardShortcuts />
      <CommandPalette open={isOpen} onOpenChange={setIsOpen} />
      <ThemeCustomizer 
        isOpen={themeCustomizerOpen} 
        onToggle={() => setThemeCustomizerOpen(prev => !prev)} 
      />
      <AchievementSystem />
      <ErrorBoundary>
        <Suspense fallback={<PageLoader />}>
          <Routes>
            <Route path="/" element={<Index />} />
            <Route path="/arena" element={<Arena />} />
            <Route path="/bench" element={<Bench />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/admin" element={<Admin />} />
            <Route path="/privacy" element={<Privacy />} />
            <Route path="/terms" element={<Terms />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Suspense>
        <AccessibilityToolbar />
        <ColorBlindnessFilters />
      </ErrorBoundary>
    </>
  );
};

const App = () => {
  useEffect(() => {
    // Initialize production systems
    // initSecurity();
    // initAdvancedSEO();
    // initMonitoring();
    
    // Initialize smooth scrolling
    initSmoothScroll();
    
    // Initialize performance monitoring
    initPerformanceMonitoring({
      trackCoreWebVitals: true,
      trackNavigation: true,
      debug: import.meta.env.DEV,
    });
    
    // Register service worker for caching and offline functionality
    registerSW({
      onUpdate: (registration) => {
        console.log('New content available! Please refresh.');
        // Could show a toast notification here
      },
      onSuccess: () => {
        console.log('Content cached for offline use.');
      },
      onError: (error) => {
        console.error('Service worker registration failed:', error);
      },
    });
  }, []);

  return (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <NotificationProvider>
          <KeyboardProvider>
            <AppRoutes />
          </KeyboardProvider>
        </NotificationProvider>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
  );
};

export default App;
