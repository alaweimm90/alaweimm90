import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Routes, Route } from "react-router-dom";
import ErrorBoundary from "@/components/ui/error-boundary";
import Layout from "./components/Layout";
import { Suspense, lazy } from "react";
// Resume Tailor module temporarily disabled and moved to src/_hidden for future integration

const queryClient = new QueryClient();

// Lazy load ALL page components for optimal performance
const Index = lazy(() => import("./pages/Index"));
const Scan = lazy(() => import("./pages/Scan"));
const Auth = lazy(() => import("./pages/Auth"));
const Results = lazy(() => import("./pages/Results"));
const Workspace = lazy(() => import("./pages/Workspace"));
const Settings = lazy(() => import("./pages/Settings"));
const Documentation = lazy(() => import("./pages/Documentation"));
const NotFound = lazy(() => import("./pages/NotFound"));

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <ErrorBoundary>
        <Toaster />
        <Sonner />
        <Layout>
          <Suspense fallback={
            <div className="flex items-center justify-center min-h-[60vh]">
              <div className="text-center space-y-4">
                <div className="animate-spin h-8 w-8 border-4 border-primary/20 border-t-primary mx-auto rounded-full" />
                <p className="text-muted-foreground">Loading…</p>
              </div>
            </div>
          }>
            <Routes>
              <Route path="/" element={<Index />} />
              <Route path="/scan" element={<Scan />} />
              <Route path="/results/:docId" element={<Results />} />
              <Route path="/workspace" element={<Workspace />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="/documentation" element={<Documentation />} />
              <Route path="/auth" element={<Auth />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </Suspense>
        </Layout>
      </ErrorBoundary>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
