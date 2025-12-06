
import { Toaster } from "@/ui/molecules/Toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from "@/contexts/AuthContext";
import { AnalyticsProvider } from "@/components/AnalyticsProvider";
import { AdminRoute } from "@/components/AdminRoute";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { SecurityProvider } from "@/security/SecurityProvider";
import { AccessibilityProvider } from "@/accessibility/AccessibilityProvider";
import "@/i18n/index";
import ErrorBoundary from "@/components/ErrorBoundary";
import { LoadingSpinner } from "@/components/ui/loading-spinner";
import { ProductionErrorBoundary } from "@/components/shared/ProductionErrorBoundary";
import { SEOHead } from "@/components/SEOHead";
import { useCoreWebVitals } from "@/hooks/usePerformanceMonitor";
import { Suspense, lazy } from "react";
import React from "react";

// Lazy load components that exist
const SystemHealth = lazy(() => import("./pages/SystemHealth"));
const CoachAdmin = lazy(() => import("./pages/CoachAdmin"));
const SystemDashboard = lazy(() => import("./pages/SystemDashboard"));
const TestingDashboard = lazy(() => import("./pages/TestingDashboard"));
const SmartPricingPage = lazy(() => import("./pages/pricing/SmartPricingPage"));
const ElegantPricing = lazy(() => import("./pages/pricing/ElegantPricing"));
const ElegantDashboard = lazy(() => import("./components/dashboard/ElegantDashboard"));
const IntakeEmail = lazy(() => import("./pages/IntakeEmail"));
const IntakeLanding = lazy(() => import("./pages/IntakeLanding"));
const IntakeSuccess = lazy(() => import("./pages/IntakeSuccess"));
const NonPortalClients = lazy(() => import("./pages/admin/NonPortalClients"));

// Static imports for critical pages
import Index from "./pages/Index";
const Login = lazy(() => import("./pages/auth/Login"));
const SignUp = lazy(() => import("./pages/auth/SignUp"));
const TermsOfService = lazy(() => import("./pages/TermsOfService"));
const PrivacyPolicy = lazy(() => import("./pages/PrivacyPolicy"));
const LiabilityWaiver = lazy(() => import("./pages/LiabilityWaiver"));
const HealthDisclaimer = lazy(() => import("./pages/HealthDisclaimer"));
const Messages = lazy(() => import("./pages/Messages"));
const Sessions = lazy(() => import("./pages/Sessions"));
const NotFound = lazy(() => import("./pages/NotFound"));
const MonthlyCoachingPrices = lazy(() => import("./pages/MonthlyCoachingPrices"));
const InPersonTrainingPrices = lazy(() => import("./pages/InPersonTrainingPrices"));
const InPersonTraining = lazy(() => import("./pages/InPersonTraining"));
const AIAssistant = lazy(() => import("./pages/AIAssistant"));
const Biomarkers = lazy(() => import("./pages/Biomarkers"));
const Pricing = lazy(() => import("./pages/Pricing"));
const Account = lazy(() => import("./pages/Account"));
const PaymentSuccess = lazy(() => import("./pages/PaymentSuccess"));

import { useTierAccess } from "@/hooks/useTierAccess";
import { TierType } from "@/constants/tiers";

// Tier gate component
const TierGate: React.FC<{ requiredTier: TierType; children: React.ReactNode }> = ({ requiredTier, children }) => {
  const { hasMinimumTier } = useTierAccess();

  if (!hasMinimumTier(requiredTier)) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900">
        <div className="text-center p-8 max-w-md">
          <h2 className="text-2xl font-bold text-white mb-4">Premium Feature</h2>
          <p className="text-gray-400 mb-6">
            This feature requires the {requiredTier.charAt(0).toUpperCase() + requiredTier.slice(1)} tier or higher.
          </p>
          <button
            className="px-6 py-3 bg-gradient-to-r from-orange-500 to-orange-600 text-white rounded-lg font-semibold hover:from-orange-600 hover:to-orange-700 transition-all"
            onClick={() => window.location.href = '/pricing'}
          >
            Upgrade Now
          </button>
        </div>
      </div>
    );
  }

  return <>{children}</>;
};

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5,
      gcTime: 1000 * 60 * 10,
      refetchOnWindowFocus: false,
      refetchOnReconnect: false,
      retry: 2,
    },
  },
});

const TierAwareDashboard = () => {
  return (
    <Suspense fallback={<LoadingSpinner size="lg" message="Loading dashboard..." />}>
      <ElegantDashboard userName="Athlete" />
    </Suspense>
  );
};

const App = () => {
  try {
    useCoreWebVitals();
  } catch (error) {
    console.warn('[App] Performance monitoring disabled:', error);
  }

  return (
    <ProductionErrorBoundary componentName="App">
      <ErrorBoundary>
        <QueryClientProvider client={queryClient}>
          <SecurityProvider>
            <AccessibilityProvider>
              <AuthProvider>
                <TooltipProvider>
                  <Toaster />
                  <Sonner />
                  <BrowserRouter>
                    <SEOHead />
                    <AnalyticsProvider>
                      <Routes>
                        <Route path="/" element={<Index />} />

                        {/* Authentication Routes */}
                        <Route path="/login" element={
                          <Suspense fallback={<LoadingSpinner size="lg" message="Loading login..." />}>
                            <Login />
                          </Suspense>
                        } />
                        <Route path="/signup" element={
                          <Suspense fallback={<LoadingSpinner size="lg" message="Loading signup..." />}>
                            <SignUp />
                          </Suspense>
                        } />

                        {/* Protected Routes */}

                        {/* Pricing Routes */}
                        <Route path="/pricing" element={
                          <Suspense fallback={<LoadingSpinner size="lg" message="Loading pricing..." />}>
                            <SmartPricingPage />
                          </Suspense>
                        } />

                        <Route path="/account" element={
                          <ProtectedRoute>
                            <Suspense fallback={<LoadingSpinner size="lg" message="Loading account..." />}>
                              <Account />
                            </Suspense>
                          </ProtectedRoute>
                        } />

                        <Route path="/dashboard" element={
                          <ProtectedRoute>
                            <TierAwareDashboard />
                          </ProtectedRoute>
                        } />

                        <Route path="/messages" element={
                          <ProtectedRoute>
                            <Suspense fallback={<LoadingSpinner size="lg" message="Loading messages..." />}>
                              <Messages />
                            </Suspense>
                          </ProtectedRoute>
                        } />

                        <Route path="/sessions" element={
                          <ProtectedRoute>
                            <Suspense fallback={<LoadingSpinner size="lg" message="Loading sessions..." />}>
                              <Sessions />
                            </Suspense>
                          </ProtectedRoute>
                        } />

                        <Route path="/coach-admin" element={
                          <AdminRoute>
                            <Suspense fallback={<LoadingSpinner size="lg" message="Loading admin..." />}>
                              <CoachAdmin />
                            </Suspense>
                          </AdminRoute>
                        } />

                        <Route path="/system-health" element={
                          <AdminRoute>
                            <Suspense fallback={<LoadingSpinner size="lg" message="Loading system health..." />}>
                              <SystemHealth />
                            </Suspense>
                          </AdminRoute>
                        } />

                        <Route path="/system-dashboard" element={
                          <AdminRoute>
                            <Suspense fallback={<LoadingSpinner size="lg" message="Loading system dashboard..." />}>
                              <SystemDashboard />
                            </Suspense>
                          </AdminRoute>
                        } />

                        <Route path="/testing-dashboard" element={
                          <AdminRoute>
                            <Suspense fallback={<LoadingSpinner size="lg" message="Loading testing dashboard..." />}>
                              <TestingDashboard />
                            </Suspense>
                          </AdminRoute>
                        } />

                        <Route
                          path="/ai-assistant"
                          element={
                            <ProtectedRoute>
                              <TierGate requiredTier="performance">
                                <Suspense fallback={<LoadingSpinner size="lg" message="Loading AI assistant..." />}>
                                  <AIAssistant />
                                </Suspense>
                              </TierGate>
                            </ProtectedRoute>
                          }
                        />

                        <Route
                          path="/biomarkers"
                          element={
                            <ProtectedRoute>
                              <TierGate requiredTier="adaptive">
                                <Suspense fallback={<LoadingSpinner size="lg" message="Loading biomarkers..." />}>
                                  <Biomarkers />
                                </Suspense>
                              </TierGate>
                            </ProtectedRoute>
                          }
                        />

                        <Route
                          path="/in-person-training"
                          element={
                            <ProtectedRoute>
                              <TierGate requiredTier="longevity">
                                <Suspense fallback={<LoadingSpinner size="lg" message="Loading training..." />}>
                                  <InPersonTraining />
                                </Suspense>
                              </TierGate>
                            </ProtectedRoute>
                          }
                        />

                        {/* Non-Portal Client Intake */}
                        <Route path="/intake" element={
                          <Suspense fallback={<LoadingSpinner size="lg" message="Loading..." />}>
                            <IntakeLanding />
                          </Suspense>
                        } />
                        <Route path="/intake-email" element={
                          <Suspense fallback={<LoadingSpinner size="lg" message="Loading intake form..." />}>
                            <IntakeEmail />
                          </Suspense>
                        } />
                        <Route path="/intake-success" element={
                          <Suspense fallback={<LoadingSpinner size="lg" message="Loading..." />}>
                            <IntakeSuccess />
                          </Suspense>
                        } />

                        {/* Admin: Non-Portal Clients */}
                        <Route path="/admin/non-portal-clients" element={
                          <AdminRoute>
                            <Suspense fallback={<LoadingSpinner size="lg" message="Loading non-portal clients..." />}>
                              <NonPortalClients />
                            </Suspense>
                          </AdminRoute>
                        } />

                        {/* Legal and static pages */}
                        <Route path="/terms-of-service" element={
                          <Suspense fallback={<LoadingSpinner size="lg" message="Loading..." />}>
                            <TermsOfService />
                          </Suspense>
                        } />
                        <Route path="/privacy-policy" element={
                          <Suspense fallback={<LoadingSpinner size="lg" message="Loading..." />}>
                            <PrivacyPolicy />
                          </Suspense>
                        } />
                        <Route path="/liability-waiver" element={
                          <Suspense fallback={<LoadingSpinner size="lg" message="Loading..." />}>
                            <LiabilityWaiver />
                          </Suspense>
                        } />
                        <Route path="/health-disclaimer" element={
                          <Suspense fallback={<LoadingSpinner size="lg" message="Loading..." />}>
                            <HealthDisclaimer />
                          </Suspense>
                        } />
                        <Route path="/monthly-coaching-prices" element={
                          <Suspense fallback={<LoadingSpinner size="lg" message="Loading pricing..." />}>
                            <MonthlyCoachingPrices />
                          </Suspense>
                        } />
                        <Route path="/in-person-training-prices" element={
                          <Suspense fallback={<LoadingSpinner size="lg" message="Loading pricing..." />}>
                            <InPersonTrainingPrices />
                          </Suspense>
                        } />
                        <Route path="/payment-success" element={
                          <Suspense fallback={<LoadingSpinner size="lg" message="Loading payment status..." />}>
                            <PaymentSuccess />
                          </Suspense>
                        } />

                        <Route path="*" element={
                          <Suspense fallback={<LoadingSpinner size="lg" message="Loading..." />}>
                            <NotFound />
                          </Suspense>
                        } />
                      </Routes>
                    </AnalyticsProvider>
                  </BrowserRouter>
                </TooltipProvider>
              </AuthProvider>
            </AccessibilityProvider>
          </SecurityProvider>
        </QueryClientProvider>
      </ErrorBoundary>
    </ProductionErrorBoundary>
  );
};

export default App;
