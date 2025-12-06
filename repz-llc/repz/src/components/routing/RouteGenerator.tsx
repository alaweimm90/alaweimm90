import React, { Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { ProtectedRoute } from '@/components/ProtectedRoute';
import { AdminRoute } from '@/components/AdminRoute';
import { LoadingSpinner } from '@/components/ui/loading-spinner';
import { isFeatureEnabled } from '@/config/feature-flags';
import { ROUTES } from '@/constants/routes';

// Lazy load page components from their actual locations
const RepzHome = lazy(() => import('@/pages/RepzHome'));
const Index = lazy(() => import('@/pages/Index'));
const Login = lazy(() => import('@/components/auth/FitnessAuthPage').then(m => ({ default: m.FitnessAuthPage })));
const SignUp = lazy(() => import('@/components/auth/FitnessAuthPage').then(m => ({ default: m.FitnessAuthPage })));
const FitnessAuthPage = lazy(() => import('@/components/auth/FitnessAuthPage').then(m => ({ default: m.FitnessAuthPage })));

const SmartPricingPage = lazy(() => import('@/pages/pricing/SmartPricingPage'));
const SystemHealth = lazy(() => import('@/pages/SystemHealth'));
const CoachAdmin = lazy(() => import('@/pages/CoachAdmin'));
const ElegantPricing = lazy(() => import('@/pages/pricing/ElegantPricing'));
const TermsOfService = lazy(() => import('@/pages/TermsOfService'));
const PrivacyPolicy = lazy(() => import('@/pages/PrivacyPolicy'));
const LiabilityWaiver = lazy(() => import('@/pages/LiabilityWaiver'));
const HealthDisclaimer = lazy(() => import('@/pages/HealthDisclaimer'));
const Messages = lazy(() => import('@/pages/Messages'));
const Sessions = lazy(() => import('@/pages/Sessions'));
const MonthlyCoachingPrices = lazy(() => import('@/pages/MonthlyCoachingPrices'));
const InPersonTrainingPrices = lazy(() => import('@/pages/InPersonTrainingPrices'));
const Account = lazy(() => import('@/pages/Account'));
const PaymentSuccess = lazy(() => import('@/pages/PaymentSuccess'));

// Wrapper for Suspense with loading state (MOVED BEFORE USE)
interface SuspenseWrapperProps {
  component: React.LazyExoticComponent<React.ComponentType>;
  message: string;
}

const SuspenseWrapper: React.FC<SuspenseWrapperProps> = ({ component: Component, message }) => (
  <Suspense fallback={<LoadingSpinner size="lg" message={message} />}>
    <Component />
  </Suspense>
);

interface RouteConfig {
  path: string;
  element: React.ReactNode;
  protected?: boolean;
  adminRequired?: boolean;
  permissions?: string[];
  tierRequired?: string;
  devOnly?: boolean;
}

// Centralized route configuration with proper component loading
const ROUTE_CONFIGS: RouteConfig[] = [
  // Public routes
  { path: ROUTES.HOME, element: <SuspenseWrapper component={RepzHome} message="Loading home..." /> },
  { path: ROUTES.AUTH, element: <SuspenseWrapper component={FitnessAuthPage} message="Loading authentication..." /> },
  { path: ROUTES.LOGIN, element: <SuspenseWrapper component={Login} message="Loading login..." /> },
  { path: ROUTES.SIGNUP, element: <SuspenseWrapper component={SignUp} message="Loading signup..." /> },
  { path: ROUTES.TERMS, element: <SuspenseWrapper component={TermsOfService} message="Loading terms..." /> },
  { path: ROUTES.PRIVACY, element: <SuspenseWrapper component={PrivacyPolicy} message="Loading privacy policy..." /> },
  { path: ROUTES.LIABILITY_WAIVER, element: <SuspenseWrapper component={LiabilityWaiver} message="Loading waiver..." /> },
  { path: ROUTES.HEALTH_DISCLAIMER, element: <SuspenseWrapper component={HealthDisclaimer} message="Loading disclaimer..." /> },

  // Pricing routes
  { path: ROUTES.PRICING, element: <SuspenseWrapper component={SmartPricingPage} message="Loading pricing..." /> },
  { path: ROUTES.MONTHLY_COACHING, element: <SuspenseWrapper component={MonthlyCoachingPrices} message="Loading monthly coaching..." /> },
  { path: ROUTES.IN_PERSON_TRAINING, element: <SuspenseWrapper component={InPersonTrainingPrices} message="Loading in-person training..." /> },

  // Protected user routes
  { 
    path: ROUTES.DASHBOARD, 
    element: <SuspenseWrapper component={Index} message="Loading dashboard..." />,
    protected: true
  },
  { 
    path: ROUTES.MESSAGES, 
    element: <SuspenseWrapper component={Messages} message="Loading messages..." />, 
    protected: true 
  },
  { 
    path: ROUTES.SESSIONS, 
    element: <SuspenseWrapper component={Sessions} message="Loading sessions..." />, 
    protected: true 
  },
  { 
    path: ROUTES.PROGRESS, 
    element: <SuspenseWrapper component={Index} message="Loading progress..." />, 
    protected: true 
  },
  { 
    path: ROUTES.ACCOUNT, 
    element: <SuspenseWrapper component={Account} message="Loading account..." />, 
    protected: true 
  },
  { 
    path: ROUTES.PAYMENT_SUCCESS, 
    element: <SuspenseWrapper component={PaymentSuccess} message="Loading payment success..." />, 
    protected: true 
  },

  // Admin routes
  { 
    path: ROUTES.SYSTEM_HEALTH, 
    element: <SuspenseWrapper component={SystemHealth} message="Loading system health..." />,
    protected: true,
    adminRequired: true
  }
];

// Legacy redirects for missing routes
const LEGACY_REDIRECTS = [
  { from: ROUTES.PRICING_ELEGANT, to: ROUTES.PRICING },
  { from: ROUTES.PRICING_LEGACY, to: ROUTES.PRICING },
  { from: ROUTES.COMPREHENSIVE_PRICING, to: ROUTES.PRICING },
];

// Route wrapper that handles protection and admin requirements
interface RouteWrapperProps {
  config: RouteConfig;
}

const RouteWrapper: React.FC<RouteWrapperProps> = ({ config }) => {
  // Skip dev-only routes in production
  if (config.devOnly && process.env.NODE_ENV === 'production') {
    return null;
  }

  // Handle protected routes
  if (config.protected) {
    if (config.adminRequired) {
      return (
        <ProtectedRoute>
          <AdminRoute requiredPermissions={config.permissions}>
            {config.element}
          </AdminRoute>
        </ProtectedRoute>
      );
    }
    return <ProtectedRoute>{config.element}</ProtectedRoute>;
  }

  return <>{config.element}</>;
};

// Main route generator component
export const RouteGenerator: React.FC = () => {
  const useCentralizedRouting = isFeatureEnabled('USE_CENTRALIZED_ROUTING');

  if (!useCentralizedRouting) {
    // Return null to let App.tsx handle routing the old way
    return null;
  }

  return (
    <Routes>
      {/* Generate routes from configuration */}
      {ROUTE_CONFIGS.map((config) => (
        <Route
          key={config.path}
          path={config.path}
          element={<RouteWrapper config={config} />}
        />
      ))}

      {/* Legacy redirects */}
      {LEGACY_REDIRECTS.map((redirect) => (
        <Route
          key={redirect.from}
          path={redirect.from}
          element={<Navigate to={redirect.to} replace />}
        />
      ))}

      {/* 404 fallback */}
      <Route path="*" element={<Navigate to={ROUTES.HOME} replace />} />
    </Routes>
  );
};

// Export utilities for external use
export const getRouteConfigs = () => ROUTE_CONFIGS;
export const getLegacyRedirects = () => LEGACY_REDIRECTS;

export default RouteGenerator;