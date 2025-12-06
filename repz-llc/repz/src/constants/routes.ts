// Type-safe route configuration for REPZ platform
export const ROUTES = {
  // Public routes
  HOME: '/',
  AUTH: '/auth',
  LOGIN: '/login',
  SIGNUP: '/signup',
  IN_PERSON: '/in-person',
  PLANS: '/plans',
  
  // Protected routes
  DASHBOARD: '/dashboard',
  DASHBOARD_LEGACY: '/dashboard-legacy',
  PROGRESS: '/progress',
  SESSIONS: '/sessions',
  MESSAGES: '/messages',
  PROFILE: '/profile',
  
  // Coach routes
  COACH_DASHBOARD: '/coach',
  COACH_CLIENTS: '/coach/clients',
  COACH_ANALYTICS: '/coach/analytics',
  
  // Admin routes
  ADMIN: '/admin',
  RESERVATION_ADMIN: '/reservation-admin',
  SYSTEM_HEALTH: '/system-health',
  
  // Intake flow
  INTAKE_START: '/intake',
  INTAKE_STEP: '/intake/step/:stepNumber',
  
  // Pricing routes
  PRICING: '/pricing',
  PRICING_ELEGANT: '/pricing-elegant',
  PRICING_LEGACY: '/pricing-legacy', 
  COMPREHENSIVE_PRICING: '/comprehensive-pricing',
  MONTHLY_COACHING: '/monthly-coaching-prices',
  ACCOUNT: '/account',
  PAYMENT_SUCCESS: '/payment-success',
  IN_PERSON_TRAINING: '/in-person-training-prices',
  TERMS: '/terms-of-service',
  PRIVACY: '/privacy-policy',
  LIABILITY_WAIVER: '/liability-waiver',
  HEALTH_DISCLAIMER: '/health-disclaimer',
  ANALYTICS: '/analytics',
  PRODUCTION: '/production',
  PROJECT_STATUS: '/project-status',
  INTEGRATION_STATUS: '/integration-status',
  TIERS_TEMPLATES: '/tiers-templates',
} as const;

export type RouteKey = keyof typeof ROUTES;
export type RoutePath = typeof ROUTES[RouteKey];

// Tier types for access control
export type TierType = 'core' | 'adaptive' | 'performance' | 'longevity';
export type UserRole = 'client' | 'coach' | 'admin';

// Route metadata for navigation and access control
export interface RouteMetadata {
  path: RoutePath;
  title: string;
  requiresAuth: boolean;
  requiredRole?: UserRole;
  requiredTier?: TierType;
  showInNavigation: boolean;
  mobileOnly?: boolean;
  icon?: string;
  description?: string;
}

export const ROUTE_METADATA: Record<string, RouteMetadata> = {
  HOME: {
    path: ROUTES.HOME,
    title: 'Home',
    requiresAuth: false,
    showInNavigation: false,
    icon: 'home'
  },
  DASHBOARD: {
    path: ROUTES.DASHBOARD,
    title: 'Dashboard',
    requiresAuth: true,
    requiredRole: 'client',
    showInNavigation: true,
    icon: 'home',
    description: 'Your personal dashboard'
  },
  PROGRESS: {
    path: ROUTES.PROGRESS,
    title: 'Progress',
    requiresAuth: true,
    requiredRole: 'client',
    requiredTier: 'core',
    showInNavigation: true,
    icon: 'trending-up',
    description: 'Track your fitness progress'
  },
  SESSIONS: {
    path: ROUTES.SESSIONS,
    title: 'Sessions',
    requiresAuth: true,
    requiredRole: 'client',
    requiredTier: 'adaptive',
    showInNavigation: true,
    icon: 'calendar',
    description: 'Book and manage coaching sessions'
  },
  MESSAGES: {
    path: ROUTES.MESSAGES,
    title: 'Messages',
    requiresAuth: true,
    requiredRole: 'client',
    requiredTier: 'adaptive',
    showInNavigation: true,
    icon: 'message-circle',
    description: 'Chat with your coach'
  },
  COACH_DASHBOARD: {
    path: ROUTES.COACH_DASHBOARD,
    title: 'Dashboard',
    requiresAuth: true,
    requiredRole: 'coach',
    showInNavigation: true,
    icon: 'home',
    description: 'Coach dashboard'
  },
  COACH_CLIENTS: {
    path: ROUTES.COACH_CLIENTS,
    title: 'Clients',
    requiresAuth: true,
    requiredRole: 'coach',
    showInNavigation: true,
    icon: 'users',
    description: 'Manage your clients'
  },
  ANALYTICS: {
    path: ROUTES.ANALYTICS,
    title: 'Analytics',
    requiresAuth: true,
    requiredRole: 'coach',
    showInNavigation: true,
    icon: 'bar-chart',
    description: 'View analytics and insights'
  },
  PLANS: {
    path: ROUTES.PLANS,
    title: 'Plans',
    requiresAuth: false,
    showInNavigation: false,
    icon: 'credit-card',
    description: 'View subscription plans'
  }
};

// Route validation utilities
export const isValidRoute = (path: string): path is RoutePath => {
  return Object.values(ROUTES).includes(path as RoutePath);
};

export const getRouteMetadata = (path: RoutePath): RouteMetadata | undefined => {
  const routeKey = Object.entries(ROUTES).find(([_, routePath]) => routePath === path)?.[0];
  return routeKey ? ROUTE_METADATA[routeKey] : undefined;
};

// Get routes for specific role
export const getRoutesForRole = (role: UserRole): RouteMetadata[] => {
  return Object.values(ROUTE_METADATA).filter(
    metadata => !metadata.requiredRole || metadata.requiredRole === role
  );
};

// Get navigation items for mobile
export const getNavigationItems = (role: UserRole, tier?: TierType): RouteMetadata[] => {
  return Object.values(ROUTE_METADATA).filter(metadata => {
    // Must be shown in navigation
    if (!metadata.showInNavigation) return false;
    
    // Check role requirements
    if (metadata.requiredRole && metadata.requiredRole !== role) return false;
    
    // Check tier requirements for clients
    if (metadata.requiredTier && role === 'client') {
      if (!tier) return false;
      
      const tierHierarchy: Record<TierType, number> = {
        core: 1,
        adaptive: 2,
        performance: 3,
        longevity: 4
      };
      
      const userTierLevel = tierHierarchy[tier];
      const requiredTierLevel = tierHierarchy[metadata.requiredTier];
      
      return userTierLevel >= requiredTierLevel;
    }
    
    return true;
  });
};

// Safe navigation helper
export const createSafeNavigate = (navigate: (path: string) => void) => {
  return (path: string) => {
    if (isValidRoute(path)) {
      navigate(path);
    } else {
      console.error(`[ROUTES] Invalid route attempted: ${path}`);
      // Fallback to dashboard or home
      navigate(ROUTES.DASHBOARD);
    }
  };
};

// Replace hardcoded navigation paths with centralized navigation
export const navigateToRoute = (route: RoutePath) => {
  // Replace window.location.href = '/pricing' with proper navigation
  if (typeof window !== 'undefined') {
    window.location.href = route;
  }
};

// Specific navigation helpers to replace hardcoded paths
export const navigateToPricing = () => navigateToRoute(ROUTES.PRICING);
export const navigateToDashboard = () => navigateToRoute(ROUTES.DASHBOARD);
export const navigateToLogin = () => navigateToRoute(ROUTES.LOGIN);
export const navigateToSignup = () => navigateToRoute(ROUTES.SIGNUP);

// Route validation for development
export const validateAllRoutes = (): { isValid: boolean; errors: string[] } => {
  const errors: string[] = [];
  
  // Check for duplicate routes
  const routePaths = Object.values(ROUTES);
  const duplicates = routePaths.filter((path, index) => routePaths.indexOf(path) !== index);
  
  if (duplicates.length > 0) {
    errors.push(`Duplicate routes found: ${duplicates.join(', ')}`);
  }
  
  // Check for missing metadata
  Object.entries(ROUTES).forEach(([key, path]) => {
    if (!ROUTE_METADATA[key]) {
      errors.push(`Missing metadata for route: ${key} (${path})`);
    }
  });
  
  return {
    isValid: errors.length === 0,
    errors
  };
};