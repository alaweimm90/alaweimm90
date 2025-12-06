import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { useTierAccess } from '@/hooks/useTierAccess';
import { TierType, UserRole } from '@/types/fitness';
import { TierGate } from '@/components/auth/TierGate';
import { logger } from '@/utils/logger';

interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredRole?: UserRole;
  requiredTier?: TierType;
  feature?: string;
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ 
  children, 
  requiredRole,
  requiredTier,
  feature 
}) => {
  const { isAuthenticated, loading, authChecked, user, role } = useAuth();
  const { hasMinimumTier } = useTierAccess();
  const location = useLocation();

  // Show loading until auth is properly checked
  if (loading || !authChecked) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-[hsl(var(--tier-adaptive))] mx-auto mb-4"></div>
          <p className="text-white">Authenticating...</p>
        </div>
      </div>
    );
  }

  // Redirect to login if not authenticated
  if (!isAuthenticated) {
    logger.info('Protected route: User not authenticated, redirecting to login');
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  // Check role requirements
  if (requiredRole && role !== requiredRole) {
    logger.warn('Protected route: Insufficient role, redirecting', { userRole: role, requiredRole });
    return <Navigate to="/" replace />;
  }

  // Check tier requirements with TierGate component
  if (requiredTier && !hasMinimumTier(requiredTier)) {
    return (
      <div className="min-h-screen flex items-center justify-center p-4">
        <TierGate requiredTier={requiredTier} feature={feature}>
          {children}
        </TierGate>
      </div>
    );
  }

  logger.info('Protected route: User authenticated with sufficient access', { userRole: role, requiredRole });
  return <>{children}</>;
};
