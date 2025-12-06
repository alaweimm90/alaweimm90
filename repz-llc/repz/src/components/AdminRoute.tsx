import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { useAdmin } from '@/hooks/useAdmin';
import { Card, CardContent } from '@/ui/molecules/Card';
import { AlertCircle, Loader2 } from 'lucide-react';

interface AdminRouteProps {
  children: React.ReactNode;
  requiredPermissions?: string[];
}

const AdminRoute: React.FC<AdminRouteProps> = ({ 
  children, 
  requiredPermissions = [] 
}) => {
  const { user } = useAuth();
  const { isAdmin, loading, hasAnyPermission, adminUser } = useAdmin();

  // Show loading spinner while checking admin status
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Card className="w-96">
          <CardContent className="p-6 text-center">
            <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4 text-primary" />
            <h3 className="text-lg font-semibold mb-2">Verifying Access</h3>
            <p className="text-muted-foreground">Checking admin permissions...</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Redirect to login if not authenticated
  if (!user) {
    return <Navigate to="/" replace />;
  }

  // Show access denied if not admin
  if (!isAdmin) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Card className="w-96">
          <CardContent className="p-6 text-center">
            <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">Access Denied</h3>
            <p className="text-muted-foreground mb-4">
              You don't have admin privileges to access this page.
            </p>
            <p className="text-sm text-muted-foreground">
              Contact an administrator if you believe this is an error.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Check specific permissions if required
  if (requiredPermissions.length > 0 && !hasAnyPermission(requiredPermissions)) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Card className="w-96">
          <CardContent className="p-6 text-center">
            <AlertCircle className="w-12 h-12 text-amber-500 mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">Insufficient Permissions</h3>
            <p className="text-muted-foreground mb-4">
              You don't have the required permissions to access this feature.
            </p>
            <div className="text-sm text-muted-foreground space-y-1">
              <p><strong>Your email:</strong> {user?.email}</p>
              <p><strong>Admin status:</strong> {isAdmin ? 'Yes' : 'No'}</p>
              <p><strong>Required permissions:</strong> {requiredPermissions.join(', ')}</p>
              {adminUser && (
                <p><strong>Your permissions:</strong> Admin</p>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Render the protected content
  return <>{children}</>;
};

export { AdminRoute };