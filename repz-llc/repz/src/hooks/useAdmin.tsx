import { useAuth } from '@/contexts/AuthContext';
import { useState, useEffect } from 'react';

export const useAdmin = () => {
  const { role, user, loading: authLoading } = useAuth();
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    setLoading(authLoading);
  }, [authLoading]);
  
  const isAdmin = role === 'admin';
  
  const hasPermission = (permission: string) => isAdmin;
  
  const hasAnyPermission = (permissions: string[]) => {
    return isAdmin;
  };
  
  return {
    isAdmin,
    hasPermission,
    hasAnyPermission,
    adminUser: isAdmin ? user : null,
    loading,
  };
};
