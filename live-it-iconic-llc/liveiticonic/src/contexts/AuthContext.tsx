import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { authService, User } from '@/services/authService';

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  signUp: (email: string, password: string, name?: string) => Promise<void>;
  signIn: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check for existing session on mount
    const checkAuth = async () => {
      try {
        const currentUser = await authService.getCurrentUser();
        setUser(currentUser);
      } catch (error) {
        console.error('Auth check failed:', error);
        setUser(null);
      } finally {
        setIsLoading(false);
      }
    };

    checkAuth();
  }, []);

  const signUp = async (email: string, password: string, name?: string) => {
    const { user } = await authService.signUp({ email, password, name });
    setUser(user);
  };

  const signIn = async (email: string, password: string) => {
    const { user } = await authService.signIn({ email, password });
    setUser(user);
  };

  const signOut = async () => {
    await authService.signOut();
    setUser(null);
  };

  const value: AuthContextType = {
    user,
    isLoading,
    signUp,
    signIn,
    signOut,
    isAuthenticated: !!user,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};
