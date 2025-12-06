/**
 * REPZ Feature Flag Provider
 * React context provider for feature flag management
 */

import React, { createContext, useEffect, useState, ReactNode } from 'react';
import { FeatureFlagManager } from '../core/FeatureFlagManager';
import type { FeatureFlagConfig, UserContext } from '../types';

export interface FeatureFlagContextValue {
  manager: FeatureFlagManager;
  userContext: UserContext;
  isReady: boolean;
}

export interface FeatureFlagProviderProps {
  /** Child components */
  children: ReactNode;
  
  /** Feature flag configuration */
  config: Partial<FeatureFlagConfig>;
  
  /** User context for evaluation */
  userContext: UserContext;
  
  /** Loading component while initializing */
  fallback?: ReactNode;
}

export const FeatureFlagContext = createContext<FeatureFlagContextValue | null>(null);

/**
 * Feature Flag Provider Component
 * 
 * Provides feature flag functionality to the React component tree
 * 
 * @example
 * ```tsx
 * function App() {
 *   const userContext = {
 *     id: user.id,
 *     role: user.role,
 *     tier: user.tier,
 *     attributes: {
 *       createdAt: user.createdAt,
 *       emailDomain: user.email.split('@')[1],
 *       subscriptionStatus: user.subscriptionStatus,
 *       custom: {}
 *     }
 *   };
 *   
 *   return (
 *     <FeatureFlagProvider 
 *       config={{ environment: 'production' }}
 *       userContext={userContext}
 *       fallback={<LoadingSpinner />}
 *     >
 *       <Dashboard />
 *     </FeatureFlagProvider>
 *   );
 * }
 * ```
 */
export function FeatureFlagProvider({
  children,
  config,
  userContext,
  fallback = null
}: FeatureFlagProviderProps): JSX.Element {
  const [manager, setManager] = useState<FeatureFlagManager | null>(null);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    const flagManager = new FeatureFlagManager(config);
    setManager(flagManager);
    
    // Wait a brief moment for initial flag loading
    setTimeout(() => {
      setIsReady(true);
    }, 100);

    return () => {
      flagManager.destroy();
    };
  }, []);

  // Update user context when it changes
  useEffect(() => {
    if (manager && isReady) {
      // Force cache clear when user context changes significantly
      manager.refreshFlags();
    }
  }, [userContext.id, userContext.tier, userContext.role, manager, isReady]);

  if (!manager || !isReady) {
    return <>{fallback}</>;
  }

  const contextValue: FeatureFlagContextValue = {
    manager,
    userContext,
    isReady
  };

  return (
    <FeatureFlagContext.Provider value={contextValue}>
      {children}
    </FeatureFlagContext.Provider>
  );
}