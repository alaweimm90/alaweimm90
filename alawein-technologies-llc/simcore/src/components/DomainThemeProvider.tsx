/**
 * Domain Theme Provider Component
 * Automatically switches themes based on scientific modules
 */

import React, { createContext, useContext, useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { domainThemeController, PhysicsDomain } from '@/lib/domain-theme-controller';
import { physicsModules } from '@/data/modules';

interface DomainThemeContextType {
  currentDomain: PhysicsDomain;
  setDomain: (domain: PhysicsDomain) => void;
  autoDetectFromRoute: () => void;
}

const DomainThemeContext = createContext<DomainThemeContextType | undefined>(undefined);

interface DomainThemeProviderProps {
  children: React.ReactNode;
  defaultDomain?: PhysicsDomain;
}

export function DomainThemeProvider({ 
  children, 
  defaultDomain = 'quantum' 
}: DomainThemeProviderProps) {
  const [currentDomain, setCurrentDomain] = useState<PhysicsDomain>(defaultDomain);
  const location = useLocation();

  // Initialize theme controller
  useEffect(() => {
    domainThemeController.initialize(defaultDomain);
    
    // Subscribe to domain changes
    const unsubscribe = domainThemeController.subscribe(setCurrentDomain);
    return unsubscribe;
  }, [defaultDomain]);

  // Auto-detect domain from current route
  const autoDetectFromRoute = React.useCallback(() => {
    const currentPath = location.pathname;
    
    // Find module by route
    const currentModule = physicsModules.find(module => 
      module.route === currentPath
    );
    
    if (currentModule) {
      const detectedDomain = domainThemeController.autoDetectDomain(currentModule.category);
      domainThemeController.setDomain(detectedDomain);
    }
  }, [location.pathname]);

  // Auto-detect domain when route changes
  useEffect(() => {
    autoDetectFromRoute();
  }, [autoDetectFromRoute]);

  const setDomain = React.useCallback((domain: PhysicsDomain) => {
    domainThemeController.setDomain(domain);
  }, []);

  const contextValue: DomainThemeContextType = {
    currentDomain,
    setDomain,
    autoDetectFromRoute
  };

  return (
    <DomainThemeContext.Provider value={contextValue}>
      <div data-domain={currentDomain} className="min-h-screen transition-colors duration-500">
        {children}
      </div>
    </DomainThemeContext.Provider>
  );
}

export function useDomainTheme() {
  const context = useContext(DomainThemeContext);
  if (context === undefined) {
    throw new Error('useDomainTheme must be used within a DomainThemeProvider');
  }
  return context;
}

// Hook for getting domain-specific styles
export function useDomainStyles(domain?: PhysicsDomain) {
  const { currentDomain } = useDomainTheme();
  const targetDomain = domain || currentDomain;
  
  return React.useMemo(() => {
    return domainThemeController.generateDomainClasses(targetDomain);
  }, [targetDomain]);
}