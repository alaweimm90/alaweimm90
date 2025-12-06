import React, { createContext, useContext, ReactNode } from 'react';
import { useAccessibility, useAriaLive, AccessibilityPreferences } from '@/hooks/useAccessibility';

interface AccessibilityContextType extends AccessibilityPreferences {
  announce: (message: string, priority?: 'polite' | 'assertive') => void;
  announcement: string;
}

const AccessibilityContext = createContext<AccessibilityContextType | undefined>(undefined);

export const useAccessibilityContext = () => {
  const context = useContext(AccessibilityContext);
  if (!context) {
    throw new Error('useAccessibilityContext must be used within AccessibilityProvider');
  }
  return context;
};

interface AccessibilityProviderProps {
  children: ReactNode;
}

export const AccessibilityProvider: React.FC<AccessibilityProviderProps> = ({ children }) => {
  const preferences = useAccessibility();
  const { announcement, announce } = useAriaLive();

  const value = {
    ...preferences,
    announce,
    announcement
  };

  return (
    <AccessibilityContext.Provider value={value}>
      {children}
      
      {/* ARIA Live Region for Screen Reader Announcements */}
      <div
        role="status"
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
        id="accessibility-announcements"
      >
        {announcement}
      </div>
      
      {/* Skip Navigation Menu */}
      <nav 
        className="sr-only focus-within:not-sr-only focus-within:absolute focus-within:top-4 focus-within:left-4 focus-within:z-50"
        aria-label="Skip navigation"
      >
        <ul className="flex flex-col gap-2">
          <li>
            <a 
              href="#main-content" 
              className="bg-primary text-primary-foreground px-4 py-2 rounded-md font-medium focus:outline-none focus:ring-2 focus:ring-primary-foreground"
            >
              Skip to main content
            </a>
          </li>
          <li>
            <a 
              href="#circuit-builder" 
              className="bg-primary text-primary-foreground px-4 py-2 rounded-md font-medium focus:outline-none focus:ring-2 focus:ring-primary-foreground"
            >
              Skip to circuit builder
            </a>
          </li>
          <li>
            <a 
              href="#training-dashboard" 
              className="bg-primary text-primary-foreground px-4 py-2 rounded-md font-medium focus:outline-none focus:ring-2 focus:ring-primary-foreground"
            >
              Skip to training dashboard
            </a>
          </li>
        </ul>
      </nav>
    </AccessibilityContext.Provider>
  );
};