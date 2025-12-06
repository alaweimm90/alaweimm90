// Theme Provider - Industry Standard Context Pattern
// Follows Shopify/Airbnb practices for theme injection

import React, { createContext, useContext } from 'react';
import type { UserRole, TierType } from '@/types/business';
import { coachTheme, baseTheme, type ThemeTokens } from './tokens';
import { getClientTheme, type ClientTheme } from './client';

interface ThemeContextValue {
  theme: ThemeTokens | ClientTheme;
  role: UserRole;
  tier?: TierType;
}

const ThemeContext = createContext<ThemeContextValue>({
  theme: baseTheme,
  role: 'client',
});

interface ThemeProviderProps {
  children: React.ReactNode;
  role: UserRole;
  tier?: TierType;
}

export function ThemeProvider({ children, role, tier }: ThemeProviderProps) {
  // Theme Selection Logic (Industry Standard)
  const theme = React.useMemo(() => {
    if (role === 'coach') {
      // Coach always gets canonical theme
      return coachTheme;
    }
    
    if (role === 'client' && tier) {
      // Client gets base + tier overrides
      return getClientTheme(tier);
    }
    
    // Fallback to base theme
    return baseTheme;
  }, [role, tier]);

  return (
    <ThemeContext.Provider value={{ theme, role, tier }}>
      {children}
    </ThemeContext.Provider>
  );
}

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider');
  }
  return context;
};