// Centralized Design Tokens for Pricing Section
export const pricingTheme = {
  // Base Colors
  backgrounds: {
    primary: '#121212',
    secondary: '#1E1E1E', 
    surface: '#2C2C2C',
    surfaceLight: '#3B3B3B',
    glass: 'rgba(44, 44, 44, 0.8)',
  },
  
  // Tier-Specific Colors
  tiers: {
    core: {
      primary: '#3B82F6',    // Blue-500
      light: '#60A5FA',      // Blue-400
      dark: '#2563EB',       // Blue-600
      bg: 'rgba(59, 130, 246, 0.1)',
      border: 'rgba(59, 130, 246, 0.3)',
    },
    adaptive: {
      primary: '#F97316',    // Orange-500
      light: '#FB923C',      // Orange-400
      dark: '#EA580C',       // Orange-600
      bg: 'rgba(249, 115, 22, 0.1)',
      border: 'rgba(249, 115, 22, 0.3)',
    },
    performance: {
      primary: '#8B5CF6',    // Purple-500
      light: '#A78BFA',      // Purple-400
      dark: '#7C3AED',       // Purple-600
      bg: 'rgba(139, 92, 246, 0.1)',
      border: 'rgba(139, 92, 246, 0.3)',
    },
    longevity: {
      primary: '#EAB308',    // Yellow-500 (Gold)
      light: '#FDE047',      // Yellow-300
      dark: '#CA8A04',       // Yellow-600
      bg: 'rgba(234, 179, 8, 0.1)',
      border: 'rgba(234, 179, 8, 0.3)',
    }
  },
  
  // Status Colors
  status: {
    success: '#10B981',     // Emerald-500
    error: '#EF4444',       // Red-500
    warning: '#F59E0B',     // Amber-500
    info: '#3B82F6',        // Blue-500
  },
  
  // Text Colors
  text: {
    primary: '#FFFFFF',
    secondary: '#E5E7EB',   // Gray-200
    muted: '#9CA3AF',       // Gray-400
    disabled: '#6B7280',    // Gray-500
  },
  
  // Shadows & Effects
  shadows: {
    sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
    xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1)',
    glow: '0 0 20px rgba(139, 92, 246, 0.3)',
  },
  
  // Spacing
  spacing: {
    xs: '0.25rem',   // 4px
    sm: '0.5rem',    // 8px
    md: '1rem',      // 16px
    lg: '1.5rem',    // 24px
    xl: '2rem',      // 32px
    xxl: '3rem',     // 48px
  },
  
  // Border Radius
  radius: {
    sm: '0.25rem',   // 4px
    md: '0.5rem',    // 8px
    lg: '0.75rem',   // 12px
    xl: '1rem',      // 16px
    full: '9999px',
  },
  
  // Transitions
  transitions: {
    fast: '150ms ease-in-out',
    normal: '300ms ease-in-out',
    slow: '500ms ease-in-out',
  }
} as const;

// Helper functions for consistent styling
export const getTierColor = (tierId: string, variant: 'primary' | 'light' | 'dark' | 'bg' | 'border' = 'primary') => {
  const tierMap: Record<string, keyof typeof pricingTheme.tiers> = {
    'core': 'core',
    'adaptive': 'adaptive', 
    'performance': 'performance',
    'longevity': 'longevity'
  };
  
  const tier = tierMap[tierId] || 'core';
  return pricingTheme.tiers[tier][variant];
};

export const getStatusColor = (status: 'included' | 'excluded' | 'highlight') => {
  switch(status) {
    case 'included': return pricingTheme.status.success;
    case 'excluded': return pricingTheme.status.error;
    case 'highlight': return pricingTheme.tiers.longevity.primary;
    default: return pricingTheme.text.muted;
  }
};