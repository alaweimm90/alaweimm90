/**
 * CENTRALIZED DESIGN SYSTEM CONFIGURATION
 * Single source of truth for all UI components, themes, and styling
 */

export const DESIGN_TOKENS = {
  // Typography Scale
  typography: {
    fontFamily: {
      primary: 'var(--font-montserrat)',
      secondary: 'var(--font-inter)',
      mono: 'var(--font-mono)'
    },
    fontSize: {
      xs: '0.75rem',
      sm: '0.875rem', 
      base: '1rem',
      lg: '1.125rem',
      xl: '1.25rem',
      '2xl': '1.5rem',
      '3xl': '1.875rem',
      '4xl': '2.25rem'
    },
    fontWeight: {
      normal: '400',
      medium: '500',
      semibold: '600',
      bold: '700',
      extrabold: '800'
    },
    lineHeight: {
      tight: '1.25',
      normal: '1.5',
      relaxed: '1.75'
    }
  },

  // Color System - All HSL values for consistency
  colors: {
    primary: {
      50: 'hsl(31, 100%, 97%)',
      100: 'hsl(31, 100%, 93%)',
      500: 'hsl(31, 91%, 60%)', // repz-orange
      600: 'hsl(31, 91%, 55%)',
      700: 'hsl(31, 91%, 50%)',
      900: 'hsl(31, 91%, 30%)'
    },
    success: {
      400: 'hsl(142, 71%, 45%)', // emerald-400
      500: 'hsl(142, 71%, 40%)',
      600: 'hsl(142, 71%, 35%)'
    },
    warning: {
      400: 'hsl(45, 93%, 58%)', // amber-400
      500: 'hsl(45, 93%, 53%)',
      600: 'hsl(45, 93%, 48%)'
    },
    error: {
      400: 'hsl(0, 84%, 60%)', // red-400
      500: 'hsl(0, 84%, 55%)',
      600: 'hsl(0, 84%, 50%)'
    },
    neutral: {
      50: 'hsl(210, 40%, 98%)',
      100: 'hsl(210, 40%, 96%)',
      200: 'hsl(214, 32%, 91%)',
      300: 'hsl(213, 27%, 84%)',
      400: 'hsl(215, 20%, 65%)',
      500: 'hsl(215, 16%, 47%)',
      600: 'hsl(215, 19%, 35%)',
      700: 'hsl(215, 25%, 27%)',
      800: 'hsl(217, 33%, 17%)',
      900: 'hsl(222, 84%, 5%)'
    }
  },

  // Spacing Scale
  spacing: {
    xs: '0.25rem',
    sm: '0.5rem',
    md: '1rem',
    lg: '1.5rem',
    xl: '2rem',
    '2xl': '3rem',
    '3xl': '4rem',
    '4xl': '6rem'
  },

  // Border Radius
  borderRadius: {
    none: '0',
    sm: '0.25rem',
    md: '0.375rem',
    lg: '0.5rem',
    xl: '0.75rem',
    '2xl': '1rem',
    full: '9999px'
  },

  // Shadows
  shadows: {
    sm: '0 1px 2px 0 rgb(0 0 0 / 0.05)',
    md: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
    lg: '0 10px 15px -3px rgb(0 0 0 / 0.1)',
    xl: '0 20px 25px -5px rgb(0 0 0 / 0.1)',
    glow: '0 0 40px hsl(var(--primary) / 0.4)'
  },

  // Animation Durations
  animation: {
    fast: '150ms',
    normal: '300ms',
    slow: '500ms'
  }
} as const;

/**
 * COMPONENT VARIANTS - Centralized styling for all UI components
 */
export const COMPONENT_VARIANTS = {
  button: {
    primary: {
      base: 'bg-gradient-to-r from-primary-500 to-primary-600 text-white hover:from-primary-600 hover:to-primary-700',
      sizes: {
        sm: 'px-3 py-2 text-sm',
        md: 'px-4 py-2 text-base',
        lg: 'px-6 py-3 text-lg'
      }
    },
    secondary: {
      base: 'bg-neutral-200 text-neutral-800 hover:bg-neutral-300',
      sizes: {
        sm: 'px-3 py-2 text-sm',
        md: 'px-4 py-2 text-base', 
        lg: 'px-6 py-3 text-lg'
      }
    },
    luxury: {
      base: 'bg-gradient-to-r from-warning-400 to-warning-500 text-black hover:from-warning-500 hover:to-warning-600',
      sizes: {
        sm: 'px-3 py-2 text-sm',
        md: 'px-4 py-2 text-base',
        lg: 'px-6 py-3 text-lg'
      }
    }
  },

  card: {
    default: {
      base: 'bg-white border border-neutral-200 rounded-xl shadow-md',
      padding: 'p-6'
    },
    glass: {
      base: 'bg-white/80 backdrop-blur-xl border border-white/20 rounded-xl',
      padding: 'p-6'
    },
    tier: {
      base: 'relative overflow-hidden transition-all duration-300 border-0 backdrop-blur-xl bg-background/80',
      padding: 'p-6'
    }
  },

  modal: {
    overlay: 'fixed inset-0 bg-black/50 backdrop-blur-sm z-50',
    content: 'fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-white rounded-xl shadow-2xl max-w-md w-full mx-4 p-6 z-50'
  }
} as const;

/**
 * TIER-SPECIFIC STYLING
 */
export const TIER_STYLES = {
  core: {
    gradient: 'from-blue-500/20 via-blue-600/30 to-blue-700/20',
    accent: 'text-blue-500',
    button: 'bg-blue-500 hover:bg-blue-600'
  },
  adaptive: {
    gradient: 'from-purple-500/20 via-purple-600/30 to-purple-700/20', 
    accent: 'text-purple-500',
    button: 'bg-purple-500 hover:bg-purple-600'
  },
  performance: {
    gradient: 'from-primary-500/20 via-primary-600/30 to-primary-700/20',
    accent: 'text-primary-500',
    button: 'bg-primary-500 hover:bg-primary-600'
  },
  longevity: {
    gradient: 'from-warning-400/20 via-warning-500/30 to-warning-600/20',
    accent: 'text-warning-400',
    button: 'bg-warning-400 hover:bg-warning-500 text-black'
  }
} as const;

/**
 * ACCESSIBILITY STANDARDS
 */
export const ACCESSIBILITY = {
  minContrastRatio: 4.5,
  focusRing: 'focus:ring-2 focus:ring-primary-500 focus:ring-offset-2',
  clickableMinSize: '44px', // WCAG minimum touch target
  animationReducedMotion: '@media (prefers-reduced-motion: reduce)'
} as const;

/**
 * Helper function to get tier-specific styles
 */
export const getTierStyles = (tier: 'core' | 'adaptive' | 'performance' | 'longevity') => {
  return TIER_STYLES[tier];
};

/**
 * Helper function to validate design token usage
 */
export const validateDesignToken = (category: string, token: string): boolean => {
  return (DESIGN_TOKENS as Record<string, Record<string, unknown>>)[category]?.[token] !== undefined;
};