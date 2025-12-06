// Enhanced design system configuration with centralized enforcement
// This file serves as the SINGLE SOURCE OF TRUTH for all design decisions

// Import centralized business types
import type { TierType } from '@/types/business';

// CRITICAL: All design tokens must be defined here
// Components MUST reference these tokens, never hardcode values
export const DESIGN_TOKENS = {
  // Color system - HSL values for perfect theme support
  colors: {
    // Brand colors (NEVER CHANGE without approval)
    brand: {
      repzOrange: 'hsl(14, 87%, 54%)', // #F15B23 - Primary brand
      professionalBlack: 'hsl(0, 0%, 0%)', // #000000 - Foundation
    },
    
    // Tier colors (CRITICAL - tied to business model)
    tier: {
      core: 'hsl(218, 91%, 60%)', // #3B82F6 - Trust/Foundation
      adaptive: 'hsl(14, 87%, 54%)', // #F15B23 - REPZ Brand  
      performance: 'hsl(262, 83%, 58%)', // #A855F7 - Sophistication
      longevity: 'hsl(45, 93%, 47%)', // #EAB308 - Luxury
    },
    
    // Semantic colors
    semantic: {
      primary: 'var(--primary)',
      secondary: 'var(--secondary)', 
      accent: 'var(--accent)',
      muted: 'var(--muted)',
      destructive: 'var(--destructive)',
      success: 'var(--success)',
      warning: 'var(--warning)',
      info: 'var(--info)',
    },
    
    // Surface colors for elegant UI
    surface: {
      elegant: 'var(--surface-elegant)',
      velvet: 'var(--surface-velvet)',
      glass: 'var(--surface-glass)',
      premium: 'var(--surface-premium)',
    }
  },
  
  // Typography system
  typography: {
    fontFamily: {
      sans: ['Inter', 'system-ui', 'sans-serif'],
      heading: ['Montserrat', 'Inter', 'system-ui', 'sans-serif'],
      mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
    },
    fontSize: {
      xs: '0.75rem',
      sm: '0.875rem',
      base: '1rem',
      lg: '1.125rem',
      xl: '1.25rem',
      '2xl': '1.5rem',
      '3xl': '1.875rem',
      '4xl': '2.25rem',
      '5xl': '3rem',
    },
    fontWeight: {
      light: '300',
      normal: '400',
      medium: '500',
      semibold: '600',
      bold: '700',
      extrabold: '800',
    },
    lineHeight: {
      tight: '1.25',
      normal: '1.5',
      relaxed: '1.75',
    }
  },
  
  // Spacing system
  spacing: {
    xs: '0.25rem',
    sm: '0.5rem',
    md: '1rem',
    lg: '1.5rem',
    xl: '2rem',
    '2xl': '3rem',
    '3xl': '4rem',
    '4xl': '6rem',
  },
  
  // Border radius system
  borderRadius: {
    none: '0',
    sm: '0.125rem',
    md: '0.375rem',
    lg: '0.5rem',
    xl: '0.75rem',
    '2xl': '1rem',
    full: '9999px',
  },
  
  // Shadow system
  shadows: {
    sm: '0 1px 2px 0 rgb(0 0 0 / 0.05)',
    md: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
    lg: '0 10px 15px -3px rgb(0 0 0 / 0.1)',
    xl: '0 20px 25px -5px rgb(0 0 0 / 0.1)',
    elegant: '0 10px 30px -10px hsl(var(--primary) / 0.3)',
    glow: '0 0 40px hsl(var(--primary) / 0.4)',
  },
  
  // Animation system
  animation: {
    duration: {
      fast: '150ms',
      normal: '300ms',
      slow: '500ms',
    },
    easing: {
      smooth: 'cubic-bezier(0.4, 0, 0.2, 1)',
      bounce: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
    }
  }
} as const;

// Component variants - SINGLE SOURCE for all component styling
export const COMPONENT_VARIANTS = {
  // Button variants
  button: {
    base: 'inline-flex items-center justify-center font-medium transition-all focus:outline-none focus:ring-2 focus:ring-offset-2',
    variants: {
      primary: 'bg-primary text-primary-foreground hover:bg-primary/90 focus:ring-primary',
      secondary: 'bg-secondary text-secondary-foreground hover:bg-secondary/80 focus:ring-secondary',
      outline: 'border border-input bg-background hover:bg-accent hover:text-accent-foreground focus:ring-primary',
      ghost: 'hover:bg-accent hover:text-accent-foreground focus:ring-primary',
      destructive: 'bg-destructive text-destructive-foreground hover:bg-destructive/90 focus:ring-destructive',
      
      // Tier-specific variants
      tierCore: 'bg-tier-core text-white hover:bg-tier-core/90 focus:ring-tier-core',
      tierAdaptive: 'bg-tier-adaptive text-white hover:bg-tier-adaptive/90 focus:ring-tier-adaptive',
      tierPerformance: 'bg-tier-performance text-white hover:bg-tier-performance/90 focus:ring-tier-performance',
      tierLongevity: 'bg-tier-longevity text-white hover:bg-tier-longevity/90 focus:ring-tier-longevity',
    },
    sizes: {
      sm: 'h-8 px-3 text-xs rounded-md',
      md: 'h-10 px-4 text-sm rounded-md',
      lg: 'h-12 px-6 text-base rounded-lg',
      xl: 'h-14 px-8 text-lg rounded-lg',
    }
  },
  
  // Card variants
  card: {
    base: 'rounded-lg border bg-card text-card-foreground shadow-sm transition-all',
    variants: {
      default: 'border-border',
      elevated: 'shadow-lg hover:shadow-xl',
      glass: 'bg-surface-glass border-white/20 backdrop-blur-sm',
      tier: 'border-2 hover:shadow-elegant',
    },
    padding: {
      none: '',
      sm: 'p-4',
      md: 'p-6',
      lg: 'p-8',
    }
  },
  
  // Modal variants
  modal: {
    overlay: 'fixed inset-0 z-50 bg-background/80 backdrop-blur-sm',
    content: 'fixed left-[50%] top-[50%] z-50 translate-x-[-50%] translate-y-[-50%] bg-background border rounded-lg shadow-lg',
    sizes: {
      sm: 'w-[90vw] max-w-md',
      md: 'w-[90vw] max-w-lg', 
      lg: 'w-[90vw] max-w-2xl',
      xl: 'w-[90vw] max-w-4xl',
    }
  }
} as const;

// Tier-specific styling (CRITICAL - business model integration)
export const TIER_STYLES: Record<TierType, {
  gradient: string;
  accent: string;
  button: keyof typeof COMPONENT_VARIANTS.button.variants;
  border: string;
}> = {
  core: {
    gradient: 'from-tier-core/20 to-tier-core/5',
    accent: 'tier-core',
    button: 'tierCore',
    border: 'border-tier-core/30',
  },
  adaptive: {
    gradient: 'from-tier-adaptive/20 to-tier-adaptive/5',
    accent: 'tier-adaptive', 
    button: 'tierAdaptive',
    border: 'border-tier-adaptive/30',
  },
  performance: {
    gradient: 'from-tier-performance/20 to-tier-performance/5',
    accent: 'tier-performance',
    button: 'tierPerformance', 
    border: 'border-tier-performance/30',
  },
  longevity: {
    gradient: 'from-tier-longevity/20 to-tier-longevity/5',
    accent: 'tier-longevity',
    button: 'tierLongevity',
    border: 'border-tier-longevity/30',
  },
} as const;

// Accessibility standards (WCAG compliance)
export const ACCESSIBILITY = {
  minContrastRatio: 4.5,
  focusRing: 'focus:ring-2 focus:ring-offset-2 focus:ring-primary',
  minClickableSize: '44px',
  reducedMotion: '@media (prefers-reduced-motion: reduce)',
} as const;

// Helper functions for consistent usage
export const getTierStyles = (tier: TierType) => {
  return TIER_STYLES[tier];
};

export const getComponentVariant = (component: keyof typeof COMPONENT_VARIANTS, variant: string, size?: string) => {
  const comp = COMPONENT_VARIANTS[component];
  
  if (!comp || typeof comp !== 'object') return '';
  
  const baseClass = 'base' in comp ? comp.base : '';
  const variantClass = 'variants' in comp && comp.variants ? comp.variants[variant as keyof typeof comp.variants] || '' : '';
  const sizeClass = size && 'sizes' in comp && comp.sizes ? comp.sizes[size as keyof typeof comp.sizes] || '' : '';
  
  return `${baseClass} ${variantClass} ${sizeClass}`.trim();
};

export const validateDesignToken = (category: string, token: string): boolean => {
  return DESIGN_TOKENS[category as keyof typeof DESIGN_TOKENS]?.[token as keyof (typeof DESIGN_TOKENS)[keyof typeof DESIGN_TOKENS]] !== undefined;
};

// Design system enforcement rules
export const DESIGN_SYSTEM_RULES = {
  // Colors that should never be used directly
  bannedClasses: [
    'bg-blue-500', 'bg-red-500', 'bg-green-500', 'bg-yellow-500',
    'text-blue-500', 'text-red-500', 'text-green-500', 'text-yellow-500',
    'border-blue-500', 'border-red-500', 'border-green-500', 'border-yellow-500'
  ],
  
  // Required prefixes for custom styles
  allowedPrefixes: [
    'bg-primary', 'bg-secondary', 'bg-accent', 'bg-tier-', 'bg-surface-',
    'text-primary', 'text-secondary', 'text-accent', 'text-tier-',
    'border-primary', 'border-secondary', 'border-accent', 'border-tier-'
  ],
  
  // Components that must use variants
  requireVariants: ['Button', 'Card', 'Modal', 'Dialog'],
} as const;

export default DESIGN_TOKENS;