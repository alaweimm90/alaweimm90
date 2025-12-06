// ============================================================================
// LIVE IT ICONIC - CANONICAL DESIGN TOKENS
// Premium Lifestyle Merchandise Brand
// ============================================================================

export const tokens = {
  // ============================================================================
  // COLOR PALETTE - Luxury Brand Standards
  // ============================================================================
  colors: {
    // Primary Brand Colors
    carbonBlack: '#1a1a1a', // Primary brand color
    championshipGold: '#d4af37', // Premium accent
    platinumSilver: '#c0c0c0', // Secondary option

    // Refined palette
    gold: '#d4af37', // Championship Gold
    goldPress: '#c19d2e',
    goldHover: '#e0c050',

    // Neutral Palette
    bg: '#0B0B0C', // Primary background
    ink: '#14161A', // Secondary background
    charcoal: '#36454f', // Brand charcoal gray
    graphite: '#3A4048', // Component backgrounds
    ash: '#8C93A3', // Body text
    cloud: '#E6E9EF', // Headings and accents
    snow: '#F8F9FA', // Pure white accents

    // Semantic Colors
    signalRed: '#E03A2F', // Errors, warnings
    success: '#10B981', // Success states
    warning: '#F59E0B', // Warning states
    info: '#3B82F6', // Info states

    // Legacy Support
    champagne: '#f7e7ce', // Champagne gold
    gunmetal: '#2a3439', // Hardware finish
    luxury: '#d4af37', // Championship gold alias
  },

  // ============================================================================
  // 📐 SPACING SCALE - Luxury Proportions
  // ============================================================================
  spacing: {
    0: '0',
    1: '0.25rem', // 4px
    2: '0.5rem', // 8px
    3: '0.75rem', // 12px
    4: '1rem', // 16px
    5: '1.25rem', // 20px
    6: '1.5rem', // 24px
    8: '2rem', // 32px
    10: '2.5rem', // 40px
    12: '3rem', // 48px
    16: '4rem', // 64px
    20: '5rem', // 80px
    24: '6rem', // 96px
    32: '8rem', // 128px
  },

  // ============================================================================
  // BORDER RADIUS - Premium Design Language
  // ============================================================================
  radius: {
    none: '0',
    sm: '0.125rem', // 2px - Small accents
    md: '0.375rem', // 6px - Form elements
    lg: '0.5rem', // 8px - Cards, buttons
    xl: '0.75rem', // 12px - Primary buttons, modals
    '2xl': '1rem', // 16px - Large containers
    '3xl': '1.5rem', // 24px - Hero sections
    full: '9999px', // Fully rounded
    // Legacy aliases for backward compatibility
    card: 16, // 16px - Canonical card radius
    control: 12, // 12px - Canonical button/input radius
  },

  // ============================================================================
  // ✍️ TYPOGRAPHY - Display Serif & UI Sans
  // ============================================================================
  typography: {
    // Font Families
    fonts: {
      display: ['Playfair Display', 'serif'],
      ui: ['Inter Variable', 'sans-serif'],
      mono: ['JetBrains Mono', 'monospace'],
    },

    // Font Sizes (Mobile-First)
    sizes: {
      xs: '0.75rem', // 12px - Captions
      sm: '0.875rem', // 14px - Small text
      base: '1rem', // 16px - Body text
      lg: '1.125rem', // 18px - Large body
      xl: '1.25rem', // 20px - Small headings
      '2xl': '1.5rem', // 24px - Component headings
      '3xl': '1.875rem', // 30px - Section headings
      '4xl': '2.25rem', // 36px - Page headings
      '5xl': '3rem', // 48px - Hero headings
      '6xl': '3.75rem', // 60px - Display headings
      '7xl': '4.5rem', // 72px - Large display
      '8xl': '6rem', // 96px - Massive display
      '9xl': '8rem', // 128px - Hero display
    },

    // Line Heights
    lineHeights: {
      none: '1',
      tight: '1.25',
      snug: '1.375',
      normal: '1.5',
      relaxed: '1.625',
      loose: '2',
    },

    // Letter Spacing
    letterSpacing: {
      tighter: '-0.05em',
      tight: '-0.025em',
      normal: '0',
      wide: '0.025em',
      wider: '0.05em',
      widest: '0.1em',
    },

    // Font Weights
    weights: {
      thin: '100',
      extralight: '200',
      light: '300',
      normal: '400',
      medium: '500',
      semibold: '600',
      bold: '700',
      extrabold: '800',
      black: '900',
    },
  },

  // ============================================================================
  // 🎭 ELEVATION - Luxury Shadow System
  // ============================================================================
  elevation: {
    none: 'none',
    xs: '0 1px 2px 0 rgba(193, 160, 96, 0.05)',
    sm: '0 1px 3px 0 rgba(193, 160, 96, 0.1), 0 1px 2px 0 rgba(193, 160, 96, 0.06)',
    md: '0 4px 6px -1px rgba(193, 160, 96, 0.1), 0 2px 4px -1px rgba(193, 160, 96, 0.06)',
    lg: '0 10px 15px -3px rgba(193, 160, 96, 0.1), 0 4px 6px -2px rgba(193, 160, 96, 0.05)',
    xl: '0 20px 25px -5px rgba(193, 160, 96, 0.1), 0 10px 10px -5px rgba(193, 160, 96, 0.04)',
    '2xl': '0 25px 50px -12px rgba(193, 160, 96, 0.25)',
    // Legacy aliases for backward compatibility
    base: '0 8px 24px rgba(193, 160, 96, 0.12)',
    hover: '0 12px 32px rgba(193, 160, 96, 0.16)',
    active: '0 4px 16px rgba(193, 160, 96, 0.08)',
  },

  // ============================================================================
  // MOTION - Refined Transitions
  // ============================================================================
  motion: {
    // Duration Scale
    duration: {
      instant: '0ms',
      fast: '150ms',
      normal: '300ms',
      slow: '500ms',
      slower: '700ms',
      // Legacy aliases for backward compatibility
      enter: 280, // ms - Canonical entrance duration
      micro: 140, // ms - Canonical micro-interaction duration
    },

    // Easing Functions
    easing: {
      linear: 'linear',
      in: 'ease-in',
      out: 'ease-out',
      inOut: 'ease-in-out',
      brand: 'cubic-bezier(0.16, 1, 0.3, 1)', // Custom brand easing
      bounce: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
    },

    // Common Transitions
    transition: {
      colors: 'color 0.3s ease-out, background-color 0.3s ease-out, border-color 0.3s ease-out',
      transform: 'transform 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
      opacity: 'opacity 0.3s ease-out',
      all: 'all 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
    },

    // Legacy alias for backward compatibility
    ease: 'cubic-bezier(0.16, 1, 0.3, 1)',
  },

  // ============================================================================
  // 📱 BREAKPOINTS - Mobile-First Luxury
  // ============================================================================
  breakpoints: {
    sm: '640px',
    md: '768px',
    lg: '1024px',
    xl: '1280px',
    '2xl': '1536px',
  },

  // ============================================================================
  // 🎯 Z-INDEX SCALE - Layer Management
  // ============================================================================
  zIndex: {
    hide: -1,
    auto: 'auto',
    base: 0,
    docked: 10,
    dropdown: 1000,
    sticky: 1020,
    banner: 1030,
    overlay: 1040,
    modal: 1050,
    popover: 1060,
    skipLink: 1070,
    toast: 1080,
    tooltip: 1090,
  },

  // ============================================================================
  // 🎨 COMPONENT TOKENS - Pre-configured Styles
  // ============================================================================
  components: {
    // Button Variants
    buttons: {
      primary: {
        backgroundColor: '#C1A060',
        color: '#0B0B0C',
        borderRadius: '0.75rem',
        padding: '0.75rem 2rem',
        fontWeight: '600',
        fontSize: '1rem',
        transition: 'all 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
        '&:hover': {
          backgroundColor: '#AE8D55',
          transform: 'translateY(-1px)',
          boxShadow: '0 10px 25px -5px rgba(193, 160, 96, 0.4)',
        },
      },
      secondary: {
        backgroundColor: 'transparent',
        color: '#C1A060',
        border: '1.5px solid #C1A060',
        borderRadius: '0.75rem',
        padding: '0.75rem 2rem',
        fontWeight: '500',
        fontSize: '1rem',
        transition: 'all 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
        '&:hover': {
          backgroundColor: 'rgba(193, 160, 96, 0.1)',
          borderColor: '#AE8D55',
          color: '#AE8D55',
        },
      },
    },

    // Form Elements
    forms: {
      input: {
        backgroundColor: 'rgba(193, 160, 96, 0.05)',
        border: '1px solid rgba(193, 160, 96, 0.2)',
        borderRadius: '0.5rem',
        padding: '0.75rem 1rem',
        color: '#E6E9EF',
        fontSize: '1rem',
        transition: 'all 0.3s ease-out',
        '&:focus': {
          borderColor: '#C1A060',
          boxShadow: '0 0 0 3px rgba(193, 160, 96, 0.1)',
          outline: 'none',
        },
      },
    },

    // Cards
    cards: {
      default: {
        backgroundColor: 'rgba(193, 160, 96, 0.02)',
        border: '1px solid rgba(193, 160, 96, 0.1)',
        borderRadius: '1rem',
        padding: '2rem',
        boxShadow: '0 8px 24px rgba(193, 160, 96, 0.08)',
        transition: 'all 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
        '&:hover': {
          borderColor: 'rgba(193, 160, 96, 0.2)',
          boxShadow: '0 12px 32px rgba(193, 160, 96, 0.12)',
          transform: 'translateY(-2px)',
        },
      },
    },
  },
} as const;

// ============================================================================
// 🎯 TYPE EXPORTS - TypeScript Support
// ============================================================================
export type ColorTokens = typeof tokens.colors;
export type SpacingTokens = typeof tokens.spacing;
export type RadiusTokens = typeof tokens.radius;
export type TypographyTokens = typeof tokens.typography;
export type MotionTokens = typeof tokens.motion;
export type BreakpointTokens = typeof tokens.breakpoints;

// ============================================================================
// 🔧 UTILITY FUNCTIONS - Token Access Helpers
// ============================================================================
export const getColor = (key: keyof ColorTokens) => tokens.colors[key];
export const getSpacing = (key: keyof SpacingTokens) => tokens.spacing[key];
export const getRadius = (key: keyof RadiusTokens) => tokens.radius[key];
export const getBreakpoint = (key: keyof BreakpointTokens) => tokens.breakpoints[key];

// ============================================================================
// 📚 USAGE EXAMPLES
// ============================================================================
/*
CSS-in-JS Usage:
const buttonStyles = {
  backgroundColor: tokens.colors.gold,
  borderRadius: tokens.radius.xl,
  padding: `${tokens.spacing[4]} ${tokens.spacing[8]}`,
  fontSize: tokens.typography.sizes.base,
  transition: tokens.motion.transition.colors,
};

Tailwind Integration (tailwind.config.ts):
colors: {
  'lii-gold': tokens.colors.gold,
  'lii-bg': tokens.colors.bg,
  // ... map all colors
},
borderRadius: {
  'lii-sm': tokens.radius.sm,
  'lii-md': tokens.radius.md,
  // ... map all radii
}
*/
