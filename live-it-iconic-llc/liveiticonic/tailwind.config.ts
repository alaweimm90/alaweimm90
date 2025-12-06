import type { Config } from 'tailwindcss';
import tailwindAnimate from 'tailwindcss-animate';
import { tokens as legacyTokens } from './src/theme/tokens';
import {
  colors,
  typography,
  spacing,
  sizes,
  shadows,
  borders,
  motion,
} from './src/design-tokens';

export default {
  darkMode: ['class'],
  content: [
    './pages/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './app/**/*.{ts,tsx}',
    './src/**/*.{ts,tsx}',
  ],
  prefix: '',
  theme: {
    container: {
      center: true,
      padding: '2rem',
      screens: {
        '2xl': '1400px',
      },
    },
    extend: {
      // =====================================================================
      // TYPOGRAPHY - Using new modular design tokens
      // =====================================================================
      fontFamily: {
        display: typography.fontFamilies.display,
        body: typography.fontFamilies.body,
        mono: typography.fontFamilies.mono,
        // Legacy compatibility
        ui: legacyTokens.typography.fonts.ui,
      },
      fontSize: typography.fontSizes as Record<string, string>,
      lineHeight: Object.entries(typography.lineHeights).reduce(
        (acc, [key, value]) => ({ ...acc, [key]: value.toString() }),
        {}
      ),
      letterSpacing: typography.letterSpacing as Record<string, string>,
      fontWeight: Object.entries(typography.fontWeights).reduce(
        (acc, [key, value]) => ({ ...acc, [key]: value.toString() }),
        {}
      ),

      // =====================================================================
      // COLORS - Using new modular design tokens
      // =====================================================================
      colors: {
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))',
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary))',
          foreground: 'hsl(var(--secondary-foreground))',
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))',
        },
        accent: {
          DEFAULT: 'hsl(var(--accent))',
          foreground: 'hsl(var(--accent-foreground))',
        },
        popover: {
          DEFAULT: 'hsl(var(--popover))',
          foreground: 'hsl(var(--popover-foreground))',
        },
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))',
        },

        /* ================================================================
           LIVE IT ICONIC BRAND COLORS - NEW MODULAR DESIGN TOKENS
           ================================================================ */
        lii: {
          gold: 'hsl(var(--lii-gold))',
          'gold-press': 'hsl(var(--lii-gold-press))',
          charcoal: 'hsl(var(--lii-charcoal))',
          cloud: 'hsl(var(--lii-cloud))',
          bg: 'hsl(var(--lii-bg))',
          ash: 'hsl(var(--lii-ash))',
          ink: 'hsl(var(--lii-ink))',
          graphite: 'hsl(var(--lii-graphite))',
          'signal-red': 'hsl(var(--lii-signal-red))',
          black: 'hsl(var(--lii-black))',
          champagne: 'hsl(var(--lii-champagne))',
          bronze: 'hsl(var(--lii-bronze))',

          success: colors.success,
          warning: colors.warning,
          error: colors.error,
          info: colors.info,
        },
      },

      // =====================================================================
      // SHADOWS - Using new modular design tokens
      // =====================================================================
      boxShadow: {
        ...(shadows as Record<string, string>),
        // Legacy compatibility
        base: legacyTokens.elevation.base,
        hover: legacyTokens.elevation.hover,
        active: legacyTokens.elevation.active,
        elevated: 'var(--shadow-elevated)',
        floating: 'var(--shadow-floating)',
      },

      backgroundImage: {
        'lii-card': 'var(--lii-card-gradient)',
      },

      // =====================================================================
      // BORDER RADIUS - Using new modular design tokens
      // =====================================================================
      borderRadius: {
        ...(borders.radii as Record<string, string>),
        // Legacy compatibility
        card: `${legacyTokens.radius.card}px`,
        control: `${legacyTokens.radius.control}px`,
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 2px)',
        sm: 'calc(var(--radius) - 4px)',
      },

      // =====================================================================
      // SPACING - Using new modular design tokens
      // =====================================================================
      spacing: spacing as Record<string | number, string>,
      gap: spacing as Record<string | number, string>,
      padding: spacing as Record<string | number, string>,
      margin: spacing as Record<string | number, string>,
      inset: spacing as Record<string | number, string>,
      width: sizes as Record<string, string>,
      height: sizes as Record<string, string>,
      maxWidth: sizes as Record<string, string>,

      // =====================================================================
      // MOTION - Using new modular design tokens
      // =====================================================================
      transitionDuration: motion.durations as Record<string, string>,
      transitionTimingFunction: motion.easings as Record<string, string>,
      keyframes: {
        'accordion-down': {
          from: {
            height: '0',
          },
          to: {
            height: 'var(--radix-accordion-content-height)',
          },
        },
        'accordion-up': {
          from: {
            height: 'var(--radix-accordion-content-height)',
          },
          to: {
            height: '0',
          },
        },
        'fade-in-up': {
          '0%': {
            opacity: '0',
            transform: 'translateY(20px)',
          },
          '100%': {
            opacity: '1',
            transform: 'translateY(0)',
          },
        },
        'scale-in': {
          '0%': {
            opacity: '0',
            transform: 'scale(0.95)',
          },
          '100%': {
            opacity: '1',
            transform: 'scale(1)',
          },
        },
        'slide-in-right': {
          '0%': {
            opacity: '0',
            transform: 'translateX(30px)',
          },
          '100%': {
            opacity: '1',
            transform: 'translateX(0)',
          },
        },
      },
      animation: {
        'accordion-down': 'accordion-down 0.2s ease-out',
        'accordion-up': 'accordion-up 0.2s ease-out',
        'fade-in-up': 'fade-in-up 0.6s ease-out',
        'scale-in': 'scale-in 0.4s ease-out',
        'slide-in-right': 'slide-in-right 0.5s ease-out',
      },
    },
  },
  plugins: [tailwindAnimate],
} satisfies Config;
