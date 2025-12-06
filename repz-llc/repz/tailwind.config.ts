// =====================================================================
// REPZ COACH - ENHANCED THEME UPDATE SYSTEM
// Updated tailwind.config.ts for centralized design system
// =====================================================================

import type { Config } from "tailwindcss";
import { designTokens } from "./src/theme/tokens";

export default {
  darkMode: ["class"],
  content: [
    "./index.html",
    "./src/**/*.{ts,tsx}",
  ],
  prefix: "",
  theme: {
    container: {
      center: true,
      padding: '2rem',
      screens: {
        '2xl': '1400px'
      }
    },
    extend: {
      fontFamily: {
        'montserrat': ['Montserrat', 'sans-serif'],
        'inter': ['Inter', 'sans-serif'],
        'sans': ['Inter', 'sans-serif'],
        'heading': ['Montserrat', 'sans-serif'],
        'body': ['Inter', 'sans-serif'],
      },
      colors: {
        // Semantic shadcn colors
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))'
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary))',
          foreground: 'hsl(var(--secondary-foreground))'
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))'
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))'
        },
        accent: {
          DEFAULT: 'hsl(var(--accent))',
          foreground: 'hsl(var(--accent-foreground))'
        },
        popover: {
          DEFAULT: 'hsl(var(--popover))',
          foreground: 'hsl(var(--popover-foreground))'
        },
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))'
        },
        
        /* ===============================================
           REPZ UNIFIED COLOR SYSTEM
           =============================================== */
        
        // BRAND FOUNDATION
        'repz-primary': 'hsl(var(--repz-primary))',
        'repz-black': 'hsl(var(--repz-black))',
        'repz-white': 'hsl(var(--repz-white))',
        
        // TIER SYSTEM
        'tier-core': {
          DEFAULT: 'hsl(var(--tier-core))',
          light: 'hsl(var(--tier-core-light))',
          dark: 'hsl(var(--tier-core-dark))',
        },
        'tier-adaptive': {
          DEFAULT: 'hsl(var(--tier-adaptive))',
          light: 'hsl(var(--tier-adaptive-light))',
          dark: 'hsl(var(--tier-adaptive-dark))',
        },
        'tier-performance': {
          DEFAULT: 'hsl(var(--tier-performance))',
          light: 'hsl(var(--tier-performance-light))',
          dark: 'hsl(var(--tier-performance-dark))',
        },
        'tier-longevity': {
          DEFAULT: 'hsl(var(--tier-longevity))',
          light: 'hsl(var(--tier-longevity-light))',
          dark: 'hsl(var(--tier-longevity-dark))',
        },
        
        // SEMANTIC COLORS
        'success': 'hsl(var(--success))',
        'warning': 'hsl(var(--warning))',
        'error': 'hsl(var(--error))',
        'info': 'hsl(var(--info))',
        
        // SURFACE SYSTEM
        'surface-base': 'hsl(var(--surface-base))',
        'surface-elevated': 'hsl(var(--surface-elevated))',
        'surface-overlay': 'hsl(var(--surface-overlay))',
        'surface-glass': 'hsla(var(--surface-glass))',
        'surface-frost': 'hsla(var(--surface-frost))',
        
        // Legacy aliases for backward compatibility
        'repz-orange': 'hsl(var(--repz-primary))',
        'amber-elegant': 'hsl(var(--amber-elegant))',
        'gold-luxe': 'hsl(var(--gold-luxe))',
        'blue-steel': 'hsl(var(--blue-steel))',
        'charcoal-premium': 'hsl(var(--charcoal-premium))',
        'graphite-soft': 'hsl(var(--graphite-soft))',
      },
      
      spacing: {
        'xs': 'var(--spacing-xs)',
        'sm': 'var(--spacing-sm)',
        'md': 'var(--spacing-md)',
        'lg': 'var(--spacing-lg)',
        'xl': 'var(--spacing-xl)',
        '2xl': 'var(--spacing-2xl)',
        '3xl': 'var(--spacing-3xl)',
        '4xl': 'var(--spacing-4xl)',
        '5xl': 'var(--spacing-5xl)',
      },
      
      borderRadius: {
        'xs': 'var(--radius-xs)',
        'sm': 'var(--radius-sm)',
        'md': 'var(--radius-md)',
        'lg': 'var(--radius-lg)',
        'xl': 'var(--radius-xl)',
        '2xl': 'var(--radius-2xl)',
        '3xl': 'var(--radius-3xl)',
      },
      
      boxShadow: {
        'subtle': 'var(--shadow-subtle)',
        'moderate': 'var(--shadow-moderate)',
        'strong': 'var(--shadow-strong)',
        'elegant': 'var(--shadow-elegant)',
        'glow-orange': 'var(--shadow-glow-orange)',
        'glow-gold': 'var(--shadow-glow-gold)',
        'glow-blue': 'var(--shadow-glow-blue)',
        'glow-purple': 'var(--shadow-glow-purple)',
      },
      
      transitionDuration: {
        'instant': 'var(--transition-instant)',
        'quick': 'var(--transition-quick)',
        'smooth': 'var(--transition-smooth)',
        'elegant': 'var(--transition-elegant)',
        'luxurious': 'var(--transition-luxurious)',
      },
      
      transitionTimingFunction: {
        'premium': 'var(--easing-premium)',
        'elegant': 'var(--easing-elegant)',
        'smooth-curve': 'var(--easing-smooth)',
      },
      
      keyframes: {
        // All existing keyframes preserved
        "accordion-down": {
          from: { height: "0", opacity: "0" },
          to: { height: "var(--radix-accordion-content-height)", opacity: "1" }
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)", opacity: "1" },
          to: { height: "0", opacity: "0" }
        },
        "fade-in": {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" }
        },
        "fade-out": {
          "0%": { opacity: "1", transform: "translateY(0)" },
          "100%": { opacity: "0", transform: "translateY(10px)" }
        },
        "scale-in": {
          "0%": { transform: "scale(0.95)", opacity: "0" },
          "100%": { transform: "scale(1)", opacity: "1" }
        },
        "scale-out": {
          from: { transform: "scale(1)", opacity: "1" },
          to: { transform: "scale(0.95)", opacity: "0" }
        },
        "slide-in-right": {
          "0%": { transform: "translateX(100%)" },
          "100%": { transform: "translateX(0)" }
        },
        "slide-out-right": {
          "0%": { transform: "translateX(0)" },
          "100%": { transform: "translateX(100%)" }
        },
        "elegant-float": {
          "0%, 100%": { transform: "translateY(0px) rotate(0deg)" },
          "25%": { transform: "translateY(-8px) rotate(1deg)" },
          "50%": { transform: "translateY(-12px) rotate(0deg)" },
          "75%": { transform: "translateY(-6px) rotate(-1deg)" }
        },
        "premium-glow": {
          "0%, 100%": { boxShadow: "var(--shadow-glow-orange)" },
          "25%": { boxShadow: "var(--shadow-glow-gold)" },
          "50%": { boxShadow: "var(--shadow-glow-blue)" },
          "75%": { boxShadow: "var(--shadow-glow-purple)" }
        },
      },
      
      animation: {
        // Basic animations
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        "fade-in": "fade-in 0.3s ease-out",
        "fade-out": "fade-out 0.3s ease-out",
        "scale-in": "scale-in 0.2s ease-out",
        "scale-out": "scale-out 0.2s ease-out",
        "slide-in-right": "slide-in-right 0.3s ease-out",
        "slide-out-right": "slide-out-right 0.3s ease-out",
        "enter": "fade-in 0.3s ease-out, scale-in 0.2s ease-out",
        "exit": "fade-out 0.3s ease-out, scale-out 0.2s ease-out",
        "elegant-float": "elegant-float 8s ease-in-out infinite",
        "premium-glow": "premium-glow 4s ease-in-out infinite",
      },
      
      scale: {
        '102': '1.02',
      }
    }
  },
  plugins: [require("tailwindcss-animate")],
} satisfies Config;