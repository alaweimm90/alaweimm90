import { cva } from 'class-variance-authority';
import { tokens } from '@/theme/tokens';

// ============================================================================
// LIVE IT ICONIC - CANONICAL BUTTON SYSTEM
// Premium Lifestyle Merchandise Brand
// ============================================================================

export const buttonVariants = cva(
  // Base styles - Premium brand standards with WCAG 2.2 focus indicators
  'inline-flex items-center justify-center gap-2 whitespace-nowrap text-sm font-medium transition-all focus-visible:outline focus-visible:outline-2 focus-visible:outline-lii-gold focus-visible:outline-offset-2 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0',
  {
    variants: {
      variant: {
        // Primary - Gold background, dark text (canonical CTA)
        primary: `bg-lii-gold text-lii-bg hover:bg-lii-gold-press rounded-xl shadow-[0_0_20px_rgba(193,160,96,0.3)] hover:shadow-[0_0_30px_rgba(193,160,96,0.5)] transition-all duration-300 hover:transform hover:-translate-y-0.5`,

        // Secondary - Gold outline, gold text (alternative CTA)
        secondary: `border-[1.5px] border-lii-gold text-lii-gold bg-transparent hover:bg-lii-gold/10 rounded-xl transition-all duration-300`,

        // Ghost - Subtle text, transparent (tertiary actions)
        ghost: `text-lii-cloud bg-transparent hover:bg-lii-cloud/10 rounded-xl transition-all duration-300`,

        // Outline - Alternative to secondary (legacy support)
        outline: `border-[1.5px] border-lii-gold text-lii-gold bg-transparent hover:bg-lii-gold/10 rounded-xl transition-all duration-300`,

        // Destructive - Error states (signal red)
        destructive: `bg-lii-signal-red text-lii-cloud hover:bg-lii-signal-red/90 rounded-xl transition-all duration-300`,

        // Link - Text-only links
        link: `text-lii-gold underline-offset-4 hover:underline bg-transparent rounded-xl transition-all duration-300`,

        // Legacy aliases for backward compatibility
        default: `bg-lii-gold text-lii-bg hover:bg-lii-gold-press rounded-xl shadow-[0_0_20px_rgba(193,160,96,0.3)] hover:shadow-[0_0_30px_rgba(193,160,96,0.5)] transition-all duration-300 hover:transform hover:-translate-y-0.5`,
      },
      size: {
        // Default - Standard button size
        default: 'h-12 px-6 py-3 text-base',

        // Small - Compact buttons
        sm: 'h-10 px-4 py-2 text-sm',

        // Large - Hero/primary CTAs
        lg: 'h-14 px-8 py-4 text-lg',

        // Icon - Square icon buttons
        icon: 'h-12 w-12 p-0',
      },
    },
    defaultVariants: {
      variant: 'primary', // Canonical primary variant
      size: 'default',
    },
  }
);

export default buttonVariants;
