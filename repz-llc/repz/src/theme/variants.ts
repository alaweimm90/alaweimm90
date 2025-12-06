// =====================================================================
// REPZ COACH - COMPONENT VARIANT SYSTEM
// Centralized component theming with tier support
// =====================================================================

import { designTokens } from './tokens';

export type TierType = 'core' | 'adaptive' | 'performance' | 'longevity';
export type ComponentSize = 'sm' | 'md' | 'lg' | 'xl';
export type ComponentVariant = 'default' | 'primary' | 'secondary' | 'ghost' | 'outline';

// BUTTON VARIANT SYSTEM
export const buttonVariants = {
  base: [
    'inline-flex items-center justify-center',
    'font-medium rounded-lg transition-all',
    'focus:outline-none focus:ring-2 focus:ring-offset-2',
    'disabled:opacity-50 disabled:cursor-not-allowed',
  ].join(' '),

  variants: {
    default: [
      'bg-primary text-white shadow-md',
      'hover:bg-primary/90 hover:shadow-lg',
      'focus:ring-primary/50',
    ].join(' '),

    primary: [
      'bg-gradient-to-r from-[#F15B23] to-[#FB923C]',
      'text-white shadow-lg backdrop-blur-sm',
      'hover:shadow-xl hover:scale-105',
      'focus:ring-[#F15B23]/50',
      'border border-white/10',
    ].join(' '),

    secondary: [
      'bg-white/8 text-white border border-white/20',
      'backdrop-blur-sm hover:bg-white/12',
      'hover:border-white/30 hover:scale-102',
      'focus:ring-white/50',
    ].join(' '),

    ghost: [
      'text-white/80 hover:text-[#F15B23]',
      'hover:bg-[#F15B23]/10 backdrop-blur-sm',
      'focus:ring-[#F15B23]/50',
    ].join(' '),

    outline: [
      'border-2 border-current text-white',
      'hover:bg-current hover:text-black',
      'focus:ring-current/50',
    ].join(' '),
  },

  sizes: {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-base',
    lg: 'px-6 py-3 text-lg',
    xl: 'px-8 py-4 text-xl',
  },
};

// CARD VARIANT SYSTEM
export const cardVariants = {
  base: [
    'relative rounded-xl transition-all duration-500',
    'backdrop-blur-xl shadow-lg border',
    'hover:scale-105 hover:shadow-xl',
  ].join(' '),

  variants: {
    default: [
      'bg-[#181A20] border-white/10',
      'hover:border-white/20',
    ].join(' '),

    glass: [
      'bg-white/5 border-white/10',
      'backdrop-blur-2xl',
      'hover:bg-white/10 hover:border-white/20',
    ].join(' '),

    elevated: [
      'bg-[#1F1F1F] border-white/15',
      'shadow-2xl',
      'hover:shadow-3xl',
    ].join(' '),
  },

  tiers: {
    core: [
      'bg-gradient-to-br from-gray-500/3 to-gray-600/2',
      'border-gray-400/20 hover:border-gray-400/40',
    ].join(' '),

    adaptive: [
      'bg-gradient-to-br from-pink-300/3 to-pink-400/2',
      'border-pink-300/20 hover:border-pink-300/40',
    ].join(' '),

    performance: [
      'bg-gradient-to-br from-indigo-600/3 to-purple-700/2',
      'border-indigo-400/20 hover:border-indigo-400/40',
    ].join(' '),

    longevity: [
      'bg-gradient-to-br from-amber-500/3 to-amber-600/2',
      'border-amber-400/20 hover:border-amber-400/40',
    ].join(' '),
  },
};

// BADGE VARIANT SYSTEM
export const badgeVariants = {
  base: [
    'inline-flex items-center px-3 py-1',
    'text-xs font-bold uppercase rounded-full',
    'transition-all duration-300',
  ].join(' '),

  variants: {
    primary: [
      'bg-[#F15B23] text-white',
      'shadow-md hover:shadow-lg',
    ].join(' '),

    secondary: [
      'bg-white/10 text-white border border-white/20',
      'backdrop-blur-sm hover:bg-white/20',
    ].join(' '),

    success: [
      'bg-green-500/20 text-green-400 border border-green-500/30',
      'backdrop-blur-sm hover:bg-green-500/30',
    ].join(' '),

    warning: [
      'bg-amber-500/20 text-amber-400 border border-amber-500/30',
      'backdrop-blur-sm hover:bg-amber-500/30',
    ].join(' '),

    premium: [
      'bg-gradient-to-r from-yellow-400 to-yellow-600',
      'text-black border border-yellow-500',
      'shadow-md hover:shadow-lg',
    ].join(' '),
  },
};

// PANEL VARIANT SYSTEM
export const panelVariants = {
  base: [
    'relative rounded-2xl p-6',
    'backdrop-blur-xl transition-all duration-500',
  ].join(' '),

  variants: {
    glass: [
      'bg-white/5 border border-white/10',
      'shadow-2xl backdrop-blur-2xl',
    ].join(' '),

    frost: [
      'bg-white/8 border border-white/5',
      'shadow-xl backdrop-blur-lg',
    ].join(' '),

    velvet: [
      'bg-black/40 border border-white/15',
      'shadow-lg',
    ].join(' '),

    hero: [
      'bg-gradient-to-br from-[#141414] via-[#1F1F1F] to-[#2E2E2E]',
      'border border-[#F15B23]/20 shadow-2xl',
    ].join(' '),
  },
};

// TIER COLOR UTILITIES
export const getTierColors = (tier: TierType) => {
  const colors = {
    core: {
      primary: 'text-gray-400',
      bg: 'bg-gray-500/20',
      border: 'border-gray-400/30',
      glow: 'shadow-gray-400/20',
    },
    adaptive: {
      primary: 'text-pink-300',
      bg: 'bg-pink-300/20',
      border: 'border-pink-300/30',
      glow: 'shadow-pink-300/20',
    },
    performance: {
      primary: 'text-indigo-400',
      bg: 'bg-indigo-600/20',
      border: 'border-indigo-400/30',
      glow: 'shadow-indigo-400/20',
    },
    longevity: {
      primary: 'text-amber-400',
      bg: 'bg-amber-500/20',
      border: 'border-amber-400/30',
      glow: 'shadow-amber-400/20',
    },
  };

  return colors[tier];
};

// COMPONENT FACTORY FUNCTIONS
export const createButtonClass = (
  variant: ComponentVariant = 'default',
  size: ComponentSize = 'md'
) => {
  return [
    buttonVariants.base,
    buttonVariants.variants[variant],
    buttonVariants.sizes[size],
  ].join(' ');
};

export const createCardClass = (
  variant: 'default' | 'glass' | 'elevated' = 'default',
  tier?: TierType
) => {
  const classes = [
    cardVariants.base,
    cardVariants.variants[variant],
  ];

  if (tier) {
    classes.push(cardVariants.tiers[tier]);
  }

  return classes.join(' ');
};

export const createBadgeClass = (
  variant: keyof typeof badgeVariants.variants = 'primary'
) => {
  return [
    badgeVariants.base,
    badgeVariants.variants[variant],
  ].join(' ');
};

export const createPanelClass = (
  variant: keyof typeof panelVariants.variants = 'glass'
) => {
  return [
    panelVariants.base,
    panelVariants.variants[variant],
  ].join(' ');
};