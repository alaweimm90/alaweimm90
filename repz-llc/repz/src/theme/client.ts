// Client Tier Themes - Follows WHOOP/Peloton tier-based overrides
// Base theme + tier-specific accent colors only

import { baseTheme, type ThemeTokens } from './tokens';
import type { TierType } from '@/types/business';

// Tier-specific overrides (accent colors only)
export const tierOverrides = {
  core: {
    accentColor: 'text-tier-core',
    accentBg: 'bg-tier-core',
    accentBorder: 'border-tier-core',
    accentGlow: 'shadow-glow-gray',
    dashboardAccent: 'bg-tier-core/10',
  },
  adaptive: {
    accentColor: 'text-tier-adaptive', // Light Pink
    accentBg: 'bg-tier-adaptive',
    accentBorder: 'border-tier-adaptive',
    accentGlow: 'shadow-glow-pink',
    dashboardAccent: 'bg-tier-adaptive/10',
  },
  performance: {
    accentColor: 'text-tier-performance',
    accentBg: 'bg-tier-performance',
    accentBorder: 'border-tier-performance',
    accentGlow: 'shadow-glow-purple',
    dashboardAccent: 'bg-tier-performance/10',
  },
  longevity: {
    accentColor: 'text-tier-longevity',
    accentBg: 'bg-tier-longevity',
    accentBorder: 'border-tier-longevity',
    accentGlow: 'shadow-glow-gold',
    dashboardAccent: 'bg-tier-longevity/10',
  },
} as const;

// Generate complete client themes (base + tier overrides)
export const clientThemes = {
  core: { ...baseTheme, ...tierOverrides.core },
  adaptive: { ...baseTheme, ...tierOverrides.adaptive },
  performance: { ...baseTheme, ...tierOverrides.performance },
  longevity: { ...baseTheme, ...tierOverrides.longevity },
} as const;

export const getClientTheme = (tier: TierType) => clientThemes[tier];

export type ClientTheme = typeof clientThemes.core;