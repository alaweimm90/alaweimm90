// =====================================================================
// REPZ COACH - CENTRALIZED DESIGN TOKENS
// Enterprise-grade design system foundation
// =====================================================================

export const designTokens = {
  // CORE BRAND IDENTITY
  brand: {
    primary: 'hsl(14, 87%, 54%)',      // #F15B23 - Official REPZ Orange
    primaryLight: 'hsl(25, 96%, 62%)', // #FB923C
    primaryDark: 'hsl(16, 84%, 45%)',  // #D4460A
    black: 'hsl(0, 0%, 0%)',           // #000000 - Official REPZ Black
    white: 'hsl(0, 0%, 100%)',         // #FFFFFF
  },

  // TIER BUSINESS MODEL COLORS
  tiers: {
    core: {
      primary: 'hsl(210, 15%, 60%)',    // Gray foundation
      light: 'hsl(210, 15%, 70%)',
      dark: 'hsl(210, 15%, 50%)',
      bg: 'hsl(210, 15%, 60%, 0.03)',
      border: 'hsl(210, 15%, 60%, 0.2)',
    },
    adaptive: {
      primary: 'hsl(330, 60%, 75%)',    // Pink enhanced
      light: 'hsl(330, 60%, 85%)',
      dark: 'hsl(330, 60%, 65%)',
      bg: 'hsl(330, 60%, 75%, 0.03)',
      border: 'hsl(330, 60%, 75%, 0.2)',
    },
    performance: {
      primary: 'hsl(269, 60%, 35%)',    // Purple elite
      light: 'hsl(269, 60%, 45%)',
      dark: 'hsl(269, 60%, 25%)',
      bg: 'hsl(269, 60%, 35%, 0.03)',
      border: 'hsl(269, 60%, 35%, 0.2)',
    },
    longevity: {
      primary: 'hsl(43, 84%, 40%)',     // Gold luxury
      light: 'hsl(43, 84%, 50%)',
      dark: 'hsl(43, 84%, 30%)',
      bg: 'hsl(43, 84%, 40%, 0.03)',
      border: 'hsl(43, 84%, 40%, 0.2)',
    },
  },
} as const;

// Legacy compatibility exports
export const baseTheme = designTokens;
export const coachTheme = designTokens;
export type ThemeTokens = typeof designTokens;
export type DesignTokens = typeof designTokens;