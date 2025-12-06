// ============================================================================
// LIVE IT ICONIC - COLOR DESIGN TOKENS
// Premium Luxury Brand Color Palette
// ============================================================================

export const colors = {
  // ========================================================================
  // PRIMARY PALETTE - Brand Colors
  // ========================================================================
  gold: {
    50: '#F5EFE4',
    100: '#E8DCC4',
    200: '#D9C69F',
    300: '#C1A060', // Primary gold (canonical)
    400: '#AE8D55',
    500: '#9A7A4A',
    600: '#7D6339',
    700: '#5F4C2B',
    800: '#42351D',
    900: '#241E0F',
  },

  charcoal: {
    50: '#8C93A3',
    100: '#5C6270',
    200: '#2D3142',
    300: '#0B0B0C', // Primary charcoal (canonical)
    400: '#080809',
    500: '#050506',
  },

  cloud: {
    50: '#FFFFFF',
    100: '#F5F7FA',
    200: '#E6E9EF', // Primary cloud (canonical)
    300: '#D0D5DD',
    400: '#BCC2CC',
  },

  // ========================================================================
  // LEGACY BRAND COLORS - Backward Compatibility
  // ========================================================================
  carbonBlack: '#1a1a1a',
  championshipGold: '#d4af37',
  platinumSilver: '#c0c0c0',
  champagne: '#f7e7ce',
  gunmetal: '#2a3439',

  // ========================================================================
  // NEUTRAL PALETTE - Grays & Backgrounds
  // ========================================================================
  bg: '#0B0B0C', // Primary background
  ink: '#14161A', // Secondary background
  graphite: '#3A4048', // Component backgrounds
  ash: '#8C93A3', // Body text
  snow: '#F8F9FA', // Pure white accents

  // ========================================================================
  // SEMANTIC COLORS - Status & Meaning
  // ========================================================================
  success: {
    50: '#ECFDF5',
    100: '#D1FAE5',
    200: '#A7F3D0',
    300: '#6EE7B7',
    400: '#34D399',
    500: '#10B981', // Primary success
    600: '#059669',
    700: '#047857',
    800: '#065F46',
    900: '#064E3B',
  },

  warning: {
    50: '#FFFBEB',
    100: '#FEF3C7',
    200: '#FDE68A',
    300: '#FCD34D',
    400: '#FBBF24',
    500: '#F59E0B', // Primary warning
    600: '#D97706',
    700: '#B45309',
    800: '#92400E',
    900: '#78350F',
  },

  error: {
    50: '#FEF2F2',
    100: '#FEE2E2',
    200: '#FECACA',
    300: '#FCA5A5',
    400: '#F87171',
    500: '#E03A2F', // Primary error (from existing tokens)
    600: '#DC2626',
    700: '#B91C1C',
    800: '#991B1B',
    900: '#7F1D1D',
  },

  info: {
    50: '#EFF6FF',
    100: '#DBEAFE',
    200: '#BFDBFE',
    300: '#93C5FD',
    400: '#60A5FA',
    500: '#3B82F6', // Primary info
    600: '#2563EB',
    700: '#1D4ED8',
    800: '#1E40AF',
    900: '#1E3A8A',
  },

  // ========================================================================
  // SIGNAL COLORS - Component States
  // ========================================================================
  signalRed: '#E03A2F',
  signalGreen: '#10B981',
  signalOrange: '#F59E0B',
  signalBlue: '#3B82F6',
} as const;

// ============================================================================
// TYPE EXPORTS
// ============================================================================
export type Colors = typeof colors;
export type GoldShades = typeof colors.gold;
export type CharcoalShades = typeof colors.charcoal;
export type CloudShades = typeof colors.cloud;
export type SuccessShades = typeof colors.success;
export type WarningShades = typeof colors.warning;
export type ErrorShades = typeof colors.error;
export type InfoShades = typeof colors.info;

// ============================================================================
// COLOR UTILITY FUNCTIONS
// ============================================================================
export const getColor = (path: string): string | undefined => {
  const keys = path.split('.');
  let current: any = colors;

  for (const key of keys) {
    current = current?.[key];
  }

  return typeof current === 'string' ? current : undefined;
};

/**
 * Creates a semi-transparent color using rgba
 */
export const withOpacity = (color: string, opacity: number): string => {
  if (color.startsWith('#')) {
    const hex = color.replace('#', '');
    const r = parseInt(hex.substring(0, 2), 16);
    const g = parseInt(hex.substring(2, 4), 16);
    const b = parseInt(hex.substring(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${opacity})`;
  }
  return color;
};

/**
 * Color palette reference for design documentation
 */
export const colorPalette = {
  primary: {
    gold: colors.gold[300],
    charcoal: colors.charcoal[300],
    cloud: colors.cloud[200],
  },
  semantic: {
    success: colors.success[500],
    warning: colors.warning[500],
    error: colors.error[500],
    info: colors.info[500],
  },
  backgrounds: {
    primary: colors.bg,
    secondary: colors.ink,
    tertiary: colors.graphite,
  },
  text: {
    primary: colors.cloud[200],
    secondary: colors.ash,
    inverted: colors.charcoal[300],
  },
} as const;
