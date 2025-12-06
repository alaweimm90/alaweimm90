// ============================================================================
// LIVE IT ICONIC - DESIGN TOKENS MAIN EXPORT
// Comprehensive Design System Token Library
// ============================================================================

// Re-export all token modules
export * from './colors';
export * from './typography';
export * from './spacing';
export * from './shadows';
export * from './borders';
export * from './motion';
export * from './breakpoints';
export * from './z-index';

// Import all tokens for aggregated export
import { colors } from './colors';
import { typography } from './typography';
import { spacing, sizes, grid } from './spacing';
import { shadows, shadowPresets } from './shadows';
import { borders, radiusPresets } from './borders';
import { motion, motionPresets } from './motion';
import { breakpoints } from './breakpoints';
import { zIndex, zIndexPresets } from './z-index';

// ============================================================================
// AGGREGATED TOKENS OBJECT
// ============================================================================
export const tokens = {
  colors,
  typography,
  spacing,
  sizes,
  grid,
  shadows,
  shadowPresets,
  borders,
  radiusPresets,
  motion,
  motionPresets,
  breakpoints,
  zIndex,
  zIndexPresets,
} as const;

// ============================================================================
// TYPE EXPORTS
// ============================================================================
export type DesignTokens = typeof tokens;
export type TokenCategory = keyof typeof tokens;

// ============================================================================
// UTILITY FUNCTION - Get any token by path
// ============================================================================
/**
 * Deep retrieves a token value by dot notation path
 * @example getToken('colors.gold.300') -> '#C1A060'
 * @example getToken('typography.fontSizes.base') -> '1rem'
 */
export const getToken = (path: string): any => {
  const keys = path.split('.');
  let current: any = tokens;

  for (const key of keys) {
    if (current === undefined || current === null) {
      console.warn(`Token path not found: ${path}`);
      return undefined;
    }
    current = current[key];
  }

  return current;
};

// ============================================================================
// TOKEN SUMMARY & STATISTICS
// ============================================================================
export const tokenStats = {
  colors: Object.keys(colors).length,
  typographySizes: Object.keys(typography.fontSizes).length,
  typographyWeights: Object.keys(typography.fontWeights).length,
  spacingValues: Object.keys(spacing).length,
  containerSizes: Object.keys(sizes).length,
  shadows: Object.keys(shadows).length,
  borderRadii: Object.keys(borders.radii).length,
  borderWidths: Object.keys(borders.widths).length,
  durations: Object.keys(motion.durations).length,
  easings: Object.keys(motion.easings).length,
  breakpoints: Object.keys(breakpoints).length,
  zIndexLayers: Object.keys(zIndex).length,
} as const;

// ============================================================================
// QUICK ACCESS HELPERS
// ============================================================================
export const tokenHelpers = {
  // Color shortcuts
  colors: {
    primary: colors.gold[300],
    secondary: colors.cloud[200],
    accent: colors.gold[400],
    text: colors.cloud[200],
    textSecondary: colors.ash,
    background: colors.charcoal[300],
    surface: colors.charcoal[400],
    border: 'rgba(193, 160, 96, 0.2)',
    success: colors.success[500],
    warning: colors.warning[500],
    error: colors.error[500],
    info: colors.info[500],
  },

  // Spacing shortcuts
  spacing: {
    xxs: spacing[1],
    xs: spacing[2],
    sm: spacing[4],
    md: spacing[6],
    lg: spacing[8],
    xl: spacing[12],
    xxl: spacing[16],
  },

  // Border radius shortcuts
  radius: {
    none: borders.radii.none,
    button: borders.radii.xl,
    card: borders.radii['2xl'],
    container: borders.radii['3xl'],
    pill: borders.radii.full,
  },

  // Typography shortcuts
  text: {
    display: {
      fontFamily: typography.fontFamilies.display,
      fontSize: typography.fontSizes['5xl'],
      fontWeight: typography.fontWeights.bold,
    },
    heading: {
      fontFamily: typography.fontFamilies.body,
      fontSize: typography.fontSizes['2xl'],
      fontWeight: typography.fontWeights.bold,
    },
    body: {
      fontFamily: typography.fontFamilies.body,
      fontSize: typography.fontSizes.base,
      fontWeight: typography.fontWeights.normal,
    },
    caption: {
      fontFamily: typography.fontFamilies.body,
      fontSize: typography.fontSizes.xs,
      fontWeight: typography.fontWeights.medium,
    },
  },

  // Motion shortcuts
  motion: {
    fast: `${motion.durations.fast} ${motion.easings.easeOutCubic}`,
    normal: `${motion.durations.normal} ${motion.easings.easeOutCubic}`,
    slow: `${motion.durations.slow} ${motion.easings.easeOutCubic}`,
  },

  // Shadow shortcuts
  shadow: {
    none: shadows.none,
    subtle: shadows.xs,
    small: shadows.sm,
    medium: shadows.md,
    large: shadows.lg,
    xlarge: shadows.xl,
    huge: shadows['2xl'],
  },
} as const;

// ============================================================================
// PALETTE EXPORT FOR DESIGN TOOLS
// ============================================================================
export const designPalette = {
  name: 'Live It Iconic - Premium Luxury Brand Design System',
  version: '1.0.0',
  colors: {
    primary: {
      gold: colors.gold,
      charcoal: colors.charcoal,
      cloud: colors.cloud,
    },
    semantic: {
      success: colors.success,
      warning: colors.warning,
      error: colors.error,
      info: colors.info,
    },
  },
  typography: {
    families: typography.fontFamilies,
    sizes: typography.fontSizes,
    weights: typography.fontWeights,
  },
  spacing: spacing,
  shadows: shadows,
  radii: borders.radii,
  breakpoints: breakpoints,
} as const;

// ============================================================================
// EXPORT SUMMARY
// ============================================================================
export const tokenDocumentation = {
  version: '1.0.0',
  lastUpdated: new Date().toISOString(),
  categories: [
    'colors (with semantic variants)',
    'typography (fonts, sizes, weights, line-heights, letter-spacing)',
    'spacing (base scale + grid system)',
    'shadows (elevation system)',
    'borders (radius + widths)',
    'motion (durations + easings + keyframes)',
    'breakpoints (mobile-first responsive)',
    'z-index (stacking context)',
  ],
  stats: tokenStats,
  usage: {
    typescript: 'import { tokens } from "@/design-tokens"',
    css: '@import url("/dist/tokens/variables.css")',
    tailwind: 'Integrated via tailwind.config.ts',
  },
} as const;
