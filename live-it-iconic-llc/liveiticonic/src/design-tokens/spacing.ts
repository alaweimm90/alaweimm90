// ============================================================================
// LIVE IT ICONIC - SPACING DESIGN TOKENS
// Luxury Proportions & Layout Grid
// ============================================================================

export const spacing = {
  0: '0', // None
  1: '0.25rem', // 4px
  2: '0.5rem', // 8px (base unit)
  3: '0.75rem', // 12px
  4: '1rem', // 16px (canonical)
  5: '1.25rem', // 20px
  6: '1.5rem', // 24px
  8: '2rem', // 32px
  10: '2.5rem', // 40px
  12: '3rem', // 48px
  16: '4rem', // 64px
  20: '5rem', // 80px
  24: '6rem', // 96px
  32: '8rem', // 128px
} as const;

export const sizes = {
  xs: '20rem', // 320px
  sm: '24rem', // 384px
  md: '28rem', // 448px
  lg: '32rem', // 512px
  xl: '36rem', // 576px
  '2xl': '42rem', // 672px
  '3xl': '48rem', // 768px
  '4xl': '56rem', // 896px
  '5xl': '64rem', // 1024px
  '6xl': '72rem', // 1152px
  full: '100%',
  screen: '100vw',
  min: 'min-content',
  max: 'max-content',
  fit: 'fit-content',
} as const;

// ============================================================================
// SPACING SCALES - Common Combinations
// ============================================================================
export const spacingScales = {
  // Padding/Margin Presets
  xs: spacing[1], // 4px
  sm: spacing[2], // 8px
  md: spacing[4], // 16px
  lg: spacing[6], // 24px
  xl: spacing[8], // 32px
  '2xl': spacing[12], // 48px
  '3xl': spacing[16], // 64px
  '4xl': spacing[20], // 80px

  // Gap Presets (for flexbox/grid)
  gapXs: spacing[1],
  gapSm: spacing[2],
  gapMd: spacing[4],
  gapLg: spacing[6],
  gapXl: spacing[8],
  gap2xl: spacing[12],

  // Inset Values (position absolute/fixed)
  insetXs: spacing[1],
  insetSm: spacing[2],
  insetMd: spacing[4],
  insetLg: spacing[6],
  insetXl: spacing[8],
} as const;

// ============================================================================
// GRID & LAYOUT CONSTANTS
// ============================================================================
export const grid = {
  // Base grid unit (8px)
  baseUnit: spacing[2],

  // Grid columns
  columns: {
    mobile: 4,
    tablet: 8,
    desktop: 12,
  },

  // Container widths (max-widths)
  containers: {
    sm: '36rem', // 576px
    md: '48rem', // 768px
    lg: '64rem', // 1024px
    xl: '80rem', // 1280px
    '2xl': '96rem', // 1536px
    full: '100%',
  },

  // Common gaps
  gaps: {
    xs: spacing[1],
    sm: spacing[2],
    md: spacing[4],
    lg: spacing[6],
    xl: spacing[8],
    '2xl': spacing[12],
  },

  // Common padding
  padding: {
    xs: spacing[2],
    sm: spacing[4],
    md: spacing[6],
    lg: spacing[8],
    xl: spacing[12],
    '2xl': spacing[16],
  },

  // Common margins
  margin: {
    xs: spacing[2],
    sm: spacing[4],
    md: spacing[6],
    lg: spacing[8],
    xl: spacing[12],
    '2xl': spacing[16],
  },
} as const;

// ============================================================================
// TYPE EXPORTS
// ============================================================================
export type Spacing = typeof spacing;
export type Sizes = typeof sizes;
export type SpacingKey = keyof typeof spacing;
export type SizeKey = keyof typeof sizes;

// ============================================================================
// SPACING UTILITY FUNCTIONS
// ============================================================================
export const getSpacing = (key: SpacingKey): string => {
  return spacing[key];
};

export const getSize = (key: SizeKey): string => {
  return sizes[key];
};

/**
 * Creates a responsive padding string
 * @example responsivePadding(4, 6, 8) -> "p-4 sm:p-6 lg:p-8"
 */
export const responsivePadding = (mobile: SpacingKey, tablet?: SpacingKey, desktop?: SpacingKey): string => {
  let classes = `p-${mobile}`;
  if (tablet) classes += ` sm:p-${tablet}`;
  if (desktop) classes += ` lg:p-${desktop}`;
  return classes;
};

/**
 * Creates a responsive gap string
 * @example responsiveGap(2, 4, 6) -> "gap-2 sm:gap-4 lg:gap-6"
 */
export const responsiveGap = (mobile: SpacingKey, tablet?: SpacingKey, desktop?: SpacingKey): string => {
  let classes = `gap-${mobile}`;
  if (tablet) classes += ` sm:gap-${tablet}`;
  if (desktop) classes += ` lg:gap-${desktop}`;
  return classes;
};

/**
 * Creates shorthand spacing object for CSS-in-JS
 */
export const createSpacingObject = (top: SpacingKey, right?: SpacingKey, bottom?: SpacingKey, left?: SpacingKey) => {
  const r = right ?? top;
  const b = bottom ?? top;
  const l = left ?? r;

  return {
    marginTop: spacing[top],
    marginRight: spacing[r],
    marginBottom: spacing[b],
    marginLeft: spacing[l],
  };
};

/**
 * Creates padding shorthand object
 */
export const createPaddingObject = (top: SpacingKey, right?: SpacingKey, bottom?: SpacingKey, left?: SpacingKey) => {
  const r = right ?? top;
  const b = bottom ?? top;
  const l = left ?? r;

  return {
    paddingTop: spacing[top],
    paddingRight: spacing[r],
    paddingBottom: spacing[b],
    paddingLeft: spacing[l],
  };
};
