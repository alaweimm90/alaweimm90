// ============================================================================
// LIVE IT ICONIC - TYPOGRAPHY DESIGN TOKENS
// Premium Font Stacks & Text Styling
// ============================================================================

export const typography = {
  // ========================================================================
  // FONT FAMILIES
  // ========================================================================
  fontFamilies: {
    display: "'Playfair Display', serif",
    body: "'Inter Variable', sans-serif",
    mono: "'JetBrains Mono', monospace",
  },

  // ========================================================================
  // FONT SIZES (Mobile-First Scale)
  // ========================================================================
  fontSizes: {
    xs: '0.75rem', // 12px - Captions, labels
    sm: '0.875rem', // 14px - Small text
    base: '1rem', // 16px - Body text (canonical)
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

  // ========================================================================
  // FONT WEIGHTS
  // ========================================================================
  fontWeights: {
    thin: 100,
    extralight: 200,
    light: 300,
    normal: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
    extrabold: 800,
    black: 900,
  },

  // ========================================================================
  // LINE HEIGHTS
  // ========================================================================
  lineHeights: {
    none: 1,
    tight: 1.25,
    snug: 1.375,
    normal: 1.5,
    relaxed: 1.625,
    loose: 2,
  },

  // ========================================================================
  // LETTER SPACING
  // ========================================================================
  letterSpacing: {
    tighter: '-0.05em',
    tight: '-0.025em',
    normal: '0',
    wide: '0.025em',
    wider: '0.05em',
    widest: '0.1em',
  },
} as const;

// ============================================================================
// TYPE EXPORTS
// ============================================================================
export type Typography = typeof typography;
export type FontSize = keyof typeof typography.fontSizes;
export type FontWeight = keyof typeof typography.fontWeights;
export type LineHeight = keyof typeof typography.lineHeights;
export type LetterSpacing = keyof typeof typography.letterSpacing;

// ============================================================================
// TYPOGRAPHY PRESETS - Common Text Styles
// ============================================================================
export const typographyPresets = {
  // Display/Hero Presets
  displayLarge: {
    fontFamily: typography.fontFamilies.display,
    fontSize: typography.fontSizes['7xl'],
    fontWeight: typography.fontWeights.bold,
    lineHeight: typography.lineHeights.tight,
    letterSpacing: typography.letterSpacing.tight,
  },

  displayMedium: {
    fontFamily: typography.fontFamilies.display,
    fontSize: typography.fontSizes['5xl'],
    fontWeight: typography.fontWeights.bold,
    lineHeight: typography.lineHeights.tight,
  },

  displaySmall: {
    fontFamily: typography.fontFamilies.display,
    fontSize: typography.fontSizes['4xl'],
    fontWeight: typography.fontWeights.semibold,
    lineHeight: typography.lineHeights.snug,
  },

  // Heading Presets
  headingLarge: {
    fontFamily: typography.fontFamilies.body,
    fontSize: typography.fontSizes['3xl'],
    fontWeight: typography.fontWeights.bold,
    lineHeight: typography.lineHeights.snug,
  },

  headingMedium: {
    fontFamily: typography.fontFamilies.body,
    fontSize: typography.fontSizes['2xl'],
    fontWeight: typography.fontWeights.bold,
    lineHeight: typography.lineHeights.snug,
  },

  headingSmall: {
    fontFamily: typography.fontFamilies.body,
    fontSize: typography.fontSizes.xl,
    fontWeight: typography.fontWeights.semibold,
    lineHeight: typography.lineHeights.normal,
  },

  // Body Presets
  bodyLarge: {
    fontFamily: typography.fontFamilies.body,
    fontSize: typography.fontSizes.lg,
    fontWeight: typography.fontWeights.normal,
    lineHeight: typography.lineHeights.relaxed,
  },

  bodyRegular: {
    fontFamily: typography.fontFamilies.body,
    fontSize: typography.fontSizes.base,
    fontWeight: typography.fontWeights.normal,
    lineHeight: typography.lineHeights.normal,
  },

  bodySmall: {
    fontFamily: typography.fontFamilies.body,
    fontSize: typography.fontSizes.sm,
    fontWeight: typography.fontWeights.normal,
    lineHeight: typography.lineHeights.normal,
  },

  // Caption/Label Presets
  caption: {
    fontFamily: typography.fontFamilies.body,
    fontSize: typography.fontSizes.xs,
    fontWeight: typography.fontWeights.medium,
    lineHeight: typography.lineHeights.tight,
    letterSpacing: typography.letterSpacing.wide,
  },

  // Monospace Presets
  code: {
    fontFamily: typography.fontFamilies.mono,
    fontSize: typography.fontSizes.sm,
    fontWeight: typography.fontWeights.normal,
    lineHeight: typography.lineHeights.relaxed,
  },

  // Button Presets
  button: {
    fontFamily: typography.fontFamilies.body,
    fontSize: typography.fontSizes.base,
    fontWeight: typography.fontWeights.semibold,
    lineHeight: typography.lineHeights.tight,
    letterSpacing: typography.letterSpacing.wide,
  },
} as const;

// ============================================================================
// TYPOGRAPHY UTILITY FUNCTIONS
// ============================================================================
export const getTypography = (path: string): any => {
  const keys = path.split('.');
  let current: any = typography;

  for (const key of keys) {
    current = current?.[key];
  }

  return current;
};

export const getFontFamily = (family: keyof typeof typography.fontFamilies): string => {
  return typography.fontFamilies[family];
};

export const getFontSize = (size: keyof typeof typography.fontSizes): string => {
  return typography.fontSizes[size];
};

export const getFontWeight = (weight: keyof typeof typography.fontWeights): number => {
  return typography.fontWeights[weight];
};
