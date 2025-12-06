// ============================================================================
// LIVE IT ICONIC - BORDER DESIGN TOKENS
// Border Radius & Width System
// ============================================================================

export const borders = {
  // ========================================================================
  // BORDER RADIUS - Shape System
  // ========================================================================
  radii: {
    none: '0', // No rounding
    sm: '0.125rem', // 2px - Small accents
    md: '0.375rem', // 6px - Form elements
    lg: '0.5rem', // 8px - Cards, buttons
    xl: '0.75rem', // 12px - Primary buttons, modals
    '2xl': '1rem', // 16px - Large containers
    '3xl': '1.5rem', // 24px - Hero sections
    full: '9999px', // Fully rounded (pills)

    // ====================================================================
    // CANONICAL RADIUS VALUES - Preferred across brand
    // ====================================================================
    // card: 1rem (16px) - Use for card components
    // control: 0.75rem (12px) - Use for buttons and inputs
    // container: 1.5rem (24px) - Use for large containers
  },

  // ========================================================================
  // BORDER WIDTH - Stroke System
  // ========================================================================
  widths: {
    0: '0',
    1: '1px',
    2: '2px',
    4: '4px',
    8: '8px',
  },

  // ========================================================================
  // BORDER STYLE
  // ========================================================================
  styles: {
    solid: 'solid',
    dashed: 'dashed',
    dotted: 'dotted',
    double: 'double',
    groove: 'groove',
    ridge: 'ridge',
    inset: 'inset',
    outset: 'outset',
  },
} as const;

// ============================================================================
// BORDER PRESETS - Common Combinations
// ============================================================================
export const borderPresets = {
  // Subtle borders
  subtle: {
    borderWidth: borders.widths[1],
    borderStyle: borders.styles.solid,
    borderColor: 'rgba(193, 160, 96, 0.1)',
  },

  // Standard borders
  standard: {
    borderWidth: borders.widths[1],
    borderStyle: borders.styles.solid,
    borderColor: 'rgba(193, 160, 96, 0.2)',
  },

  // Prominent borders
  prominent: {
    borderWidth: borders.widths[2],
    borderStyle: borders.styles.solid,
    borderColor: 'rgba(193, 160, 96, 0.3)',
  },

  // Focus ring (for accessibility)
  focusRing: {
    borderWidth: borders.widths[2],
    borderStyle: borders.styles.solid,
    borderColor: 'rgba(193, 160, 96, 0.5)',
    outline: 'none',
  },

  // Divider
  divider: {
    borderWidth: borders.widths[1],
    borderStyle: borders.styles.solid,
    borderColor: 'rgba(193, 160, 96, 0.1)',
  },
} as const;

// ============================================================================
// RADIUS PRESETS - Component Specific
// ============================================================================
export const radiusPresets = {
  // Button radius
  button: borders.radii.xl, // 12px
  buttonSmall: borders.radii.lg, // 8px
  buttonLarge: borders.radii['2xl'], // 16px

  // Card radius
  card: borders.radii['2xl'], // 16px
  cardSmall: borders.radii.xl, // 12px
  cardLarge: borders.radii['3xl'], // 24px

  // Input/Form radius
  input: borders.radii.lg, // 8px
  select: borders.radii.lg, // 8px
  textarea: borders.radii.lg, // 8px
  checkbox: borders.radii.md, // 6px
  radio: borders.radii.full, // 9999px (fully rounded)

  // Container radius
  container: borders.radii['2xl'], // 16px
  containerLarge: borders.radii['3xl'], // 24px
  containerSmall: borders.radii.lg, // 8px

  // Dialog/Modal radius
  modal: borders.radii['2xl'], // 16px
  modalLarge: borders.radii['3xl'], // 24px

  // Popover/Tooltip radius
  popover: borders.radii.lg, // 8px
  tooltip: borders.radii.md, // 6px

  // Avatar radius
  avatar: borders.radii.lg, // 8px
  avatarCircle: borders.radii.full, // 9999px

  // Badge radius
  badge: borders.radii.full, // 9999px
} as const;

// ============================================================================
// TYPE EXPORTS
// ============================================================================
export type Borders = typeof borders;
export type BorderRadii = typeof borders.radii;
export type BorderWidths = typeof borders.widths;
export type BorderStyle = typeof borders.styles;
export type RadiusKey = keyof typeof borders.radii;
export type BorderWidthKey = keyof typeof borders.widths;
export type RadiusPresetKey = keyof typeof radiusPresets;

// ============================================================================
// BORDER UTILITY FUNCTIONS
// ============================================================================
export const getRadius = (key: RadiusKey): string => {
  return borders.radii[key];
};

export const getBorderWidth = (key: BorderWidthKey): string => {
  return borders.widths[key];
};

export const getRadiusPreset = (key: RadiusPresetKey): string => {
  return radiusPresets[key];
};

/**
 * Creates a border string for CSS
 */
export const createBorder = (width: BorderWidthKey = 1, style: keyof typeof borders.styles = 'solid', color: string = 'currentColor'): string => {
  return `${borders.widths[width]} ${borders.styles[style]} ${color}`;
};

/**
 * Creates a border-radius CSS value
 */
export const createBorderRadius = (topLeft: RadiusKey, topRight?: RadiusKey, bottomRight?: RadiusKey, bottomLeft?: RadiusKey): string => {
  const tr = topRight ?? topLeft;
  const br = bottomRight ?? topLeft;
  const bl = bottomLeft ?? tr;

  return `${getRadius(topLeft)} ${getRadius(tr)} ${getRadius(br)} ${getRadius(bl)}`;
};

/**
 * Creates a focus ring border style
 */
export const createFocusRing = (offset: string = '2px', color: string = 'rgba(193, 160, 96, 0.5)', width: string = '2px'): string => {
  return `${width} solid ${color}`;
};

// ============================================================================
// FOCUS RING HELPERS
// ============================================================================
export const focusRingCSS = {
  outline: `${borders.widths[2]} solid rgba(193, 160, 96, 0.5)`,
  outlineOffset: '2px',
} as const;

/**
 * CSS-in-JS object for focus ring on interactive elements
 */
export const focusRingStyles = {
  '&:focus': {
    outline: focusRingCSS.outline,
    outlineOffset: focusRingCSS.outlineOffset,
  },
} as const;
