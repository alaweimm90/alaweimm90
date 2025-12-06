// ============================================================================
// LIVE IT ICONIC - SHADOW DESIGN TOKENS (ELEVATION SYSTEM)
// Luxury Shadow Palette with Gold Undertones
// ============================================================================

export const shadows = {
  // ========================================================================
  // ELEVATION SHADOWS - Subtle to Dramatic
  // ========================================================================
  none: 'none',

  xs: '0 1px 2px 0 rgba(193, 160, 96, 0.05)',

  sm: '0 1px 3px 0 rgba(193, 160, 96, 0.1), 0 1px 2px 0 rgba(193, 160, 96, 0.06)',

  md: '0 4px 6px -1px rgba(193, 160, 96, 0.1), 0 2px 4px -1px rgba(193, 160, 96, 0.06)',

  lg: '0 10px 15px -3px rgba(193, 160, 96, 0.1), 0 4px 6px -2px rgba(193, 160, 96, 0.05)',

  xl: '0 20px 25px -5px rgba(193, 160, 96, 0.1), 0 10px 10px -5px rgba(193, 160, 96, 0.04)',

  '2xl': '0 25px 50px -12px rgba(193, 160, 96, 0.25)',

  // ========================================================================
  // SPECIALIZED SHADOWS - Component Specific
  // ========================================================================
  inner: 'inset 0 2px 4px 0 rgba(193, 160, 96, 0.06)',

  // ========================================================================
  // LEGACY ALIASES - Backward Compatibility
  // ========================================================================
  base: '0 8px 24px rgba(193, 160, 96, 0.12)',
  hover: '0 12px 32px rgba(193, 160, 96, 0.16)',
  active: '0 4px 16px rgba(193, 160, 96, 0.08)',
} as const;

// ============================================================================
// SHADOW PRESETS - Elevation Levels
// ============================================================================
export const shadowPresets = {
  // No elevation
  none: shadows.none,

  // Level 1 - Subtle
  elevation1: shadows.xs,

  // Level 2 - Light
  elevation2: shadows.sm,

  // Level 3 - Medium
  elevation3: shadows.md,

  // Level 4 - Prominent
  elevation4: shadows.lg,

  // Level 5 - High
  elevation5: shadows.xl,

  // Level 6 - Very High
  elevation6: shadows['2xl'],

  // Component-specific shadows
  card: shadows.md,
  cardHover: shadows.lg,
  modal: shadows['2xl'],
  popover: shadows.xl,
  tooltip: shadows.sm,
  dropdown: shadows.md,
  sticky: shadows.sm,
  button: shadows.sm,
  buttonHover: shadows.md,
  input: shadows.xs,
  inputFocus: shadows.sm,
} as const;

// ============================================================================
// DYNAMIC SHADOW GENERATION
// ============================================================================
export const createShadow = (
  offsetX: number,
  offsetY: number,
  blur: number,
  spread: number,
  opacity: number
): string => {
  // Using gold color for luxury aesthetic
  const alpha = Math.max(0, Math.min(1, opacity)); // Clamp 0-1
  return `${offsetX}px ${offsetY}px ${blur}px ${spread}px rgba(193, 160, 96, ${alpha})`;
};

/**
 * Creates a layered shadow effect (multiple shadows stacked)
 */
export const createLayeredShadow = (...shadowValues: string[]): string => {
  return shadowValues.join(', ');
};

/**
 * Creates a glow effect using shadow
 */
export const createGlow = (color: string, size: number, opacity: number = 0.5): string => {
  return `0 0 ${size}px ${size / 2}px ${color}${Math.round(opacity * 255)
    .toString(16)
    .padStart(2, '0')}`;
};

// ============================================================================
// TYPE EXPORTS
// ============================================================================
export type Shadows = typeof shadows;
export type ShadowKey = keyof typeof shadows;
export type ShadowPreset = keyof typeof shadowPresets;

// ============================================================================
// SHADOW UTILITY FUNCTIONS
// ============================================================================
export const getShadow = (key: ShadowKey): string => {
  return shadows[key];
};

export const getShadowPreset = (key: ShadowPreset): string => {
  return shadowPresets[key];
};

/**
 * Maps elevation level (1-6) to appropriate shadow
 */
export const getElevationShadow = (level: 1 | 2 | 3 | 4 | 5 | 6): string => {
  const levelMap: Record<number, ShadowKey> = {
    1: 'xs',
    2: 'sm',
    3: 'md',
    4: 'lg',
    5: 'xl',
    6: '2xl',
  };

  return shadows[levelMap[level]];
};

// ============================================================================
// SHADOW ANIMATION HELPERS
// ============================================================================
export const shadowTransition = 'box-shadow 0.3s cubic-bezier(0.16, 1, 0.3, 1)';

/**
 * CSS-in-JS object for shadow on hover
 */
export const shadowOnHover = {
  transition: shadowTransition,
  '&:hover': {
    boxShadow: shadows.lg,
  },
} as const;

/**
 * CSS-in-JS object for shadow on focus
 */
export const shadowOnFocus = {
  transition: shadowTransition,
  '&:focus': {
    boxShadow: shadows.lg,
  },
} as const;
