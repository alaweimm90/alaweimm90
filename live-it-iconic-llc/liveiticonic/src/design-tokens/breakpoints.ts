// ============================================================================
// LIVE IT ICONIC - BREAKPOINT DESIGN TOKENS
// Responsive Design Breakpoints (Mobile-First)
// ============================================================================

export const breakpoints = {
  // ========================================================================
  // STANDARD BREAKPOINTS - Mobile-First Approach
  // ========================================================================
  xs: '320px', // Mobile (iPhone SE, small phones)
  sm: '640px', // Small tablets (iPad mini)
  md: '768px', // Medium tablets (iPad, standard tablet)
  lg: '1024px', // Large tablets, small laptops
  xl: '1280px', // Desktops, laptops
  '2xl': '1536px', // Large desktops, ultra-wide displays
} as const;

// ============================================================================
// BREAKPOINT DEFINITIONS - Extended Details
// ============================================================================
export const breakpointDefinitions = {
  xs: {
    name: 'Extra Small',
    size: '320px',
    width: 320,
    description: 'Mobile phones (iPhone SE, small devices)',
    devices: ['iPhone SE', 'iPhone 12 mini', 'Small Android phones'],
  },

  sm: {
    name: 'Small',
    size: '640px',
    width: 640,
    description: 'Large phones, small tablets',
    devices: ['iPhone 12+', 'Galaxy S21+', 'iPad mini'],
  },

  md: {
    name: 'Medium',
    size: '768px',
    width: 768,
    description: 'Tablets (iPad, Android tablets)',
    devices: ['iPad (10.2")', 'iPad Air', 'Standard Android tablets'],
  },

  lg: {
    name: 'Large',
    size: '1024px',
    width: 1024,
    description: 'Large tablets, small laptops',
    devices: ['iPad Pro (11")', 'iPad Pro (12.9")', 'Small laptops'],
  },

  xl: {
    name: 'Extra Large',
    size: '1280px',
    width: 1280,
    description: 'Desktops, laptops (standard)',
    devices: ['MacBook Air', 'Windows laptops', 'Desktop monitors'],
  },

  '2xl': {
    name: '2X Large',
    size: '1536px',
    width: 1536,
    description: 'Large desktops, ultra-wide displays',
    devices: ['4K monitors', '27" displays', 'Ultra-wide displays'],
  },
} as const;

// ============================================================================
// MEDIA QUERY GENERATORS - CSS Helper Functions
// ============================================================================
export const mediaQueries = {
  // Minimum width (mobile-first)
  minXs: `(min-width: ${breakpoints.xs})`,
  minSm: `(min-width: ${breakpoints.sm})`,
  minMd: `(min-width: ${breakpoints.md})`,
  minLg: `(min-width: ${breakpoints.lg})`,
  minXl: `(min-width: ${breakpoints.xl})`,
  min2xl: `(min-width: ${breakpoints['2xl']})`,

  // Maximum width (max-width)
  maxXs: `(max-width: 319px)`,
  maxSm: `(max-width: 639px)`,
  maxMd: `(max-width: 767px)`,
  maxLg: `(max-width: 1023px)`,
  maxXl: `(max-width: 1279px)`,
  max2xl: `(max-width: 1535px)`,

  // Device orientation
  portrait: '(orientation: portrait)',
  landscape: '(orientation: landscape)',

  // Reduced motion (accessibility)
  prefersReducedMotion: '(prefers-reduced-motion: reduce)',
  prefersNoReducedMotion: '(prefers-reduced-motion: no-preference)',

  // Color scheme
  prefersDark: '(prefers-color-scheme: dark)',
  prefersLight: '(prefers-color-scheme: light)',

  // Device pixel ratio (retina displays)
  retina: '(-webkit-min-device-pixel-ratio: 2), (min-resolution: 192dpi)',

  // Hover capability
  canHover: '(hover: hover)',
  noHover: '(hover: none)',
} as const;

// ============================================================================
// TYPE EXPORTS
// ============================================================================
export type Breakpoint = keyof typeof breakpoints;
export type BreakpointName = 'xs' | 'sm' | 'md' | 'lg' | 'xl' | '2xl';
export type MediaQuery = keyof typeof mediaQueries;

// ============================================================================
// BREAKPOINT UTILITY FUNCTIONS
// ============================================================================
export const getBreakpoint = (key: Breakpoint): string => {
  return breakpoints[key];
};

export const getMediaQuery = (key: MediaQuery): string => {
  return mediaQueries[key];
};

/**
 * Creates a min-width media query
 */
export const createMinWidthQuery = (breakpoint: Breakpoint): string => {
  return `(min-width: ${breakpoints[breakpoint]})`;
};

/**
 * Creates a max-width media query
 */
export const createMaxWidthQuery = (breakpoint: Breakpoint): string => {
  const breakpointValue = parseInt(breakpoints[breakpoint]);
  return `(max-width: ${breakpointValue - 1}px)`;
};

/**
 * CSS-in-JS helper for responsive styles (nested media queries)
 */
export const createResponsiveStyle = (mobileStyle: object, smStyle?: object, mdStyle?: object, lgStyle?: object, xlStyle?: object): object => {
  return {
    ...mobileStyle,
    [`@media ${mediaQueries.minSm}`]: smStyle,
    [`@media ${mediaQueries.minMd}`]: mdStyle,
    [`@media ${mediaQueries.minLg}`]: lgStyle,
    [`@media ${mediaQueries.minXl}`]: xlStyle,
  };
};

/**
 * Tailwind CSS breakpoint prefix generator
 */
export const createTailwindBreakpoints = (): Record<string, string> => {
  return {
    'sm': breakpoints.sm,
    'md': breakpoints.md,
    'lg': breakpoints.lg,
    'xl': breakpoints.xl,
    '2xl': breakpoints['2xl'],
  };
};

// ============================================================================
// CONTAINER QUERY HELPERS (CSS Container Queries)
// ============================================================================
export const containerQueries = {
  minSmall: '(min-width: 400px)',
  minMedium: '(min-width: 600px)',
  minLarge: '(min-width: 800px)',
  minXLarge: '(min-width: 1000px)',
} as const;

/**
 * Creates a container query
 */
export const createContainerQuery = (condition: string): string => {
  return `@container ${condition}`;
};

// ============================================================================
// RESPONSIVE HELPER FUNCTIONS
// ============================================================================
export const isSmallScreen = (width: number): boolean => width < parseInt(breakpoints.sm);
export const isMediumScreen = (width: number): boolean => width >= parseInt(breakpoints.md) && width < parseInt(breakpoints.lg);
export const isLargeScreen = (width: number): boolean => width >= parseInt(breakpoints.lg);

/**
 * Gets the breakpoint name for a given width
 */
export const getBreakpointName = (width: number): BreakpointName => {
  if (width < parseInt(breakpoints.sm)) return 'xs';
  if (width < parseInt(breakpoints.md)) return 'sm';
  if (width < parseInt(breakpoints.lg)) return 'md';
  if (width < parseInt(breakpoints.xl)) return 'lg';
  if (width < parseInt(breakpoints['2xl'])) return 'xl';
  return '2xl';
};
