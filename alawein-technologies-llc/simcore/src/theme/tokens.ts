/**
 * SimCore Design System - Semantic Token Architecture
 * 
 * This file provides semantic design tokens that map to existing CSS variables
 * in index.css and tailwind.config.ts, ensuring consistent theming across
 * all components while maintaining dark/light mode compatibility.
 * 
 * Architecture: Primitive Tokens → Semantic Tokens → Component Tokens
 */

// ===== SEMANTIC COLOR TOKENS =====
// Maps to CSS variables defined in index.css
export const semanticColors = {
  // Brand & Primary
  brand: 'hsl(var(--primary))',
  brandMuted: 'hsl(var(--muted))',
  brandGlow: 'hsl(var(--primary-glow))',
  
  // Physics Domain Colors
  accentPhysics: 'hsl(var(--physics-valence))',
  accentQuantum: 'hsl(var(--physics-dos))',
  accentEnergy: 'hsl(var(--physics-conduction))',
  accentStatistical: 'hsl(var(--physics-fermi))',
  accentField: 'hsl(var(--accent))',
  
  // Surface & Background
  surface: 'hsl(var(--card))',
  surfaceMuted: 'hsl(var(--muted))',
  surfaceElevated: 'var(--semantic-surface-elevated)',
  surfaceGlass: 'var(--semantic-surface-glass)',
  
  // Text Hierarchy
  textPrimary: 'hsl(var(--foreground))',
  textSecondary: 'hsl(var(--muted-foreground))',
  textMuted: 'var(--semantic-text-muted)',
  textAccent: 'var(--semantic-text-accent)',
  
  // Interactive States
  interactive: 'hsl(var(--primary))',
  interactiveHover: 'hsl(var(--primary-glow))',
  interactiveMuted: 'hsl(var(--secondary))',
  
  // Status & Validation
  success: 'hsl(var(--primitive-green-500))',
  warning: 'hsl(var(--primitive-gold-500))',
  error: 'hsl(var(--destructive))',
  info: 'hsl(var(--accent))',
} as const;

// ===== SEMANTIC SPACING TOKENS =====
// Maps to spacing primitives in index.css
export const semanticSpacing = {
  // Semantic spacing based on content hierarchy
  moduleGap: 'var(--semantic-spacing-module)',      // 2rem (32px)
  sectionGap: 'var(--semantic-spacing-section)',    // 4rem (64px)
  inlineGap: 'var(--semantic-spacing-inline)',      // 1rem (16px)
  cardPadding: 'var(--semantic-spacing-card)',      // 1.5rem (24px)
  
  // Touch targets (mobile-first)
  touchTarget: 'var(--touch-target-min)',           // 44px
  
  // Responsive spacing
  mobileSpacing: 'var(--spacing-mobile)',           // 1rem
  tabletSpacing: 'var(--spacing-tablet)',           // 1.5rem
  desktopSpacing: 'var(--spacing-desktop)',         // 2rem
} as const;

// ===== SEMANTIC SHADOW TOKENS =====
// Maps to shadow primitives in index.css
export const semanticShadows = {
  elevation1: 'var(--primitive-shadow-elevation-1)',
  elevation2: 'var(--primitive-shadow-elevation-2)',
  
  // Physics domain shadows
  quantum: 'var(--shadow-quantum)',
  field: 'var(--shadow-field)', 
  elegant: 'var(--shadow-elegant)',
  glow: 'var(--shadow-glow)',
  
  // Component-specific shadows
  cardHover: 'var(--component-card-hover-glow)',
  buttonPhysics: 'var(--component-button-physics-shadow)',
} as const;

// ===== SEMANTIC TYPOGRAPHY TOKENS =====
// Maps to font variables in index.css and tailwind.config.ts
export const semanticTypography = {
  fontHeading: 'var(--font-heading)',
  fontBody: 'var(--font-sans)', 
  fontMonospace: 'var(--font-mono)',
  fontDisplay: 'var(--font-display)',
  
  // Scientific plot typography (fluid responsive)
  plotTitle: 'var(--font-size-plot-title)',
  plotSubtitle: 'var(--font-size-plot-subtitle)',
  axisLabel: 'var(--font-size-axis-label)',
  tickLabel: 'var(--font-size-tick-label)',
  legend: 'var(--font-size-legend)',
  annotation: 'var(--font-size-annotation)',
} as const;

// ===== SEMANTIC GRADIENT TOKENS =====
// Maps to gradient variables in index.css
export const semanticGradients = {
  primary: 'var(--gradient-primary)',
  physics: 'var(--gradient-physics)',
  quantum: 'var(--gradient-quantum)',
  field: 'var(--gradient-field)',
  cosmic: 'var(--gradient-cosmic)',
  wave: 'var(--gradient-wave)',
} as const;

// ===== SEMANTIC TRANSITION TOKENS =====
// Maps to transition variables in index.css
export const semanticTransitions = {
  quantum: 'var(--transition-quantum)',
  field: 'var(--transition-field)',
  smooth: 'var(--transition-smooth)',
} as const;

// ===== COMPONENT TOKEN FACTORIES =====
// Component-specific token generators using semantic tokens

export const buttonTokens = {
  primary: {
    bg: 'var(--component-button-primary-bg)',
    text: 'var(--component-button-primary-text)', 
    hover: 'var(--component-button-primary-hover)',
  },
  secondary: {
    bg: 'var(--component-button-secondary-bg)',
    text: 'var(--component-button-secondary-text)',
    border: 'var(--component-button-secondary-border)',
    hover: 'var(--component-button-secondary-hover)',
  },
  physics: {
    bg: 'var(--component-button-physics-bg)',
    text: 'var(--component-button-physics-text)',
    shadow: 'var(--component-button-physics-shadow)',
  },
} as const;

export const cardTokens = {
  background: 'var(--component-card-background)',
  border: 'var(--component-card-border)',
  padding: 'var(--component-card-padding)',
  shadow: 'var(--component-card-shadow)',
  hoverGlow: 'var(--component-card-hover-glow)',
} as const;

// ===== UNIFIED THEME OBJECT =====
// Complete theme token set for easy import and usage
export const theme = {
  colors: semanticColors,
  spacing: semanticSpacing,
  shadows: semanticShadows,
  typography: semanticTypography,
  gradients: semanticGradients,
  transitions: semanticTransitions,
  components: {
    button: buttonTokens,
    card: cardTokens,
  },
} as const;

// ===== UTILITY FUNCTIONS =====
// Helper functions for working with semantic tokens

/**
 * Get a semantic color token with optional opacity
 */
export const getSemanticColor = (color: keyof typeof semanticColors, opacity?: number) => {
  const baseColor = semanticColors[color];
  if (opacity !== undefined) {
    // For HSL colors, we need to convert to support opacity
    return `${baseColor} / ${opacity}`;
  }
  return baseColor;
};

/**
 * Create component variants using semantic tokens
 */
export const createComponentVariant = (
  baseTokens: Record<string, string>,
  overrides: Partial<Record<string, string>>
) => ({
  ...baseTokens,
  ...overrides,
});

/**
 * Get responsive spacing token based on breakpoint
 */
export const getResponsiveSpacing = (breakpoint: 'mobile' | 'tablet' | 'desktop') => {
  switch (breakpoint) {
    case 'mobile': return semanticSpacing.mobileSpacing;
    case 'tablet': return semanticSpacing.tabletSpacing;
    case 'desktop': return semanticSpacing.desktopSpacing;
    default: return semanticSpacing.mobileSpacing;
  }
};

// Export default theme for easy importing
export default theme;