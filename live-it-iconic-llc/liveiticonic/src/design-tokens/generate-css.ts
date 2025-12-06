// ============================================================================
// LIVE IT ICONIC - CSS VARIABLES GENERATOR
// Generates CSS Custom Properties from Design Tokens
// ============================================================================

import { colors, typography, spacing, sizes, shadows, borders, motion, breakpoints, zIndex } from './index';

// ============================================================================
// CSS VARIABLES GENERATOR
// ============================================================================
export function generateCSSVariables(): string {
  const cssLines: string[] = [':root {'];

  // ========================================================================
  // COLORS
  // ========================================================================
  cssLines.push('  /* Colors */');
  Object.entries(colors).forEach(([name, value]) => {
    if (typeof value === 'object') {
      // Color with shades (gold, success, etc.)
      Object.entries(value).forEach(([shade, hexValue]) => {
        cssLines.push(`  --lii-color-${name}-${shade}: ${hexValue};`);
      });
    } else if (typeof value === 'string') {
      // Simple color value
      cssLines.push(`  --lii-color-${name}: ${value};`);
    }
  });

  // ========================================================================
  // TYPOGRAPHY
  // ========================================================================
  cssLines.push('\n  /* Typography - Font Families */');
  Object.entries(typography.fontFamilies).forEach(([key, value]) => {
    cssLines.push(`  --lii-font-family-${key}: ${value};`);
  });

  cssLines.push('\n  /* Typography - Font Sizes */');
  Object.entries(typography.fontSizes).forEach(([key, value]) => {
    cssLines.push(`  --lii-font-size-${key}: ${value};`);
  });

  cssLines.push('\n  /* Typography - Font Weights */');
  Object.entries(typography.fontWeights).forEach(([key, value]) => {
    cssLines.push(`  --lii-font-weight-${key}: ${value};`);
  });

  cssLines.push('\n  /* Typography - Line Heights */');
  Object.entries(typography.lineHeights).forEach(([key, value]) => {
    cssLines.push(`  --lii-line-height-${key}: ${value};`);
  });

  cssLines.push('\n  /* Typography - Letter Spacing */');
  Object.entries(typography.letterSpacing).forEach(([key, value]) => {
    cssLines.push(`  --lii-letter-spacing-${key}: ${value};`);
  });

  // ========================================================================
  // SPACING
  // ========================================================================
  cssLines.push('\n  /* Spacing */');
  Object.entries(spacing).forEach(([key, value]) => {
    cssLines.push(`  --lii-spacing-${key}: ${value};`);
  });

  cssLines.push('\n  /* Sizes */');
  Object.entries(sizes).forEach(([key, value]) => {
    cssLines.push(`  --lii-size-${key}: ${value};`);
  });

  // ========================================================================
  // SHADOWS
  // ========================================================================
  cssLines.push('\n  /* Shadows */');
  Object.entries(shadows).forEach(([key, value]) => {
    cssLines.push(`  --lii-shadow-${key}: ${value};`);
  });

  // ========================================================================
  // BORDERS
  // ========================================================================
  cssLines.push('\n  /* Border Radius */');
  Object.entries(borders.radii).forEach(([key, value]) => {
    cssLines.push(`  --lii-radius-${key}: ${value};`);
  });

  cssLines.push('\n  /* Border Width */');
  Object.entries(borders.widths).forEach(([key, value]) => {
    cssLines.push(`  --lii-border-width-${key}: ${value};`);
  });

  // ========================================================================
  // MOTION
  // ========================================================================
  cssLines.push('\n  /* Motion - Durations */');
  Object.entries(motion.durations).forEach(([key, value]) => {
    cssLines.push(`  --lii-duration-${key}: ${value};`);
  });

  cssLines.push('\n  /* Motion - Easings */');
  Object.entries(motion.easings).forEach(([key, value]) => {
    cssLines.push(`  --lii-easing-${key}: ${value};`);
  });

  // ========================================================================
  // BREAKPOINTS
  // ========================================================================
  cssLines.push('\n  /* Breakpoints */');
  Object.entries(breakpoints).forEach(([key, value]) => {
    cssLines.push(`  --lii-breakpoint-${key}: ${value};`);
  });

  // ========================================================================
  // Z-INDEX
  // ========================================================================
  cssLines.push('\n  /* Z-Index */');
  Object.entries(zIndex).forEach(([key, value]) => {
    cssLines.push(`  --lii-z-${key}: ${value};`);
  });

  cssLines.push('}');

  return cssLines.join('\n');
}

// ============================================================================
// MEDIA QUERY VARIABLES GENERATOR
// ============================================================================
export function generateMediaQueryVariables(): string {
  const cssLines: string[] = [];

  // ========================================================================
  // SMALL SCREENS
  // ========================================================================
  cssLines.push(`@media (min-width: ${breakpoints.sm}) {`);
  cssLines.push('  :root {');
  cssLines.push('    /* Small screen adjustments can go here */');
  cssLines.push('  }');
  cssLines.push('}');

  // ========================================================================
  // MEDIUM SCREENS
  // ========================================================================
  cssLines.push(`\n@media (min-width: ${breakpoints.md}) {`);
  cssLines.push('  :root {');
  cssLines.push('    /* Medium screen adjustments can go here */');
  cssLines.push('  }');
  cssLines.push('}');

  // ========================================================================
  // LARGE SCREENS
  // ========================================================================
  cssLines.push(`\n@media (min-width: ${breakpoints.lg}) {`);
  cssLines.push('  :root {');
  cssLines.push('    /* Large screen adjustments can go here */');
  cssLines.push('  }');
  cssLines.push('}');

  // ========================================================================
  // EXTRA LARGE SCREENS
  // ========================================================================
  cssLines.push(`\n@media (min-width: ${breakpoints.xl}) {`);
  cssLines.push('  :root {');
  cssLines.push('    /* Extra large screen adjustments can go here */');
  cssLines.push('  }');
  cssLines.push('}');

  // ========================================================================
  // 2XL SCREENS
  // ========================================================================
  cssLines.push(`\n@media (min-width: ${breakpoints['2xl']}) {`);
  cssLines.push('  :root {');
  cssLines.push('    /* 2XL screen adjustments can go here */');
  cssLines.push('  }');
  cssLines.push('}');

  return cssLines.join('\n');
}

// ============================================================================
// DARK MODE VARIABLES GENERATOR
// ============================================================================
export function generateDarkModeVariables(): string {
  const cssLines: string[] = [
    '@media (prefers-color-scheme: dark) {',
    '  :root {',
    '    /* Dark mode color overrides */\n',
  ];

  // Add dark mode specific color adjustments
  cssLines.push(`    --lii-color-bg-primary: ${colors.charcoal[300]};`);
  cssLines.push(`    --lii-color-bg-secondary: ${colors.charcoal[400]};`);
  cssLines.push(`    --lii-color-text-primary: ${colors.cloud[200]};`);
  cssLines.push(`    --lii-color-text-secondary: ${colors.ash};`);

  cssLines.push('  }');
  cssLines.push('}');

  return cssLines.join('\n');
}

// ============================================================================
// SCOPED VARIABLES GENERATOR
// ============================================================================
export function generateScopedVariables(className: string): string {
  const cssLines: string[] = [`.${className} {`];

  // Add all variables scoped to the class
  Object.entries(colors.gold).forEach(([shade, value]) => {
    cssLines.push(`  --lii-color-gold-${shade}: ${value};`);
  });

  cssLines.push('}');

  return cssLines.join('\n');
}

// ============================================================================
// UTILITY CLASSES GENERATOR
// ============================================================================
export function generateUtilityClasses(): string {
  const cssLines: string[] = [];

  // ========================================================================
  // Color Utility Classes
  // ========================================================================
  cssLines.push('/* Color Utilities */');
  Object.entries(colors.gold).forEach(([shade, value]) => {
    cssLines.push(`.text-gold-${shade} { color: ${value}; }`);
    cssLines.push(`.bg-gold-${shade} { background-color: ${value}; }`);
    cssLines.push(`.border-gold-${shade} { border-color: ${value}; }`);
  });

  // ========================================================================
  // Spacing Utility Classes
  // ========================================================================
  cssLines.push('\n/* Spacing Utilities */');
  Object.entries(spacing).forEach(([key, value]) => {
    cssLines.push(`.p-${key} { padding: ${value}; }`);
    cssLines.push(`.m-${key} { margin: ${value}; }`);
  });

  // ========================================================================
  // Radius Utility Classes
  // ========================================================================
  cssLines.push('\n/* Border Radius Utilities */');
  Object.entries(borders.radii).forEach(([key, value]) => {
    cssLines.push(`.rounded-${key} { border-radius: ${value}; }`);
  });

  // ========================================================================
  // Shadow Utility Classes
  // ========================================================================
  cssLines.push('\n/* Shadow Utilities */');
  Object.entries(shadows).forEach(([key, value]) => {
    cssLines.push(`.shadow-${key} { box-shadow: ${value}; }`);
  });

  return cssLines.join('\n');
}

// ============================================================================
// EXPORT AS JSON
// ============================================================================
export function exportTokensAsJSON(): string {
  const tokenObject = {
    colors,
    typography,
    spacing,
    sizes,
    shadows,
    borders,
    motion,
    breakpoints,
    zIndex,
  };

  return JSON.stringify(tokenObject, null, 2);
}

// ============================================================================
// EXPORT AS JAVASCRIPT MODULE
// ============================================================================
export function exportTokensAsJS(): string {
  return `
export const tokens = ${JSON.stringify(
    {
      colors,
      typography,
      spacing,
      sizes,
      shadows,
      borders,
      motion,
      breakpoints,
      zIndex,
    },
    null,
    2
  )};
`;
}

// ============================================================================
// BATCH GENERATOR - All files at once
// ============================================================================
export interface GeneratedFiles {
  'tokens.css': string;
  'tokens-media.css': string;
  'tokens-dark.css': string;
  'utilities.css': string;
  'tokens.json': string;
  'tokens.js': string;
}

export function generateAllFiles(): GeneratedFiles {
  return {
    'tokens.css': generateCSSVariables(),
    'tokens-media.css': generateMediaQueryVariables(),
    'tokens-dark.css': generateDarkModeVariables(),
    'utilities.css': generateUtilityClasses(),
    'tokens.json': exportTokensAsJSON(),
    'tokens.js': exportTokensAsJS(),
  };
}

// ============================================================================
// FILE WRITER HELPER (for Node.js environments)
// ============================================================================
export async function writeGeneratedFiles(outputDir: string): Promise<void> {
  if (typeof window !== 'undefined') {
    throw new Error('This function can only be called in Node.js environments');
  }

  const fs = await import('fs').then((m) => m.promises);
  const path = await import('path');

  const files = generateAllFiles();

  for (const [filename, content] of Object.entries(files)) {
    const filePath = path.join(outputDir, filename);
    await fs.writeFile(filePath, content, 'utf-8');
    console.log(`Generated: ${filename}`);
  }
}

// ============================================================================
// CLI USAGE EXAMPLE
// ============================================================================
export const cliUsage = `
# CSS Variables Generator CLI Usage

## Generate all token files
npx ts-node -e "import { generateAllFiles } from '@/design-tokens/generate-css'; const files = generateAllFiles(); Object.entries(files).forEach(([name, content]) => console.log(\`=== \${name} ===\n\${content}\`));"

## Generate CSS variables only
npx ts-node -e "import { generateCSSVariables } from '@/design-tokens/generate-css'; console.log(generateCSSVariables());"

## Export as JSON
npx ts-node -e "import { exportTokensAsJSON } from '@/design-tokens/generate-css'; console.log(exportTokensAsJSON());"
`;
