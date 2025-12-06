/**
 * @deprecated Use unified plotting standards from src/lib/unified-plot-standards.ts
 * This file is maintained for backward compatibility only
 */

import { 
  UNIFIED_PLOT_STANDARDS, 
  SCIENTIFIC_AXIS_STYLE, 
  SCIENTIFIC_COLORMAPS, 
  PHYSICS_AXIS_LABELS,
  PHYSICS_PLOT_CONFIG,
  PHYSICS_COLOR_SCHEMES
} from '@/lib/unified-plot-standards';

// Re-export with legacy names for backward compatibility
export const PHYSICS_PLOT_STANDARD = UNIFIED_PLOT_STANDARDS.layout;

export { SCIENTIFIC_AXIS_STYLE } from '@/lib/unified-plot-standards';

export const SCIENTIFIC_CONFIG = PHYSICS_PLOT_CONFIG;

export { PHYSICS_COLOR_SCHEMES } from '@/lib/unified-plot-standards';
export const COLOR_SCHEME = PHYSICS_COLOR_SCHEMES.bandStructure;

export { SCIENTIFIC_COLORMAPS } from '@/lib/unified-plot-standards';

export { PHYSICS_AXIS_LABELS } from '@/lib/unified-plot-standards';
export const AXIS_LABELS = PHYSICS_AXIS_LABELS;

// Backward compatibility aliases - use unified standards instead
export const GRAPHENE_PLOT_THEME = UNIFIED_PLOT_STANDARDS.layout;
export const AXIS_STYLE = SCIENTIFIC_AXIS_STYLE;
export const SURFACE_COLORS = SCIENTIFIC_COLORMAPS;