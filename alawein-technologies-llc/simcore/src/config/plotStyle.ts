/**
 * SimCore Unified Plot Configuration
 * 
 * This file composes existing plot configurations from src/lib/unified-plot-config.ts
 * and src/lib/unified-plot-standards.ts to provide a thin wrapper that ensures
 * consistent layout, interactions, and responsiveness across all plots.
 * 
 * Maintains compatibility with existing physics modules while providing
 * enhanced standardization and theming.
 */

import { UNIFIED_SCIENTIFIC_PLOT_SYSTEM as PLOT, PHYSICS_ENERGY_RANGES } from '@/lib/scientific-plot-system';
import { semanticColors, semanticTypography } from '@/theme/tokens';

// ===== ENHANCED PLOTLY CONFIGURATION =====
// Extends existing config with semantic token integration
export const unifiedPlotConfig = {
  plotly: {
    layout: {
      ...PLOT.layout,
      // Enhanced responsive behavior
      autosize: true,
      margin: { l: 50, r: 20, t: 40, b: 40 },
      
      // Semantic token integration for fonts
      font: {
        family: semanticTypography.fontBody,
        size: 14,
        color: semanticColors.textPrimary
      },
      
      // Enhanced axis styling with semantic tokens
      xaxis: {
        ...PLOT.layout.xaxis,
        title: {
          font: { 
            size: 16, 
            family: semanticTypography.fontBody, 
            color: semanticColors.textPrimary 
          },
          standoff: 25
        },
        tickfont: {
          size: 12,
          color: semanticColors.textSecondary
        }
      },
      
      yaxis: {
        ...PLOT.layout.yaxis,
        title: {
          font: { 
            size: 16, 
            family: semanticTypography.fontBody, 
            color: semanticColors.textPrimary 
          },
          standoff: 35
        },
        tickfont: {
          size: 12,
          color: semanticColors.textSecondary
        }
      },
      
      // Enhanced legend with semantic tokens
      legend: {
        ...PLOT.layout.legend,
        font: { 
          color: semanticColors.textPrimary,
          size: 12
        },
        bgcolor: 'rgba(0,0,0,0.3)',
        bordercolor: semanticColors.surfaceMuted,
      }
    },
    
    config: {
      ...PLOT.config,
      responsive: true,
      displaylogo: false,
      displayModeBar: false,
      // Enhanced interaction settings
      scrollZoom: true,
      doubleClick: 'reset+autosize',
      toImageButtonOptions: {
        format: 'svg',
        filename: 'SimCore_plot',
        scale: 1,
        width: 800,
        height: 600
      }
    }
  },
  
  // ===== THREE.JS CONFIGURATION =====
  // Standardized 3D scene settings with consistent lighting and camera
  three: {
    camera: { 
      fov: 60, 
      near: 0.1, 
      far: 1000, 
      position: [0, 0, 6] 
    },
    lighting: { 
      ambient: 0.4, 
      directional: { 
        intensity: 0.9, 
        position: [1, 1, 1] 
      } 
    },
    // Enhanced renderer settings for performance
    renderer: {
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
      // Cap device pixel ratio for performance on high-DPI displays
      pixelRatio: Math.min(window.devicePixelRatio, 2)
    }
  }
} as const;

// ===== RESPONSIVE PLOT CONFIGURATIONS =====
// Mobile-first responsive configurations for different screen sizes

export const responsivePlotConfig = {
  mobile: {
    layout: {
      ...unifiedPlotConfig.plotly.layout,
      margin: { l: 40, r: 15, t: 30, b: 40 },
      font: { size: 12 },
      xaxis: {
        ...unifiedPlotConfig.plotly.layout.xaxis,
        title: { font: { size: 14 } },
        tickfont: { size: 10 }
      },
      yaxis: {
        ...unifiedPlotConfig.plotly.layout.yaxis,
        title: { font: { size: 14 } },
        tickfont: { size: 10 }
      }
    },
    config: {
      ...unifiedPlotConfig.plotly.config,
      // Enhanced mobile interactions
      staticPlot: false,
      displayModeBar: false,
      scrollZoom: true,
      doubleClick: 'reset'
    }
  },
  
  tablet: {
    layout: {
      ...unifiedPlotConfig.plotly.layout,
      margin: { l: 50, r: 20, t: 35, b: 45 },
      font: { size: 13 }
    }
  },
  
  desktop: {
    layout: unifiedPlotConfig.plotly.layout,
    config: unifiedPlotConfig.plotly.config
  }
} as const;

// ===== PHYSICS DOMAIN CONFIGURATIONS =====
// Specialized configurations for different physics modules

export const physicsPlotConfigs = {
  // Quantum mechanics plots (TDSE, tunneling, etc.)
  quantum: {
    layout: {
      ...unifiedPlotConfig.plotly.layout,
      colorway: [
        semanticColors.accentQuantum,
        semanticColors.accentPhysics,
        semanticColors.accentField
      ],
      yaxis: {
        ...unifiedPlotConfig.plotly.layout.yaxis,
        range: [PHYSICS_ENERGY_RANGES.general.min, PHYSICS_ENERGY_RANGES.general.max]
      }
    },
    config: unifiedPlotConfig.plotly.config
  },
  
  // Band structure plots (graphene, MoS2, etc.)
  bandStructure: {
    layout: {
      ...unifiedPlotConfig.plotly.layout,
       colorway: [
         PLOT.colors.physics.valence,
         PLOT.colors.physics.conduction,
         PLOT.colors.physics.dos
       ],
      yaxis: {
        ...unifiedPlotConfig.plotly.layout.yaxis,
        title: { text: PLOT.axisLabels.energy },
        range: [PHYSICS_ENERGY_RANGES.graphene.min, PHYSICS_ENERGY_RANGES.graphene.max]
      }
    },
    config: unifiedPlotConfig.plotly.config
  },
  
  // Statistical physics plots (Ising, Boltzmann, etc.)
  statistical: {
    layout: {
      ...unifiedPlotConfig.plotly.layout,
      colorway: [
        semanticColors.accentStatistical,
        semanticColors.accentPhysics,
        semanticColors.accentField
      ]
    },
    config: unifiedPlotConfig.plotly.config
  },
  
  // 3D visualizations (crystal structures, LLG, etc.)
  threeDimensional: {
    camera: unifiedPlotConfig.three.camera,
    lighting: unifiedPlotConfig.three.lighting,
    controls: {
      enableDamping: true,
      dampingFactor: 0.25,
      enableZoom: true,
      autoRotate: false,
      maxPolarAngle: Math.PI
    }
  }
} as const;

// ===== UTILITY FUNCTIONS =====

/**
 * Get responsive plot configuration based on screen width
 */
export const getResponsivePlotConfig = (width: number) => {
  if (width < 640) return responsivePlotConfig.mobile;
  if (width < 1024) return responsivePlotConfig.tablet;
  return responsivePlotConfig.desktop;
};

/**
 * Create standardized plot layout with semantic tokens
 */
export const createStandardPlotLayout = (
  title: string,
  xLabel: string,
  yLabel: string,
  options: {
    energyRange?: { min: number; max: number };
    height?: number;
    domain?: 'quantum' | 'bandStructure' | 'statistical';
  } = {}
) => {
  const { energyRange, height = 400, domain = 'quantum' } = options;
  const domainConfig = physicsPlotConfigs[domain];
  
  // Only process 2D plot configurations (not threeDimensional)
  if (!('layout' in domainConfig)) {
    throw new Error(`Domain ${domain} does not support 2D plot layouts`);
  }
  
  return {
    ...domainConfig.layout,
    title: {
      text: title,
      font: { 
        size: 16, 
        family: semanticTypography.fontBody, 
        color: semanticColors.textPrimary 
      }
    },
    xaxis: {
      ...domainConfig.layout.xaxis,
      title: { text: xLabel }
    },
    yaxis: {
      ...domainConfig.layout.yaxis,
      title: { text: yLabel },
      ...(energyRange && { range: [energyRange.min, energyRange.max] })
    },
    height,
    hovermode: 'closest' as const
  };
};

/**
 * Create standardized Three.js scene configuration
 */
export const createStandardThreeConfig = (
  options: {
    autoRotate?: boolean;
    controls?: boolean;
    lighting?: 'standard' | 'enhanced';
  } = {}
) => {
  const { autoRotate = false, controls = true, lighting = 'standard' } = options;
  
  return {
    ...unifiedPlotConfig.three,
    controls: controls ? {
      enableDamping: true,
      dampingFactor: 0.25,
      enableZoom: true,
      autoRotate,
      autoRotateSpeed: 2.0,
      maxPolarAngle: Math.PI
    } : false,
    lighting: lighting === 'enhanced' ? {
      ambient: 0.6,
      directional: { 
        intensity: 1.2, 
        position: [1, 1, 1] 
      },
      additional: [
        { type: 'point', position: [10, 10, 10], intensity: 0.3 },
        { type: 'point', position: [-10, -10, 10], intensity: 0.3 }
      ]
    } : unifiedPlotConfig.three.lighting
  };
};

// Export the unified configuration as default
export default unifiedPlotConfig;

// Re-export existing utilities for backward compatibility
export {
  ENERGY_RANGES,
  LINE_STYLES,
  AXIS_LABELS,
  COLOR_SCHEMES,
  createPhysicsPlotLayout,
  createBandPlotData
} from '@/lib/unified-plot-config';