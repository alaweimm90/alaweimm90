/**
 * Unified Plot Wrapper Component
 * 
 * Standardizes all scientific plots across SimCore with consistent theming,
 * responsive behavior, and accessibility features.
 */

import { ReactNode, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { unifiedPlotConfig, getResponsivePlotConfig, createStandardPlotLayout } from '@/config/plotStyle';
import { useResponsive } from '@/hooks/use-responsive';
import { semanticColors } from '@/theme/tokens';

interface UnifiedPlotlyWrapperProps {
  data: any[];
  title: string;
  xLabel: string;
  yLabel: string;
  domain?: 'quantum' | 'bandStructure' | 'statistical';
  energyRange?: { min: number; max: number };
  height?: number;
  className?: string;
  onUpdate?: (data: any) => void;
  useResponsive?: boolean;
}

export function UnifiedPlotlyWrapper({
  data,
  title,
  xLabel,
  yLabel,
  domain = 'quantum',
  energyRange,
  height,
  className = '',
  onUpdate,
  useResponsive: useResponsivePlot = true
}: UnifiedPlotlyWrapperProps) {
  const { isMobile, isTablet, width: screenWidth } = useResponsive();
  
  const layout = useMemo(() => {
    const baseLayout = createStandardPlotLayout(title, xLabel, yLabel, {
      energyRange,
      height: height || (isMobile ? 300 : isTablet ? 350 : 400),
      domain
    });
    
    if (useResponsivePlot) {
      const responsiveConfig = getResponsivePlotConfig(screenWidth);
      return {
        ...baseLayout,
        ...responsiveConfig.layout,
        // Ensure responsive margins
        margin: isMobile 
          ? { l: 40, r: 15, t: 40, b: 50 }
          : isTablet
          ? { l: 50, r: 20, t: 45, b: 55 }
          : { l: 60, r: 25, t: 50, b: 60 }
      };
    }
    
    return baseLayout;
  }, [title, xLabel, yLabel, domain, energyRange, height, isMobile, isTablet, screenWidth, useResponsivePlot]);
  
  const config = useMemo(() => {
    const baseConfig = unifiedPlotConfig.plotly.config;
    
    if (useResponsivePlot) {
      return {
        ...baseConfig,
        // Mobile-specific optimizations
        scrollZoom: !isMobile, // Disable scroll zoom on mobile
        doubleClick: isMobile ? 'reset' : 'reset+autosize'
      };
    }
    
    return baseConfig;
  }, [isMobile, screenWidth, useResponsivePlot]);
  
  return (
    <div className={`w-full ${className}`}>
      <Plot
        data={data}
        layout={layout}
        config={config}
        className="w-full h-full"
        useResizeHandler={true}
        onUpdate={onUpdate}
      />
    </div>
  );
}

interface UnifiedThreeWrapperProps {
  children: ReactNode;
  title?: string;
  height?: number;
  className?: string;
  cameraPosition?: [number, number, number];
  autoRotate?: boolean;
  controls?: boolean;
}

export function UnifiedThreeWrapper({
  children,
  title,
  height,
  className = '',
  cameraPosition = [0, 0, 6],
  autoRotate = false,
  controls = true
}: UnifiedThreeWrapperProps) {
  const { isMobile, isTablet } = useResponsive();
  
  const canvasHeight = height || (isMobile ? 300 : isTablet ? 350 : 400);
  
  const threeConfig = useMemo(() => ({
    camera: { 
      fov: 60, 
      near: 0.1, 
      far: 1000, 
      position: cameraPosition 
    },
    controls: {
      enableDamping: true,
      dampingFactor: 0.25,
      enableZoom: true,
      autoRotate,
      autoRotateSpeed: 2.0,
      maxPolarAngle: Math.PI,
      // Touch-friendly settings for mobile
      enablePan: !isMobile,
      panSpeed: isMobile ? 2.0 : 1.0,
      rotateSpeed: isMobile ? 1.0 : 0.5
    }
  }), [cameraPosition, autoRotate, isMobile]);
  
  return (
    <div 
      className={`w-full ${className}`}
      style={{ height: `${canvasHeight}px` }}
      role="img"
      aria-label={title || "3D Scientific Visualization"}
    >
      <Canvas
        camera={threeConfig.camera}
        gl={{ 
          antialias: !isMobile, // Disable antialiasing on mobile for performance
          alpha: true,
          powerPreference: 'high-performance'
        }}
        style={{ 
          background: 'transparent',
          touchAction: 'manipulation' // Optimize touch interactions
        }}
      >
        {/* Lighting optimized for scientific visualization */}
        <ambientLight intensity={0.4} />
        <directionalLight 
          position={[1, 1, 1]} 
          intensity={0.9} 
          castShadow={!isMobile} // Disable shadows on mobile
        />
        
        {/* OrbitControls with responsive settings */}
        {controls && (
          <OrbitControls 
            {...threeConfig.controls}
            makeDefault
          />
        )}
        
        {children}
      </Canvas>
    </div>
  );
}

// Re-export for backward compatibility
export { UnifiedPlotlyWrapper as StandardizedPlot };
export { UnifiedThreeWrapper as Standardized3DScene };