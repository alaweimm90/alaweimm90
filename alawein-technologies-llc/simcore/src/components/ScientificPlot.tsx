import React, { useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Download, ZoomIn, ZoomOut, RotateCcw, Maximize2, Eye, EyeOff } from 'lucide-react';
import * as THREE from 'three';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Line, Html } from '@react-three/drei';

interface ScientificPlotProps {
  title: string;
  subtitle?: string;
  data: any;
  plotType: '1d' | '2d' | '3d' | 'surface' | 'vector' | 'heatmap';
  xLabel?: string;
  yLabel?: string;
  zLabel?: string;
  xUnit?: string;
  yUnit?: string;
  zUnit?: string;
  colormap?: 'viridis' | 'plasma' | 'inferno' | 'turbo' | 'coolwarm';
  showGrid?: boolean;
  showAxes?: boolean;
  showLegend?: boolean;
  annotations?: Array<{x: number, y: number, z?: number, text: string}>;
  onExport?: (format: 'png' | 'svg' | 'data') => void;
  className?: string;
}

// Color mapping functions
const colormaps = {
  viridis: (t: number) => {
    // Viridis colormap approximation
    const r = 0.267004 + t * (0.993248 - 0.267004);
    const g = 0.004874 + t * (0.984314 - 0.004874);  
    const b = 0.329415 + t * (0.143936 - 0.329415);
    return new THREE.Color(r, g, b);
  },
  plasma: (t: number) => {
    const r = 0.050383 + t * (0.940015 - 0.050383);
    const g = 0.029803 + t * (0.975158 - 0.029803);
    const b = 0.527975 + t * (0.131326 - 0.527975);
    return new THREE.Color(r, g, b);
  },
  inferno: (t: number) => {
    const r = 0.001462 + t * (0.988362 - 0.001462);
    const g = 0.000466 + t * (0.998364 - 0.000466);
    const b = 0.013866 + t * (0.644924 - 0.013866);
    return new THREE.Color(r, g, b);
  },
  turbo: (t: number) => {
    // Turbo colormap (Google's improved jet)
    const r = Math.sin(Math.PI * (0.5 * t + 0.5));
    const g = Math.sin(Math.PI * (0.5 * t + 1.0));
    const b = Math.sin(Math.PI * (0.5 * t + 1.5));
    return new THREE.Color(r, g, b);
  },
  coolwarm: (t: number) => {
    const r = 0.230 + t * (0.706 - 0.230);
    const g = 0.299 + t * (0.016 - 0.299);
    const b = 0.754 + t * (0.150 - 0.754);
    return new THREE.Color(r, g, b);
  }
};

// Axis component for 3D plots
function Axes({ size = 5, showLabels = true, xLabel = "x", yLabel = "y", zLabel = "z" }: any) {
  return (
    <group>
      {/* X Axis - Red */}
      <Line
        points={[[-size, 0, 0], [size, 0, 0]]}
        color="red"
        lineWidth={2}
      />
      {showLabels && (
        <Text
          position={[size + 0.5, 0, 0]}
          fontSize={0.5}
          color="red"
          anchorX="left"
          font="/fonts/Inter-Medium.woff"
        >
          {xLabel}
        </Text>
      )}
      
      {/* Y Axis - Green */}
      <Line
        points={[[0, -size, 0], [0, size, 0]]}
        color="green"
        lineWidth={2}
      />
      {showLabels && (
        <Text
          position={[0, size + 0.5, 0]}
          fontSize={0.5}
          color="green"
          anchorY="bottom"
          font="/fonts/Inter-Medium.woff"
        >
          {yLabel}
        </Text>
      )}
      
      {/* Z Axis - Blue */}
      <Line
        points={[[0, 0, -size], [0, 0, size]]}
        color="blue"
        lineWidth={2}
      />
      {showLabels && (
        <Text
          position={[0, 0, size + 0.5]}
          fontSize={0.5}
          color="blue"
          anchorX="center"
          font="/fonts/Inter-Medium.woff"
        >
          {zLabel}
        </Text>
      )}
    </group>
  );
}

// Grid component
function Grid({ size = 10, divisions = 10, colorCenter = "#888888", colorGrid = "#444444" }: any) {
  return (
    <gridHelper
      args={[size, divisions, colorCenter, colorGrid]}
      position={[0, 0, 0]}
    />
  );
}

// 3D Surface plot component
function Surface3D({ data, colormap = 'viridis' }: any) {
  const meshRef = useRef<THREE.Mesh>(null);
  const colormapFunc = colormaps[colormap];
  
  useEffect(() => {
    if (!meshRef.current || !data) return;
    
    const geometry = new THREE.PlaneGeometry(10, 10, data.nx - 1, data.ny - 1);
    const positions = geometry.attributes.position.array as Float32Array;
    const colors = new Float32Array(positions.length);
    
    // Update positions and colors based on data
    for (let i = 0; i < data.nx * data.ny; i++) {
      const x = i % data.nx;
      const y = Math.floor(i / data.nx);
      const idx = i * 3;
      
      // Set Z position from data
      positions[idx + 2] = data.values[i] || 0;
      
      // Set colors based on value
      const normalizedValue = (data.values[i] - data.min) / (data.max - data.min);
      const color = colormapFunc(normalizedValue);
      colors[idx] = color.r;
      colors[idx + 1] = color.g;
      colors[idx + 2] = color.b;
    }
    
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.attributes.position.needsUpdate = true;
    
  }, [data, colormap, colormapFunc]);
  
  return (
    <mesh ref={meshRef} rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={[10, 10, data?.nx - 1 || 50, data?.ny - 1 || 50]} />
      <meshStandardMaterial vertexColors wireframe={false} />
    </mesh>
  );
}

// Scientific Plot Component
export const ScientificPlot: React.FC<ScientificPlotProps> = ({
  title,
  subtitle,
  data,
  plotType,
  xLabel = "x",
  yLabel = "y", 
  zLabel = "z",
  xUnit = "",
  yUnit = "",
  zUnit = "",
  colormap = 'viridis',
  showGrid = true,
  showAxes = true,
  showLegend = true,
  annotations = [],
  onExport,
  className = ""
}) => {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showControls, setShowControls] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const handleExport = (format: 'png' | 'svg' | 'data') => {
    if (format === 'png' && canvasRef.current) {
      const link = document.createElement('a');
      link.download = `${title.replace(/\s+/g, '_').toLowerCase()}.png`;
      link.href = canvasRef.current.toDataURL();
      link.click();
    } else if (format === 'data') {
      const jsonData = JSON.stringify(data, null, 2);
      const blob = new Blob([jsonData], { type: 'application/json' });
      const link = document.createElement('a');
      link.download = `${title.replace(/\s+/g, '_').toLowerCase()}_data.json`;
      link.href = URL.createObjectURL(blob);
      link.click();
    }
    onExport?.(format);
  };

  const formatLabel = (label: string, unit: string) => {
    return unit ? `${label} (${unit})` : label;
  };

  return (
    <Card className={`relative overflow-hidden ${className} ${isFullscreen ? 'fixed inset-0 z-50' : ''}`}>
      <div className="absolute top-4 left-4 z-10">
        <div className="bg-background/90 backdrop-blur rounded-lg p-3 space-y-1">
          <h3 className="font-serif text-lg font-semibold text-foreground">{title}</h3>
          {subtitle && (
            <p className="text-sm text-muted-foreground font-mono">{subtitle}</p>
          )}
        </div>
      </div>

      {/* Plot Controls */}
      <div className="absolute top-[--spacing-md] right-[--spacing-md] z-10 flex gap-[--spacing-xs]">
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setShowControls(!showControls)}
          className="bg-background/90 backdrop-blur min-h-[--touch-target-min] min-w-[--touch-target-min]"
        >
          {showControls ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
        </Button>
        
        {showControls && (
          <>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="bg-background/90 backdrop-blur min-h-[--touch-target-min] min-w-[--touch-target-min]"
            >
              <Maximize2 className="w-4 h-4" />
            </Button>
            
            <Button
              variant="ghost"
              size="icon"
              onClick={() => handleExport('png')}
              className="bg-background/90 backdrop-blur min-h-[--touch-target-min] min-w-[--touch-target-min]"
            >
              <Download className="w-4 h-4" />
            </Button>
          </>
        )}
      </div>

      {/* Axis Labels */}
      {showControls && (
        <div className="absolute bottom-[--spacing-md] left-[--spacing-md] z-10">
          <div className="bg-background/90 backdrop-blur rounded-lg p-[--spacing-md]">
            <div className="flex gap-[--spacing-md] text-sm font-mono">
              <Badge variant="outline" className="bg-semantic-error/10 text-semantic-error border-semantic-error/20">
                {formatLabel(xLabel, xUnit)}
              </Badge>
              <Badge variant="outline" className="bg-semantic-success/10 text-semantic-success border-semantic-success/20">
                {formatLabel(yLabel, yUnit)}
              </Badge>
              {plotType === '3d' && (
                <Badge variant="outline" className="bg-semantic-info/10 text-semantic-info border-semantic-info/20">
                  {formatLabel(zLabel, zUnit)}
                </Badge>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Export Controls */}
      {showControls && onExport && (
        <div className="absolute bottom-[--spacing-md] right-[--spacing-md] z-10">
          <div className="bg-background/90 backdrop-blur rounded-lg p-[--spacing-xs] flex gap-[--spacing-xs]">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => handleExport('png')}
              className="text-xs min-h-[--touch-target-min]"
            >
              PNG
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => handleExport('data')}
              className="text-xs min-h-[--touch-target-min]"
            >
              Data
            </Button>
          </div>
        </div>
      )}

      {/* 3D Canvas */}
      <div 
        className={`w-full ${isFullscreen ? 'h-screen' : 'h-96'}`}
        role="img"
        aria-label={`Interactive scientific plot: ${title}. ${subtitle || ''}. Use mouse or touch to navigate the 3D visualization.`}
        tabIndex={0}
      >
        <Canvas
          ref={canvasRef}
          camera={{ position: [10, 10, 10], fov: 45 }}
          style={{ background: 'transparent' }}
        >
          <ambientLight intensity={0.4} />
          <pointLight position={[10, 10, 10]} intensity={0.8} />
          <directionalLight position={[-10, -10, -5]} intensity={0.5} />

          {/* Render based on plot type */}
          {plotType === '3d' || plotType === 'surface' ? (
            <>
              {showAxes && (
                <Axes 
                  size={5} 
                  xLabel={formatLabel(xLabel, xUnit)}
                  yLabel={formatLabel(yLabel, yUnit)}
                  zLabel={formatLabel(zLabel, zUnit)}
                />
              )}
              {showGrid && <Grid size={10} divisions={10} />}
              {plotType === 'surface' && data && (
                <Surface3D data={data} colormap={colormap} />
              )}
            </>
          ) : null}

          {/* Annotations */}
          {annotations.map((annotation, index) => (
            <Html
              key={index}
              position={[annotation.x, annotation.y, annotation.z || 0]}
              className="pointer-events-none"
            >
              <div className="bg-background/90 backdrop-blur rounded px-2 py-1 text-xs font-mono">
                {annotation.text}
              </div>
            </Html>
          ))}

          <OrbitControls 
            enableZoom 
            enablePan 
            enableRotate 
            enableDamping={true}
            dampingFactor={0.05}
            minDistance={2}
            maxDistance={20}
          />
        </Canvas>
        
        {/* Screen reader description of current data */}
        <div className="sr-only">
          Scientific plot showing {plotType} visualization. 
          {data && `Data range available for analysis.`}
          X-axis: {xLabel} {xUnit && `(${xUnit})`}, Y-axis: {yLabel} {yUnit && `(${yUnit})`}, Z-axis: {zLabel} {zUnit && `(${zUnit})`}.
        </div>
      </div>
    </Card>
  );
};

export default ScientificPlot;