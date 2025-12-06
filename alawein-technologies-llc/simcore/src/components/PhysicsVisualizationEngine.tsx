import React, { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stats, Environment, AdaptiveDpr, AdaptiveEvents } from '@react-three/drei';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Download, Maximize2, RotateCcw, Settings } from 'lucide-react';
import { useResponsive } from '@/hooks/use-responsive';
import { UnifiedPlotlyWrapper } from '@/components/UnifiedPlotWrapper';
import { createStandardThreeConfig, createStandardPlotLayout } from '@/config/plotStyle';
import { theme } from '@/theme/tokens';
import { cn } from '@/lib/utils';

interface VisualizationEngineProps {
  title: string;
  description?: string;
  variant?: 'quantum' | 'statistical' | 'fields' | 'crystal';
  plotData?: any;
  plotLayout?: any;
  threeScene?: React.ReactNode;
  showStats?: boolean;
  enableExport?: boolean;
  className?: string;
}

interface Plot3DProps {
  data: any[];
  layout?: any;
  config?: any;
  variant?: 'quantum' | 'statistical' | 'fields';
  responsive?: boolean;
}

interface Scene3DProps {
  children: React.ReactNode;
  variant?: 'quantum' | 'statistical' | 'fields' | 'crystal';
  showStats?: boolean;
  controls?: boolean;
  environment?: 'studio' | 'warehouse' | 'dark' | 'none';
}

export const PhysicsVisualizationEngine: React.FC<VisualizationEngineProps> = ({
  title,
  description,
  variant = 'quantum',
  plotData,
  plotLayout,
  threeScene,
  showStats = false,
  enableExport = true,
  className
}) => {
  const { isMobile, isTablet } = useResponsive();
  
  const variantConfig = {
    quantum: {
      color: 'accentQuantum',
      gradient: 'gradient-quantum',
      badge: 'Quantum'
    },
    statistical: {
      color: 'accentStatistical', 
      gradient: 'gradient-field',
      badge: 'Statistical'
    },
    fields: {
      color: 'accentField',
      gradient: 'gradient-field', 
      badge: 'Fields'
    },
    crystal: {
      color: 'accentPhysics',
      gradient: 'gradient-primary',
      badge: 'Crystal'
    }
  };
  
  const config = variantConfig[variant];
  
  const standardLayout = useMemo(() => 
    plotLayout || createStandardPlotLayout(
      title,
      'X Axis',
      'Y Axis',
      { domain: variant === 'quantum' ? 'quantum' : variant === 'statistical' ? 'statistical' : variant === 'fields' ? 'bandStructure' : 'quantum' }
    ), 
    [plotLayout, title, variant]
  );

  const handleExport = () => {
    // Export functionality - could export plot data, screenshot, etc.
    console.log('Exporting visualization data...');
  };

  return (
    <Card className={cn(
      'w-full transition-all duration-300',
      'border-muted/20 bg-card',
      'hover:shadow-elegant',
      className
    )}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <CardTitle className={cn(
                'text-textPrimary',
                isMobile ? 'text-lg' : 'text-xl'
              )}>
                {title}
              </CardTitle>
              <Badge 
                variant="outline" 
                className={cn(
                  'text-xs',
                  `border-${config.color}/30 text-${config.color}`
                )}
              >
                {config.badge}
              </Badge>
            </div>
            {description && (
              <p className="text-sm text-textSecondary">
                {description}
              </p>
            )}
          </div>
          
          {enableExport && (
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size={isMobile ? 'sm' : 'default'}
                onClick={handleExport}
                className="gap-2"
              >
                <Download className="w-4 h-4" />
                {!isMobile && 'Export'}
              </Button>
            </div>
          )}
        </div>
      </CardHeader>
      
      <CardContent className="p-0">
        <div className={cn(
          'relative w-full',
          isMobile ? 'min-h-[300px]' : isTablet ? 'min-h-[400px]' : 'min-h-[500px]'
        )}>
          {plotData && (
            <UnifiedPlotlyWrapper
              data={plotData}
              title={title}
              xLabel="X Axis"
              yLabel="Y Axis"
              domain={variant === 'quantum' ? 'quantum' : variant === 'statistical' ? 'statistical' : 'bandStructure'}
              className="w-full h-full"
            />
          )}
          
          {threeScene && (
            <Scene3D
              variant={variant}
              showStats={showStats}
              controls={true}
              environment={variant === 'crystal' ? 'studio' : 'dark'}
            >
              {threeScene}
            </Scene3D>
          )}
          
          {!plotData && !threeScene && (
            <div className="flex items-center justify-center h-[400px] text-textSecondary">
              <div className="text-center">
                <Settings className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>Configure parameters to generate visualization</p>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export const Plot3D: React.FC<Plot3DProps> = ({
  data,
  layout,
  config,
  variant = 'quantum',
  responsive = true
}) => {
  return (
    <UnifiedPlotlyWrapper
      data={data}
      title="Physics Simulation"
      xLabel="X"
      yLabel="Y"
      domain={variant === 'fields' ? 'bandStructure' : variant as 'quantum' | 'statistical'}
      className="w-full h-full"
    />
  );
};

export const Scene3D: React.FC<Scene3DProps> = ({
  children,
  variant = 'quantum',
  showStats = false,
  controls = true,
  environment = 'studio'
}) => {
  const { isMobile } = useResponsive();
  const threeConfig = createStandardThreeConfig({
    controls,
    lighting: variant === 'crystal' ? 'enhanced' : 'standard'
  });

  return (
    <div className="w-full h-full relative">
      <Canvas
        camera={threeConfig.camera}
        dpr={isMobile ? [0.5, 1] : [1, 2]}
        performance={{ min: 0.5 }}
        
      >
        {/* Adaptive performance */}
        <AdaptiveDpr pixelated />
        <AdaptiveEvents />
        
        {/* Lighting based on variant */}
        <ambientLight intensity={0.4} />
        <directionalLight position={[1, 1, 1]} intensity={0.9} />
        
        {/* Environment */}
        {environment !== 'none' && (
          <Environment preset={environment === 'dark' ? 'night' : environment} />
        )}
        
        {/* Controls */}
        {controls && (
          <OrbitControls
            enablePan={!isMobile}
            enableZoom={true}
            enableRotate={true}
            maxPolarAngle={Math.PI}
            autoRotate={false}
            autoRotateSpeed={0.5}
          />
        )}
        
        {/* Scene content */}
        {children}
        
        {/* Performance stats */}
        {showStats && <Stats />}
      </Canvas>
      
      {/* Overlay controls */}
      <div className="absolute top-2 right-2 flex gap-2">
        <Button
          variant="outline"
          size="sm"
          className="bg-background/80 backdrop-blur-sm"
          onClick={() => {
            // Reset camera position
            console.log('Resetting camera...');
          }}
        >
          <RotateCcw className="w-4 h-4" />
        </Button>
      </div>
    </div>
  );
};

// Specialized physics visualization types
export const QuantumVisualization: React.FC<Omit<VisualizationEngineProps, 'variant'>> = (props) => (
  <PhysicsVisualizationEngine {...props} variant="quantum" />
);

export const StatisticalVisualization: React.FC<Omit<VisualizationEngineProps, 'variant'>> = (props) => (
  <PhysicsVisualizationEngine {...props} variant="statistical" />
);

export const FieldVisualization: React.FC<Omit<VisualizationEngineProps, 'variant'>> = (props) => (
  <PhysicsVisualizationEngine {...props} variant="fields" />
);

export const CrystalVisualization: React.FC<Omit<VisualizationEngineProps, 'variant'>> = (props) => (
  <PhysicsVisualizationEngine {...props} variant="crystal" />
);

// Error boundary for 3D scenes
export class VisualizationErrorBoundary extends React.Component<
  { children: React.ReactNode; fallback?: React.ReactNode },
  { hasError: boolean }
> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Visualization error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <Alert className="m-4">
          <AlertDescription>
            Visualization failed to load. Please check your browser's WebGL support.
          </AlertDescription>
        </Alert>
      );
    }

    return this.props.children;
  }
}