import React, { useState, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  Atom, 
  Zap, 
  Cpu, 
  Download, 
  CheckCircle, 
  AlertCircle,
  Loader2
} from 'lucide-react';
import { usePrefersReducedMotion } from '@/lib/accessibility-utils';

interface AnimatedLoadingProps {
  type?: 'quantum' | 'molecular' | 'computational' | 'data';
  progress?: number;
  message?: string;
  className?: string;
}

export const AnimatedLoading: React.FC<AnimatedLoadingProps> = ({
  type = 'quantum',
  progress = 0,
  message = 'Loading...',
  className = ''
}) => {
  const [animationPhase, setAnimationPhase] = useState(0);
  const prefersReducedMotion = usePrefersReducedMotion();

  useEffect(() => {
    if (prefersReducedMotion) return;

    const interval = setInterval(() => {
      setAnimationPhase(prev => (prev + 1) % 100);
    }, 50);

    return () => clearInterval(interval);
  }, [prefersReducedMotion]);

  const getLoadingIcon = () => {
    switch (type) {
      case 'quantum': return Atom;
      case 'molecular': return Zap;
      case 'computational': return Cpu;
      case 'data': return Download;
      default: return Loader2;
    }
  };

  const LoadingIcon = getLoadingIcon();

  const getAnimationClasses = () => {
    if (prefersReducedMotion) return '';
    
    switch (type) {
      case 'quantum':
        return 'animate-spin';
      case 'molecular':
        return 'animate-pulse';
      case 'computational':
        return 'animate-bounce';
      case 'data':
        return 'animate-pulse';
      default:
        return 'animate-spin';
    }
  };

  const renderQuantumOrbitals = () => {
    if (prefersReducedMotion) return null;

    return (
      <div className="absolute inset-0 flex items-center justify-center">
        {/* Electron orbitals */}
        <div className="relative w-16 h-16">
          <div 
            className="absolute inset-0 border-2 border-primary/30 rounded-full animate-spin"
            style={{ animationDuration: '3s' }}
          />
          <div 
            className="absolute inset-2 border-2 border-accent/30 rounded-full animate-spin"
            style={{ animationDuration: '2s', animationDirection: 'reverse' }}
          />
          <div 
            className="absolute inset-4 border-2 border-secondary/30 rounded-full animate-spin"
            style={{ animationDuration: '1.5s' }}
          />
          
          {/* Nucleus */}
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
            <div className="w-3 h-3 bg-primary rounded-full animate-pulse" />
          </div>
          
          {/* Electrons */}
          <div 
            className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-2 h-2 bg-accent rounded-full animate-spin"
            style={{ 
              animationDuration: '3s',
              transformOrigin: '50% 32px'
            }}
          />
          <div 
            className="absolute top-1/2 right-0 transform translate-x-1/2 -translate-y-1/2 w-2 h-2 bg-secondary rounded-full animate-spin"
            style={{ 
              animationDuration: '2s',
              animationDirection: 'reverse',
              transformOrigin: '-16px 50%'
            }}
          />
        </div>
      </div>
    );
  };

  const renderMolecularStructure = () => {
    if (prefersReducedMotion) return null;

    return (
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="relative w-20 h-12">
          {/* Molecular bonds */}
          <div className="absolute top-1/2 left-2 right-2 h-0.5 bg-primary/50 transform -translate-y-1/2" />
          <div className="absolute top-2 bottom-2 left-1/2 w-0.5 bg-primary/50 transform -translate-x-1/2 rotate-45" />
          <div className="absolute top-2 bottom-2 left-1/2 w-0.5 bg-primary/50 transform -translate-x-1/2 -rotate-45" />
          
          {/* Atoms */}
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              className={`absolute w-3 h-3 rounded-full animate-pulse ${
                i === 0 ? 'bg-red-400 top-1/2 left-0 transform -translate-y-1/2' :
                i === 1 ? 'bg-blue-400 top-1/2 right-0 transform -translate-y-1/2' :
                'bg-green-400 top-0 left-1/2 transform -translate-x-1/2'
              }`}
              style={{ animationDelay: `${i * 0.2}s` }}
            />
          ))}
        </div>
      </div>
    );
  };

  const renderComputationalGrid = () => {
    if (prefersReducedMotion) return null;

    return (
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="grid grid-cols-4 gap-1 w-16 h-16">
          {Array.from({ length: 16 }).map((_, i) => (
            <div
              key={i}
              className="w-3 h-3 bg-primary/30 rounded-sm animate-pulse"
              style={{ 
                animationDelay: `${(i % 4) * 0.1 + Math.floor(i / 4) * 0.05}s`,
                animationDuration: '1s'
              }}
            />
          ))}
        </div>
      </div>
    );
  };

  const renderDataFlow = () => {
    if (prefersReducedMotion) return null;

    return (
      <div className="absolute inset-0 flex items-center justify-center overflow-hidden">
        <div className="relative w-20 h-16">
          {/* Data streams */}
          {[0, 1, 2].map((stream) => (
            <div key={stream} className="absolute w-full h-0.5 bg-gradient-to-r from-transparent via-primary to-transparent">
              <div
                className="w-full h-full bg-gradient-to-r from-transparent via-accent to-transparent animate-pulse"
                style={{
                  top: `${stream * 20}px`,
                  animationDelay: `${stream * 0.3}s`,
                  animationDuration: '2s'
                }}
              />
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderCustomAnimation = () => {
    switch (type) {
      case 'quantum': return renderQuantumOrbitals();
      case 'molecular': return renderMolecularStructure();
      case 'computational': return renderComputationalGrid();
      case 'data': return renderDataFlow();
      default: return null;
    }
  };

  return (
    <Card className={`w-full max-w-md mx-auto ${className}`}>
      <CardContent className="p-6">
        <div className="space-y-6">
          {/* Loading Animation */}
          <div className="relative flex items-center justify-center h-24">
            {/* Background circle */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-20 h-20 border-4 border-muted rounded-full" />
            </div>
            
            {/* Progress circle */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div 
                className="w-20 h-20 border-4 border-transparent border-t-primary rounded-full transition-transform duration-300"
                style={{ 
                  transform: `rotate(${progress * 3.6}deg)`,
                  borderTopColor: progress > 0 ? 'hsl(var(--primary))' : 'transparent'
                }}
              />
            </div>
            
            {/* Custom animation overlay */}
            {renderCustomAnimation()}
            
            {/* Center icon */}
            <div className="relative z-10">
              <LoadingIcon className={`w-8 h-8 text-primary ${getAnimationClasses()}`} />
            </div>
          </div>

          {/* Progress Information */}
          <div className="space-y-3">
            <div className="flex items-center justify-between text-sm">
              <span className="font-medium">{message}</span>
              <Badge variant="outline" className="text-xs">
                {Math.round(progress)}%
              </Badge>
            </div>
            
            <Progress value={progress} className="h-2" />
            
            {/* Status indicators */}
            <div className="flex items-center justify-center gap-4 text-xs text-muted-foreground">
              <div className="flex items-center gap-1">
                {progress < 100 ? (
                  <Loader2 className="w-3 h-3 animate-spin" />
                ) : (
                  <CheckCircle className="w-3 h-3 text-green-500" />
                )}
                <span>{progress < 100 ? 'Processing' : 'Complete'}</span>
              </div>
              
              <div className="flex items-center gap-1">
                <Zap className="w-3 h-3" />
                <span>WebGPU {progress > 50 ? 'Active' : 'Initializing'}</span>
              </div>
            </div>
          </div>

          {/* Physics-themed loading states */}
          <div className="grid grid-cols-3 gap-2 text-center">
            {[
              { label: 'Quantum State', complete: progress > 25 },
              { label: 'Field Calculation', complete: progress > 50 },
              { label: 'Visualization', complete: progress > 75 }
            ].map((step, index) => (
              <div key={index} className="space-y-1">
                <div className={`w-2 h-2 rounded-full mx-auto transition-colors duration-300 ${
                  step.complete ? 'bg-green-500' : 'bg-muted'
                }`} />
                <div className="text-xs text-muted-foreground">{step.label}</div>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

// Loading skeleton components for different content types
export const ModuleLoadingSkeleton: React.FC<{ className?: string }> = ({ className = '' }) => (
  <Card className={`w-full ${className}`}>
    <CardContent className="p-6">
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-muted rounded-full animate-pulse" />
          <div className="space-y-2 flex-1">
            <div className="h-4 bg-muted rounded animate-pulse" />
            <div className="h-3 bg-muted/60 rounded w-2/3 animate-pulse" />
          </div>
        </div>
        <div className="h-32 bg-muted/30 rounded-lg animate-pulse" />
        <div className="flex gap-2">
          <div className="h-8 bg-muted rounded w-20 animate-pulse" />
          <div className="h-8 bg-muted rounded w-16 animate-pulse" />
        </div>
      </div>
    </CardContent>
  </Card>
);

export const PlotLoadingSkeleton: React.FC<{ className?: string }> = ({ className = '' }) => (
  <Card className={`w-full ${className}`}>
    <CardContent className="p-6">
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="h-6 bg-muted rounded w-32 animate-pulse" />
          <div className="h-4 bg-muted rounded w-16 animate-pulse" />
        </div>
        <div className="h-64 bg-muted/30 rounded-lg animate-pulse relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent animate-shimmer" />
        </div>
        <div className="flex justify-between">
          <div className="h-3 bg-muted rounded w-24 animate-pulse" />
          <div className="h-3 bg-muted rounded w-20 animate-pulse" />
        </div>
      </div>
    </CardContent>
  </Card>
);