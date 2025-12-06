import React, { Suspense, lazy } from 'react';
import { ComponentSkeleton } from '@/components/ComponentSkeleton';

// Granular Three.js feature loading - only load what's needed
const ThreeBlochCore = lazy(() => import('./three/BlochCore'));
const ThreeQuantumVisualization = lazy(() => import('./three/QuantumVisualization'));

interface ThreeJSOptimizedProps {
  feature: 'bloch' | 'quantum-viz' | 'circuit-3d';
  fallback?: React.ReactNode;
  className?: string;
  [key: string]: any;
}

export const ThreeJSOptimized: React.FC<ThreeJSOptimizedProps> = ({
  feature,
  fallback,
  className = '',
  ...props
}) => {
  const LoadingFallback = fallback || (
    <ComponentSkeleton 
      className={`h-96 ${className}`}
      message="Loading 3D visualization..."
      showProgressBar
    />
  );

  // Feature-specific component loading
  const getFeatureComponent = () => {
    switch (feature) {
      case 'bloch':
        return <ThreeBlochCore {...props} />;
      case 'quantum-viz':
        return <ThreeQuantumVisualization {...props} />;
      default:
        return <div className="text-center p-8">3D feature not implemented yet</div>;
    }
  };

  return (
    <div className={`three-container ${className}`}>
      <Suspense fallback={LoadingFallback}>
        {getFeatureComponent()}
      </Suspense>
    </div>
  );
};

export default ThreeJSOptimized;