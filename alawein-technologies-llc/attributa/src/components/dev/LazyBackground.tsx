import React, { Suspense } from 'react';

// Lazy load the NeuralBackground component to improve bundle size
const NeuralBackground = React.lazy(() => import('./NeuralBackground'));
const AnimatedGrid = React.lazy(() => import('./AnimatedGrid'));

interface LazyBackgroundProps {
  showNeural?: boolean;
  showGrid?: boolean;
}

export default function LazyBackground({ 
  showNeural = true, 
  showGrid = true 
}: LazyBackgroundProps) {
  return (
    <>
      {showNeural && (
        <Suspense fallback={<div className="fixed inset-0 bg-background" />}>
          <NeuralBackground />
        </Suspense>
      )}
      {showGrid && (
        <Suspense fallback={null}>
          <AnimatedGrid />
        </Suspense>
      )}
    </>
  );
}