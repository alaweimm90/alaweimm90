import React, { useState, useEffect, useCallback } from 'react';
import { useResponsive } from '@/hooks/use-responsive';
import { usePrefersReducedMotion } from '@/lib/accessibility-utils';

interface PerformanceMonitorProps {
  children: React.ReactNode;
  budgets?: {
    mobile: { loadTime: number; renderTime: number };
    tablet: { loadTime: number; renderTime: number };
    desktop: { loadTime: number; renderTime: number };
  };
}

const DEFAULT_BUDGETS = {
  mobile: { loadTime: 3000, renderTime: 1000 },
  tablet: { loadTime: 2500, renderTime: 800 },
  desktop: { loadTime: 2000, renderTime: 600 }
};

export function PerformanceMonitor({ 
  children, 
  budgets = DEFAULT_BUDGETS 
}: PerformanceMonitorProps) {
  const { viewport } = useResponsive();
  const prefersReducedMotion = usePrefersReducedMotion();
  const [metrics, setMetrics] = useState({
    loadTime: 0,
    renderTime: 0,
    fps: 0
  });

  useEffect(() => {
    const startTime = performance.now();
    let frameCount = 0;
    let lastFrameTime = startTime;
    let animationFrame: number;

    const measurePerformance = () => {
      const currentTime = performance.now();
      frameCount++;
      
      if (currentTime - lastFrameTime >= 1000) {
        const fps = Math.round((frameCount * 1000) / (currentTime - lastFrameTime));
        setMetrics(prev => ({ ...prev, fps }));
        frameCount = 0;
        lastFrameTime = currentTime;
      }
      
      if (!prefersReducedMotion) {
        animationFrame = requestAnimationFrame(measurePerformance);
      }
    };

    // Measure initial load time
    const loadTime = performance.now() - (performance.timing?.navigationStart || startTime);
    setMetrics(prev => ({ ...prev, loadTime }));

    // Start FPS monitoring
    if (!prefersReducedMotion) {
      animationFrame = requestAnimationFrame(measurePerformance);
    }

    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
    };
  }, [prefersReducedMotion]);

  const currentBudget = budgets[viewport];
  const isPerformanceGood = 
    metrics.loadTime <= currentBudget.loadTime &&
    (prefersReducedMotion || metrics.fps >= 30);

  // Log performance warnings in development
  useEffect(() => {
    if (process.env.NODE_ENV === 'development') {
      if (metrics.loadTime > currentBudget.loadTime) {
        console.warn(`Performance budget exceeded: Load time ${metrics.loadTime}ms > ${currentBudget.loadTime}ms on ${viewport}`);
      }
      if (!prefersReducedMotion && metrics.fps < 30) {
        console.warn(`Performance warning: FPS ${metrics.fps} < 30 on ${viewport}`);
      }
    }
  }, [metrics, currentBudget, viewport, prefersReducedMotion]);

  return <>{children}</>;
}

interface LazyComponentProps {
  children: React.ReactNode;
  threshold?: number;
  rootMargin?: string;
  fallback?: React.ReactNode;
}

export function LazyComponent({
  children,
  threshold = 0.1,
  rootMargin = '50px',
  fallback = <div className="animate-pulse bg-muted rounded h-32" />
}: LazyComponentProps) {
  const [isVisible, setIsVisible] = useState(false);
  const [hasLoaded, setHasLoaded] = useState(false);
  const ref = React.useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !hasLoaded) {
          setIsVisible(true);
          setHasLoaded(true);
          observer.disconnect();
        }
      },
      { threshold, rootMargin }
    );

    if (ref.current) {
      observer.observe(ref.current);
    }

    return () => observer.disconnect();
  }, [threshold, rootMargin, hasLoaded]);

  return (
    <div ref={ref}>
      {isVisible ? children : fallback}
    </div>
  );
}

interface AdaptiveImageProps extends React.ImgHTMLAttributes<HTMLImageElement> {
  src: string;
  alt: string;
  sizes?: string;
  loading?: 'lazy' | 'eager';
  priority?: boolean;
}

export function AdaptiveImage({
  src,
  alt,
  sizes = '(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw',
  loading = 'lazy',
  priority = false,
  className = '',
  ...props
}: AdaptiveImageProps) {
  const { isMobile } = useResponsive();
  const [isLoaded, setIsLoaded] = useState(false);
  const [hasError, setHasError] = useState(false);

  const handleLoad = useCallback(() => {
    setIsLoaded(true);
  }, []);

  const handleError = useCallback(() => {
    setHasError(true);
  }, []);

  if (hasError) {
    return (
      <div className={`bg-muted rounded flex items-center justify-center text-muted-foreground ${className}`}>
        <span className="text-sm">Image failed to load</span>
      </div>
    );
  }

  return (
    <div className={`relative ${className}`}>
      {!isLoaded && (
        <div className="absolute inset-0 bg-muted animate-pulse rounded" />
      )}
      <img
        src={src}
        alt={alt}
        sizes={sizes}
        loading={priority ? 'eager' : loading}
        onLoad={handleLoad}
        onError={handleError}
        className={`transition-opacity duration-300 ${isLoaded ? 'opacity-100' : 'opacity-0'} ${className}`}
        {...props}
      />
    </div>
  );
}

export function useNetworkAwareLoading() {
  const [connectionType, setConnectionType] = useState<'slow' | 'fast'>('fast');

  useEffect(() => {
    // Check connection type
    const connection = (navigator as any)?.connection;
    if (connection) {
      const updateConnection = () => {
        const effectiveType = connection.effectiveType;
        setConnectionType(
          effectiveType === 'slow-2g' || effectiveType === '2g' ? 'slow' : 'fast'
        );
      };
      
      updateConnection();
      connection.addEventListener('change', updateConnection);
      
      return () => {
        connection.removeEventListener('change', updateConnection);
      };
    }
  }, []);

  return { connectionType, isSlowConnection: connectionType === 'slow' };
}