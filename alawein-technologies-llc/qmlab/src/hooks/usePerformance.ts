import { useEffect, useState, useCallback, useRef } from 'react';
import { trackQuantumEvents } from '@/lib/analytics';

// Intersection Observer Hook for Performance
export const useIntersectionObserver = () => {
  const [isIntersecting, setIsIntersecting] = useState(false);
  const [hasIntersected, setHasIntersected] = useState(false);
  const ref = useRef<HTMLElement>(null);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        setIsIntersecting(entry.isIntersecting);
        if (entry.isIntersecting && !hasIntersected) {
          setHasIntersected(true);
        }
      },
      { 
        rootMargin: '50px',
        threshold: 0.1 
      }
    );

    observer.observe(element);
    return () => observer.unobserve(element);
  }, [hasIntersected]);

  return { ref, isIntersecting, hasIntersected };
};

// Performance-aware animation hook
export const usePerformantAnimation = (
  animationName: string,
  shouldAnimate: boolean = true,
  options?: {
    respectReducedMotion?: boolean;
    pauseWhenNotVisible?: boolean;
    fpsLimit?: number;
  }
) => {
  const {
    respectReducedMotion = true,
    pauseWhenNotVisible = true,
    fpsLimit = 60
  } = options || {};

  const [isAnimating, setIsAnimating] = useState(false);
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);
  const { ref, isIntersecting } = useIntersectionObserver();
  const frameRef = useRef<number>();
  const lastFrameTimeRef = useRef(0);

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setPrefersReducedMotion(mediaQuery.matches);
    
    const handleChange = () => setPrefersReducedMotion(mediaQuery.matches);
    mediaQuery.addEventListener('change', handleChange);
    
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  const shouldRunAnimation = useCallback(() => {
    if (!shouldAnimate) return false;
    if (respectReducedMotion && prefersReducedMotion) return false;
    if (pauseWhenNotVisible && !isIntersecting) return false;
    return true;
  }, [shouldAnimate, respectReducedMotion, prefersReducedMotion, pauseWhenNotVisible, isIntersecting]);

  const startAnimation = useCallback(() => {
    if (shouldRunAnimation()) {
      setIsAnimating(true);
      trackQuantumEvents.componentLoad(animationName, performance.now());
    }
  }, [shouldRunAnimation, animationName]);

  const stopAnimation = useCallback(() => {
    setIsAnimating(false);
    if (frameRef.current) {
      cancelAnimationFrame(frameRef.current);
    }
  }, []);

  const throttledAnimationFrame = useCallback((callback: () => void) => {
    const now = performance.now();
    const elapsed = now - lastFrameTimeRef.current;
    const targetInterval = 1000 / fpsLimit;

    if (elapsed >= targetInterval) {
      callback();
      lastFrameTimeRef.current = now;
    }
    
    if (shouldRunAnimation()) {
      frameRef.current = requestAnimationFrame(() => throttledAnimationFrame(callback));
    }
  }, [shouldRunAnimation, fpsLimit]);

  useEffect(() => {
    if (shouldRunAnimation()) {
      startAnimation();
    } else {
      stopAnimation();
    }
  }, [shouldRunAnimation, startAnimation, stopAnimation]);

  return {
    ref,
    isAnimating: isAnimating && shouldRunAnimation(),
    startAnimation,
    stopAnimation,
    throttledAnimationFrame,
    prefersReducedMotion
  };
};

// Core Web Vitals tracking
export const useCoreWebVitals = () => {
  const [metrics, setMetrics] = useState<{
    lcp?: number;
    fid?: number;
    cls?: number;
    fcp?: number;
    ttfb?: number;
  }>({});

  useEffect(() => {
    // Largest Contentful Paint
    const observeLCP = () => {
      const observer = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        const lastEntry = entries[entries.length - 1] as PerformanceEventTiming;
        setMetrics(prev => ({ ...prev, lcp: lastEntry.startTime }));
      });
      observer.observe({ type: 'largest-contentful-paint', buffered: true });
    };

    // First Input Delay
    const observeFID = () => {
      const observer = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry) => {
          const fid = (entry as any).processingStart - entry.startTime;
          setMetrics(prev => ({ ...prev, fid }));
        });
      });
      observer.observe({ type: 'first-input', buffered: true });
    };

    // Cumulative Layout Shift
    const observeCLS = () => {
      let clsValue = 0;
      const observer = new PerformanceObserver((list) => {
        list.getEntries().forEach((entry) => {
          if (!(entry as any).hadRecentInput) {
            clsValue += (entry as any).value;
          }
        });
        setMetrics(prev => ({ ...prev, cls: clsValue }));
      });
      observer.observe({ type: 'layout-shift', buffered: true });
    };

    // First Contentful Paint
    const observeFCP = () => {
      const observer = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry) => {
          if (entry.name === 'first-contentful-paint') {
            setMetrics(prev => ({ ...prev, fcp: entry.startTime }));
          }
        });
      });
      observer.observe({ type: 'paint', buffered: true });
    };

    // Time to First Byte
    const observeTTFB = () => {
      const navEntry = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      if (navEntry) {
        const ttfb = navEntry.responseStart - navEntry.requestStart;
        setMetrics(prev => ({ ...prev, ttfb }));
      }
    };

    if ('PerformanceObserver' in window) {
      observeLCP();
      observeFID();
      observeCLS();
      observeFCP();
    }
    observeTTFB();

    // Track metrics after 10 seconds
    const timeout = setTimeout(() => {
      trackQuantumEvents.pageLoadTime(performance.now(), metrics);
    }, 10000);

    return () => clearTimeout(timeout);
  }, []);

  return metrics;
};

// Resource loading optimization
export const useResourceOptimization = () => {
  const [resourceLoadTimes, setResourceLoadTimes] = useState<Record<string, number>>({});

  const preloadResource = useCallback((href: string, as: string) => {
    const link = document.createElement('link');
    link.rel = 'preload';
    link.href = href;
    link.as = as;
    document.head.appendChild(link);
  }, []);

  const prefetchResource = useCallback((href: string) => {
    const link = document.createElement('link');
    link.rel = 'prefetch';
    link.href = href;
    document.head.appendChild(link);
  }, []);

  const measureResourceLoad = useCallback((resourceName: string) => {
    const startTime = performance.now();
    
    return () => {
      const endTime = performance.now();
      const loadTime = endTime - startTime;
      setResourceLoadTimes(prev => ({ ...prev, [resourceName]: loadTime }));
      trackQuantumEvents.componentLoad(resourceName, loadTime);
    };
  }, []);

  return {
    resourceLoadTimes,
    preloadResource,
    prefetchResource,
    measureResourceLoad
  };
};