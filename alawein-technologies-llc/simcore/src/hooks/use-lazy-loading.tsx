/**
 * Advanced lazy loading hook with intersection observer
 */

import { useState, useEffect, useRef, useCallback } from 'react';

export interface LazyLoadingOptions {
  threshold?: number;
  rootMargin?: string;
  triggerOnce?: boolean;
  delay?: number;
}

export function useLazyLoading(options: LazyLoadingOptions = {}) {
  const {
    threshold = 0.1,
    rootMargin = '50px',
    triggerOnce = true,
    delay = 0
  } = options;

  const [isVisible, setIsVisible] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const elementRef = useRef<HTMLElement>(null);
  const observerRef = useRef<IntersectionObserver>();

  const handleIntersection = useCallback((entries: IntersectionObserverEntry[]) => {
    const [entry] = entries;
    
    if (entry.isIntersecting) {
      if (delay > 0) {
        setTimeout(() => {
          setIsVisible(true);
          if (triggerOnce && observerRef.current && elementRef.current) {
            observerRef.current.unobserve(elementRef.current);
          }
        }, delay);
      } else {
        setIsVisible(true);
        if (triggerOnce && observerRef.current && elementRef.current) {
          observerRef.current.unobserve(elementRef.current);
        }
      }
    } else if (!triggerOnce) {
      setIsVisible(false);
    }
  }, [delay, triggerOnce]);

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    observerRef.current = new IntersectionObserver(handleIntersection, {
      threshold,
      rootMargin
    });

    observerRef.current.observe(element);

    return () => {
      if (observerRef.current && element) {
        observerRef.current.unobserve(element);
      }
    };
  }, [handleIntersection, threshold, rootMargin]);

  const markAsLoaded = useCallback(() => {
    setIsLoaded(true);
  }, []);

  return {
    elementRef,
    isVisible,
    isLoaded,
    markAsLoaded
  };
}

// Component wrapper for lazy loading
export function LazyLoadWrapper({ 
  children, 
  fallback = null, 
  ...options 
}: LazyLoadingOptions & { 
  children: React.ReactNode; 
  fallback?: React.ReactNode; 
}) {
  const { elementRef, isVisible } = useLazyLoading(options);

  return (
    <div ref={elementRef as React.RefObject<HTMLDivElement>}>
      {isVisible ? children : fallback}
    </div>
  );
}

export default useLazyLoading;