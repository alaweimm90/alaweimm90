import { useState, useEffect } from 'react';

// Apple-engineered breakpoint system (non-overlapping)
const BREAKPOINTS = {
  mobile: 640,    // 0-640px
  tablet: 1024,   // 641-1024px  
  desktop: 1440   // 1025-1440px
} as const;

export type ViewportSize = 'mobile' | 'tablet' | 'desktop';

export function useResponsiveEnhanced() {
  const [viewport, setViewport] = useState<ViewportSize>('desktop');
  const [width, setWidth] = useState<number>(typeof window !== 'undefined' ? window.innerWidth : 1024);
  const [height, setHeight] = useState<number>(typeof window !== 'undefined' ? window.innerHeight : 768);

  useEffect(() => {
    const getViewportSize = (width: number): ViewportSize => {
      if (width <= BREAKPOINTS.mobile) return 'mobile';
      if (width <= BREAKPOINTS.tablet) return 'tablet';
      return 'desktop';
    };

    const handleResize = () => {
      const newWidth = window.innerWidth;
      const newHeight = window.innerHeight;
      setWidth(newWidth);
      setHeight(newHeight);
      setViewport(getViewportSize(newWidth));
    };

    // Set initial viewport
    handleResize();

    // Use passive listeners for better performance
    window.addEventListener('resize', handleResize, { passive: true });
    window.addEventListener('orientationchange', handleResize, { passive: true });
    
    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('orientationchange', handleResize);
    };
  }, []);

  // Progressive enhancement helpers
  const isTouchDevice = typeof window !== 'undefined' && 'ontouchstart' in window;
  const hasHover = typeof window !== 'undefined' && window.matchMedia('(hover: hover)').matches;
  const prefersReducedMotion = typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  return {
    viewport,
    width,
    height,
    isMobile: viewport === 'mobile',
    isTablet: viewport === 'tablet',
    isDesktop: viewport === 'desktop',
    isTouch: isTouchDevice,
    hasHover,
    prefersReducedMotion,
    breakpoints: BREAKPOINTS,
    // Responsive grid columns
    getGridColumns: (config: { mobile: number; tablet: number; desktop: number }) => config[viewport],
    // Touch-optimized spacing
    getSpacing: () => ({
      cardPadding: viewport === 'mobile' ? 'p-4' : viewport === 'tablet' ? 'p-5' : 'p-6',
      sectionPadding: viewport === 'mobile' ? 'py-8' : viewport === 'tablet' ? 'py-12' : 'py-16',
      gridGap: viewport === 'mobile' ? 'gap-4' : viewport === 'tablet' ? 'gap-5' : 'gap-6',
      touchTarget: isTouchDevice ? 'min-h-[44px] min-w-[44px]' : '',
    }),
    // Dynamic viewport units
    getViewportHeight: () => `${height}px`,
    getViewportWidth: () => `${width}px`,
  };
}

// Responsive container hook with fluid sizing
export function useResponsiveContainer() {
  const { viewport, width } = useResponsiveEnhanced();
  
  const getMaxWidth = () => {
    switch (viewport) {
      case 'mobile': return 'max-w-full';
      case 'tablet': return 'max-w-4xl';
      case 'desktop': return 'max-w-7xl';
      default: return 'max-w-7xl';
    }
  };

  const getPadding = () => {
    switch (viewport) {
      case 'mobile': return 'px-4 py-4';
      case 'tablet': return 'px-6 py-6';
      case 'desktop': return 'px-8 py-8';
      default: return 'px-8 py-8';
    }
  };

  return {
    maxWidth: getMaxWidth(),
    padding: getPadding(),
    className: `mx-auto w-full ${getMaxWidth()} ${getPadding()}`,
  };
}

// Safe area handling for iOS devices
export function useSafeAreaEnhanced() {
  const [safeArea, setSafeArea] = useState({
    top: 0,
    bottom: 0,
    left: 0,
    right: 0
  });

  useEffect(() => {
    const updateSafeArea = () => {
      if (typeof window !== 'undefined' && CSS.supports('padding: env(safe-area-inset-top)')) {
        const style = getComputedStyle(document.documentElement);
        setSafeArea({
          top: parseInt(style.getPropertyValue('env(safe-area-inset-top)') || '0'),
          bottom: parseInt(style.getPropertyValue('env(safe-area-inset-bottom)') || '0'),
          left: parseInt(style.getPropertyValue('env(safe-area-inset-left)') || '0'),
          right: parseInt(style.getPropertyValue('env(safe-area-inset-right)') || '0')
        });
      }
    };

    updateSafeArea();
    window.addEventListener('resize', updateSafeArea, { passive: true });
    window.addEventListener('orientationchange', updateSafeArea, { passive: true });
    
    return () => {
      window.removeEventListener('resize', updateSafeArea);
      window.removeEventListener('orientationchange', updateSafeArea);
    };
  }, []);

  return {
    ...safeArea,
    cssVars: {
      '--safe-area-top': `${safeArea.top}px`,
      '--safe-area-bottom': `${safeArea.bottom}px`,
      '--safe-area-left': `${safeArea.left}px`,
      '--safe-area-right': `${safeArea.right}px`,
    }
  };
}