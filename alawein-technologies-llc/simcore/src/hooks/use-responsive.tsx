import { useState, useEffect } from 'react';

// Apple-style responsive breakpoints
const BREAKPOINTS = {
  mobile: 640,    // ≤640px
  tablet: 1024,   // 641-1023px
  desktop: 1024   // ≥1024px
} as const;

export type ViewportSize = 'mobile' | 'tablet' | 'desktop';

export function useResponsive() {
  const [viewport, setViewport] = useState<ViewportSize>('desktop');
  const [width, setWidth] = useState<number>(typeof window !== 'undefined' ? window.innerWidth : 1024);

  useEffect(() => {
    const getViewportSize = (width: number): ViewportSize => {
      if (width <= BREAKPOINTS.mobile) return 'mobile';
      if (width <= BREAKPOINTS.tablet) return 'tablet';
      return 'desktop';
    };

    const handleResize = () => {
      const newWidth = window.innerWidth;
      setWidth(newWidth);
      setViewport(getViewportSize(newWidth));
    };

    // Set initial viewport
    handleResize();

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return {
    viewport,
    width,
    isMobile: viewport === 'mobile',
    isTablet: viewport === 'tablet',
    isDesktop: viewport === 'desktop',
    breakpoints: BREAKPOINTS
  };
}

// Responsive column calculator for grids
export function useResponsiveColumns(baseColumns: { mobile: number; tablet: number; desktop: number }) {
  const { viewport } = useResponsive();
  
  return baseColumns[viewport];
}

// Touch-optimized spacing
export function useResponsiveSpacing() {
  const { isMobile } = useResponsive();
  
  return {
    cardPadding: isMobile ? 'p-4' : 'p-6',
    sectionPadding: isMobile ? 'py-8' : 'py-16',
    gridGap: isMobile ? 'gap-4' : 'gap-6',
    marginBottom: isMobile ? 'mb-8' : 'mb-12'
  };
}

// Safe area handling for iOS
export function useSafeArea() {
  const [safeArea, setSafeArea] = useState({
    top: 0,
    bottom: 0,
    left: 0,
    right: 0
  });

  useEffect(() => {
    const updateSafeArea = () => {
      const style = getComputedStyle(document.documentElement);
      setSafeArea({
        top: parseInt(style.getPropertyValue('env(safe-area-inset-top)') || '0'),
        bottom: parseInt(style.getPropertyValue('env(safe-area-inset-bottom)') || '0'),
        left: parseInt(style.getPropertyValue('env(safe-area-inset-left)') || '0'),
        right: parseInt(style.getPropertyValue('env(safe-area-inset-right)') || '0')
      });
    };

    updateSafeArea();
    window.addEventListener('resize', updateSafeArea);
    window.addEventListener('orientationchange', updateSafeArea);
    
    return () => {
      window.removeEventListener('resize', updateSafeArea);
      window.removeEventListener('orientationchange', updateSafeArea);
    };
  }, []);

  return safeArea;
}