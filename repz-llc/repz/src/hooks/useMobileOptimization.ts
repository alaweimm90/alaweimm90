import { useState, useEffect, useCallback } from 'react';

interface TouchGesture {
  type: 'tap' | 'swipe' | 'pinch' | 'long-press';
  startX: number;
  startY: number;
  endX?: number;
  endY?: number;
  duration: number;
  pressure?: number;
}

interface MobileCapabilities {
  isTouch: boolean;
  hasOrientationSupport: boolean;
  hasVibration: boolean;
  hasGeolocation: boolean;
  hasCamera: boolean;
  hasMotionSensors: boolean;
  hasNotificationSupport: boolean;
}

interface ViewportDimensions {
  width: number;
  height: number;
  orientation: 'portrait' | 'landscape';
  safeAreaInsets: {
    top: number;
    right: number;
    bottom: number;
    left: number;
  };
}

export function useMobileOptimization() {
  const [capabilities, setCapabilities] = useState<MobileCapabilities>({
    isTouch: false,
    hasOrientationSupport: false,
    hasVibration: false,
    hasGeolocation: false,
    hasCamera: false,
    hasMotionSensors: false,
    hasNotificationSupport: false,
  });

  const [viewport, setViewport] = useState<ViewportDimensions>({
    width: typeof window !== 'undefined' ? window.innerWidth : 0,
    height: typeof window !== 'undefined' ? window.innerHeight : 0,
    orientation: 'portrait',
    safeAreaInsets: { top: 0, right: 0, bottom: 0, left: 0 },
  });

  const [currentGesture, setCurrentGesture] = useState<TouchGesture | null>(null);

  // Detect mobile capabilities
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const detectCapabilities = () => {
      setCapabilities({
        isTouch: 'ontouchstart' in window || navigator.maxTouchPoints > 0,
        hasOrientationSupport: 'orientation' in window,
        hasVibration: 'vibrate' in navigator,
        hasGeolocation: 'geolocation' in navigator,
        hasCamera: 'mediaDevices' in navigator && 'getUserMedia' in navigator.mediaDevices,
        hasMotionSensors: 'DeviceMotionEvent' in window,
        hasNotificationSupport: 'Notification' in window,
      });
    };

    detectCapabilities();
  }, []);

  // Monitor viewport changes
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const updateViewport = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      const orientation = width > height ? 'landscape' : 'portrait';

      // Calculate safe area insets (for iOS notch/home indicator)
      const safeAreaInsets = {
        top: parseInt(getComputedStyle(document.documentElement).getPropertyValue('--sat') || '0'),
        right: parseInt(getComputedStyle(document.documentElement).getPropertyValue('--sar') || '0'),
        bottom: parseInt(getComputedStyle(document.documentElement).getPropertyValue('--sab') || '0'),
        left: parseInt(getComputedStyle(document.documentElement).getPropertyValue('--sal') || '0'),
      };

      setViewport({ width, height, orientation, safeAreaInsets });
    };

    updateViewport();
    window.addEventListener('resize', updateViewport);
    window.addEventListener('orientationchange', updateViewport);

    return () => {
      window.removeEventListener('resize', updateViewport);
      window.removeEventListener('orientationchange', updateViewport);
    };
  }, []);

  // Touch gesture handling
  const handleTouchStart = useCallback((event: TouchEvent) => {
    const touch = event.touches[0];
    setCurrentGesture({
      type: 'tap',
      startX: touch.clientX,
      startY: touch.clientY,
      duration: Date.now(),
      pressure: touch.force,
    });
  }, []);

  const handleTouchEnd = useCallback((event: TouchEvent) => {
    if (!currentGesture) return;

    const touch = event.changedTouches[0];
    const duration = Date.now() - currentGesture.duration;
    const deltaX = Math.abs(touch.clientX - currentGesture.startX);
    const deltaY = Math.abs(touch.clientY - currentGesture.startY);

    let gestureType: TouchGesture['type'] = 'tap';

    if (duration > 500) {
      gestureType = 'long-press';
    } else if (deltaX > 50 || deltaY > 50) {
      gestureType = 'swipe';
    }

    setCurrentGesture({
      ...currentGesture,
      type: gestureType,
      endX: touch.clientX,
      endY: touch.clientY,
      duration,
    });

    // Clear gesture after processing
    setTimeout(() => setCurrentGesture(null), 100);
  }, [currentGesture]);

  // Haptic feedback
  const triggerHapticFeedback = useCallback((
    type: 'selection' | 'impact' | 'notification' = 'selection',
    intensity: 'light' | 'medium' | 'heavy' = 'medium'
  ) => {
    if (!capabilities.hasVibration) return;

    const patterns = {
      selection: [10],
      impact: { light: [20], medium: [40], heavy: [80] },
      notification: [50, 50, 50],
    };

    let pattern = patterns[type];
    if (type === 'impact') {
      pattern = pattern[intensity];
    }

    navigator.vibrate(pattern as number[]);
  }, [capabilities.hasVibration]);

  // Optimize images for mobile
  const optimizeImageForMobile = useCallback((
    src: string,
    options: {
      quality?: number;
      format?: 'webp' | 'avif' | 'jpeg';
      maxWidth?: number;
      maxHeight?: number;
    } = {}
  ) => {
    const {
      quality = 0.8,
      format = 'webp',
      maxWidth = viewport.width * 2, // 2x for retina
      maxHeight = viewport.height * 2,
    } = options;

    // For production, this would integrate with an image optimization service
    // For now, return optimized parameters
    return {
      src,
      srcSet: `${src}?w=${maxWidth}&q=${quality}&f=${format} 2x`,
      loading: 'lazy' as const,
      decoding: 'async' as const,
    };
  }, [viewport]);

  // Check if current viewport meets touch target requirements
  const validateTouchTarget = useCallback((element: HTMLElement) => {
    const rect = element.getBoundingClientRect();
    const minSize = 44; // 44px minimum per Apple HIG

    return {
      width: rect.width,
      height: rect.height,
      meetsAppleHIG: rect.width >= minSize && rect.height >= minSize,
      meetsMaterialDesign: rect.width >= 48 && rect.height >= 48,
      hasAdequateSpacing: true, // Would need to check neighboring elements
    };
  }, []);

  // Performance monitoring for mobile
  const measurePerformance = useCallback(() => {
    if (typeof window === 'undefined' || !('performance' in window)) {
      return null;
    }

    const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
    const paint = performance.getEntriesByType('paint');

    return {
      loadTime: navigation.loadEventEnd - navigation.loadEventStart,
      domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
      firstContentfulPaint: paint.find(p => p.name === 'first-contentful-paint')?.startTime || 0,
      largestContentfulPaint: paint.find(p => p.name === 'largest-contentful-paint')?.startTime || 0,
      memoryUsage: (performance as Performance & { memory?: MemoryInfo }).memory ? {
        used: (performance as Performance & { memory?: MemoryInfo }).memory!.usedJSHeapSize,
        total: (performance as Performance & { memory?: MemoryInfo }).memory!.totalJSHeapSize,
        limit: (performance as Performance & { memory?: MemoryInfo }).memory!.jsHeapSizeLimit,
      } : null,
    };
  }, []);

  // Detect network conditions
  const [networkInfo, setNetworkInfo] = useState({
    effectiveType: '4g',
    downlink: 10,
    rtt: 100,
    saveData: false,
  });

  useEffect(() => {
    if (typeof window === 'undefined' || !('navigator' in window)) return;

    const connection = (navigator as Navigator & { connection?: NetworkInformation; mozConnection?: NetworkInformation; webkitConnection?: NetworkInformation }).connection ||
                     (navigator as Navigator & { connection?: NetworkInformation; mozConnection?: NetworkInformation; webkitConnection?: NetworkInformation }).mozConnection ||
                     (navigator as Navigator & { connection?: NetworkInformation; mozConnection?: NetworkInformation; webkitConnection?: NetworkInformation }).webkitConnection;

    if (connection) {
      const updateNetworkInfo = () => {
        setNetworkInfo({
          effectiveType: connection.effectiveType || '4g',
          downlink: connection.downlink || 10,
          rtt: connection.rtt || 100,
          saveData: connection.saveData || false,
        });
      };

      updateNetworkInfo();
      connection.addEventListener('change', updateNetworkInfo);

      return () => {
        connection.removeEventListener('change', updateNetworkInfo);
      };
    }
  }, []);

  return {
    capabilities,
    viewport,
    currentGesture,
    networkInfo,
    triggerHapticFeedback,
    optimizeImageForMobile,
    validateTouchTarget,
    measurePerformance,
    handleTouchStart,
    handleTouchEnd,
    
    // Utility functions
    isMobile: viewport.width < 768,
    isTablet: viewport.width >= 768 && viewport.width < 1024,
    isPortrait: viewport.orientation === 'portrait',
    isLandscape: viewport.orientation === 'landscape',
    hasNotch: viewport.safeAreaInsets.top > 0,
    isSlowNetwork: networkInfo.effectiveType === 'slow-2g' || networkInfo.effectiveType === '2g',
    shouldReduceMotion: capabilities.hasMotionSensors && window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  };
}