import { useState, useEffect } from 'react';
import { Capacitor } from '@capacitor/core';

export interface MobileDetection {
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  isTouchDevice: boolean;
  screenWidth: number;
  screenHeight: number;
  orientation: 'portrait' | 'landscape';
  isNative: boolean;
  platform: string;
}

export const useMobileDetection = (): MobileDetection => {
  const [detection, setDetection] = useState<MobileDetection>(() => {
    const width = window.innerWidth;
    const height = window.innerHeight;
    
    return {
      isMobile: width < 768,
      isTablet: width >= 768 && width < 1024,
      isDesktop: width >= 1024,
      isTouchDevice: 'ontouchstart' in window || navigator.maxTouchPoints > 0,
      screenWidth: width,
      screenHeight: height,
      orientation: width < height ? 'portrait' : 'landscape',
      isNative: Capacitor.isNativePlatform(),
      platform: Capacitor.getPlatform()
    };
  });

  useEffect(() => {
    const updateDetection = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      
      setDetection({
        isMobile: width < 768,
        isTablet: width >= 768 && width < 1024,
        isDesktop: width >= 1024,
        isTouchDevice: 'ontouchstart' in window || navigator.maxTouchPoints > 0,
        screenWidth: width,
        screenHeight: height,
        orientation: width < height ? 'portrait' : 'landscape',
        isNative: Capacitor.isNativePlatform(),
        platform: Capacitor.getPlatform()
      });
    };

    // Listen for resize and orientation changes
    window.addEventListener('resize', updateDetection);
    window.addEventListener('orientationchange', updateDetection);

    return () => {
      window.removeEventListener('resize', updateDetection);
      window.removeEventListener('orientationchange', updateDetection);
    };
  }, []);

  return detection;
};