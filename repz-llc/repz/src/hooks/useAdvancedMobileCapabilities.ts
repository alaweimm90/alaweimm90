import { useState, useEffect, useCallback } from 'react';
import { Capacitor } from '@capacitor/core';
import { Network } from '@capacitor/network';
import { Device } from '@capacitor/device';
import { StatusBar } from '@capacitor/status-bar';
import { Haptics, ImpactStyle } from '@capacitor/haptics';

// ============================================================================
// PHASE 3: ADVANCED MOBILE CAPABILITIES
// Native performance optimization and mobile-first features
// ============================================================================

export interface AdvancedMobileCapabilities {
  // Device info
  deviceInfo: {
    platform: string;
    model: string;
    operatingSystem: string;
    osVersion: string;
    manufacturer: string;
    isVirtual: boolean;
    webViewVersion: string;
  };
  
  // Network status
  networkStatus: {
    connected: boolean;
    connectionType: string;
    downloadSpeed?: number;
    isOnline: boolean;
    isMetered?: boolean;
  };
  
  // Performance metrics
  performance: {
    memoryUsage?: number;
    cpuUsage?: number;
    batteryLevel?: number;
    isCharging?: boolean;
    isLowPowerMode?: boolean;
  };
  
  // Screen info
  screen: {
    width: number;
    height: number;
    orientation: 'portrait' | 'landscape';
    safeAreaInsets: {
      top: number;
      bottom: number;
      left: number;
      right: number;
    };
    density: number;
  };
  
  // Capabilities
  capabilities: {
    hasCamera: boolean;
    hasGps: boolean;
    hasAccelerometer: boolean;
    hasGyroscope: boolean;
    hasBiometrics: boolean;
    hasNotifications: boolean;
    hasHaptics: boolean;
  };
}

export const useAdvancedMobileCapabilities = () => {
  const [capabilities, setCapabilities] = useState<AdvancedMobileCapabilities | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const initializeCapabilities = useCallback(async () => {
    try {
      setIsLoading(true);

      // Get device info
      const deviceInfo = await Device.getInfo();
      const deviceId = await Device.getId();
      
      // Get network status
      const networkStatus = await Network.getStatus();
      
      // Get screen info
      const width = window.innerWidth;
      const height = window.innerHeight;
      const orientation = width < height ? 'portrait' : 'landscape';
      
      // Calculate safe area insets (for notched devices)
      const safeAreaInsets = {
        top: parseInt(getComputedStyle(document.documentElement).getPropertyValue('--sat') || '0'),
        bottom: parseInt(getComputedStyle(document.documentElement).getPropertyValue('--sab') || '0'),
        left: parseInt(getComputedStyle(document.documentElement).getPropertyValue('--sal') || '0'),
        right: parseInt(getComputedStyle(document.documentElement).getPropertyValue('--sar') || '0'),
      };

      // Check capabilities
      const capabilities = {
        hasCamera: 'mediaDevices' in navigator && 'getUserMedia' in navigator.mediaDevices,
        hasGps: 'geolocation' in navigator,
        hasAccelerometer: 'DeviceMotionEvent' in window,
        hasGyroscope: 'DeviceOrientationEvent' in window,
        hasBiometrics: Capacitor.isNativePlatform(),
        hasNotifications: 'Notification' in window,
        hasHaptics: Capacitor.isNativePlatform(),
      };

      const result: AdvancedMobileCapabilities = {
        deviceInfo: {
          platform: deviceInfo.platform,
          model: deviceInfo.model,
          operatingSystem: deviceInfo.operatingSystem,
          osVersion: deviceInfo.osVersion,
          manufacturer: deviceInfo.manufacturer,
          isVirtual: deviceInfo.isVirtual,
          webViewVersion: deviceInfo.webViewVersion,
        },
        networkStatus: {
          connected: networkStatus.connected,
          connectionType: networkStatus.connectionType,
          isOnline: navigator.onLine,
        },
        performance: {
          // These will be populated by performance monitoring
        },
        screen: {
          width,
          height,
          orientation,
          safeAreaInsets,
          density: window.devicePixelRatio || 1,
        },
        capabilities,
      };

      setCapabilities(result);
    } catch (error) {
      console.error('Failed to initialize mobile capabilities:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    initializeCapabilities();

    // Listen for network changes
    Network.addListener('networkStatusChange', (status) => {
      setCapabilities(prev => prev ? {
        ...prev,
        networkStatus: {
          ...prev.networkStatus,
          connected: status.connected,
          connectionType: status.connectionType,
        }
      } : null);
    });

    // Listen for orientation changes
    const handleOrientationChange = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      const orientation = width < height ? 'portrait' : 'landscape';
      
      setCapabilities(prev => prev ? {
        ...prev,
        screen: {
          ...prev.screen,
          width,
          height,
          orientation,
        }
      } : null);
    };

    window.addEventListener('orientationchange', handleOrientationChange);
    window.addEventListener('resize', handleOrientationChange);

    return () => {
      window.removeEventListener('orientationchange', handleOrientationChange);
      window.removeEventListener('resize', handleOrientationChange);
    };
  }, [initializeCapabilities]);

  return {
    capabilities,
    isLoading,
    refresh: initializeCapabilities,
  };
};

// Native interaction helpers
export class NativeInteractions {
  static async hapticFeedback(style: ImpactStyle = ImpactStyle.Medium) {
    if (Capacitor.isNativePlatform()) {
      try {
        await Haptics.impact({ style });
      } catch (error) {
        console.warn('Haptic feedback not available:', error);
      }
    }
  }

  static async setStatusBarStyle(style: 'light' | 'dark' = 'dark') {
    if (Capacitor.isNativePlatform()) {
      try {
        await StatusBar.setStyle({ style: style === 'light' ? 'LIGHT' : 'DARK' } as { style: 'LIGHT' | 'DARK' });
      } catch (error) {
        console.warn('Status bar style not available:', error);
      }
    }
  }

  static async hideStatusBar() {
    if (Capacitor.isNativePlatform()) {
      try {
        await StatusBar.hide();
      } catch (error) {
        console.warn('Hide status bar not available:', error);
      }
    }
  }

  static async showStatusBar() {
    if (Capacitor.isNativePlatform()) {
      try {
        await StatusBar.show();
      } catch (error) {
        console.warn('Show status bar not available:', error);
      }
    }
  }
}

// Performance monitoring hook
export const usePerformanceMonitoring = () => {
  const [metrics, setMetrics] = useState({
    memoryUsage: 0,
    renderTime: 0,
    networkLatency: 0,
    frameRate: 60,
  });

  useEffect(() => {
    let frameCount = 0;
    let lastTime = performance.now();
    let animationId: number;

    const measureFrameRate = () => {
      frameCount++;
      const currentTime = performance.now();
      
      if (currentTime - lastTime >= 1000) {
        const fps = Math.round((frameCount * 1000) / (currentTime - lastTime));
        setMetrics(prev => ({ ...prev, frameRate: fps }));
        frameCount = 0;
        lastTime = currentTime;
      }
      
      animationId = requestAnimationFrame(measureFrameRate);
    };

    measureFrameRate();

    // Monitor memory usage if available
    const updateMemoryUsage = () => {
      if ('memory' in performance) {
        const memory = (performance as Performance & { memory?: MemoryInfo }).memory;
        if (memory) {
          const usedMB = memory.usedJSHeapSize / 1048576;
          setMetrics(prev => ({ ...prev, memoryUsage: usedMB }));
        }
      }
    };

    const memoryInterval = setInterval(updateMemoryUsage, 5000);

    return () => {
      cancelAnimationFrame(animationId);
      clearInterval(memoryInterval);
    };
  }, []);

  return metrics;
};

// Network quality detection
export const useNetworkQuality = () => {
  const [quality, setQuality] = useState<'poor' | 'good' | 'excellent'>('good');
  const [downloadSpeed, setDownloadSpeed] = useState(0);

  useEffect(() => {
    const measureNetworkSpeed = async () => {
      if (!navigator.onLine) {
        setQuality('poor');
        return;
      }

      try {
        const startTime = performance.now();
        const response = await fetch('/favicon.ico', { cache: 'no-cache' });
        const endTime = performance.now();
        
        if (response.ok) {
          const duration = endTime - startTime;
          const speed = duration < 100 ? 'excellent' : duration < 300 ? 'good' : 'poor';
          setQuality(speed);
          setDownloadSpeed(1000 / duration); // Requests per second
        }
      } catch (error) {
        setQuality('poor');
        setDownloadSpeed(0);
      }
    };

    measureNetworkSpeed();
    const interval = setInterval(measureNetworkSpeed, 30000); // Check every 30 seconds

    return () => clearInterval(interval);
  }, []);

  return { quality, downloadSpeed };
};