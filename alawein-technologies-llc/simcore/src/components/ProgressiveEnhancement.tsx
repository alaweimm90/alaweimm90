/**
 * Progressive Enhancement System
 * 
 * Provides intelligent feature loading based on device capabilities,
 * network conditions, and user preferences for optimal performance.
 */

import { useEffect, useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Wifi, WifiOff, Zap, Settings, Cpu, Battery } from 'lucide-react';

interface DeviceCapabilities {
  webgl: boolean;
  webgpu: boolean;
  webworkers: boolean;
  offscreenCanvas: boolean;
  highRefreshRate: boolean;
  touchSupport: boolean;
  deviceMemory: number;
  hardwareConcurrency: number;
  connectionType: string;
  effectiveType: string;
  downlink: number;
  rtt: number;
  saveData: boolean;
  reducedMotion: boolean;
  highContrast: boolean;
  batterySaving: boolean;
}

interface FeatureFlags {
  enableWebGL: boolean;
  enableWebGPU: boolean;
  enableParticleEffects: boolean;
  enable3DVisualizations: boolean;
  enableComplexAnimations: boolean;
  enableHighResolution: boolean;
  enablePrecalculation: boolean;
  enableOfflineMode: boolean;
  enableAnalytics: boolean;
  enableErrorReporting: boolean;
}

export function useProgressiveEnhancement(): {
  capabilities: DeviceCapabilities;
  features: FeatureFlags;
  performanceLevel: 'low' | 'medium' | 'high';
  isReady: boolean;
} {
  const [capabilities, setCapabilities] = useState<DeviceCapabilities>({
    webgl: false,
    webgpu: false,
    webworkers: false,
    offscreenCanvas: false,
    highRefreshRate: false,
    touchSupport: false,
    deviceMemory: 4,
    hardwareConcurrency: 2,
    connectionType: '4g',
    effectiveType: '4g',
    downlink: 10,
    rtt: 100,
    saveData: false,
    reducedMotion: false,
    highContrast: false,
    batterySaving: false
  });
  
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    const detectCapabilities = async () => {
      const caps: Partial<DeviceCapabilities> = {};

      // WebGL Detection
      try {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        caps.webgl = !!gl;
      } catch {
        caps.webgl = false;
      }

      // WebGPU Detection
      caps.webgpu = 'gpu' in navigator;

      // Worker Support
      caps.webworkers = typeof Worker !== 'undefined';
      caps.offscreenCanvas = typeof OffscreenCanvas !== 'undefined';

      // Display Capabilities
      caps.highRefreshRate = window.screen && 'refreshRate' in window.screen 
        ? (window.screen as any).refreshRate > 60 
        : false;

      // Touch Support
      caps.touchSupport = 'ontouchstart' in window || navigator.maxTouchPoints > 0;

      // Hardware Info
      caps.deviceMemory = (navigator as any).deviceMemory || 4;
      caps.hardwareConcurrency = navigator.hardwareConcurrency || 2;

      // Network Info
      const connection = (navigator as any).connection || (navigator as any).mozConnection || (navigator as any).webkitConnection;
      if (connection) {
        caps.connectionType = connection.type || '4g';
        caps.effectiveType = connection.effectiveType || '4g';
        caps.downlink = connection.downlink || 10;
        caps.rtt = connection.rtt || 100;
        caps.saveData = connection.saveData || false;
      }

      // Accessibility Preferences
      caps.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      caps.highContrast = window.matchMedia('(prefers-contrast: high)').matches;

      // Battery Status (if available)
      if ('getBattery' in navigator) {
        try {
          const battery = await (navigator as any).getBattery();
          caps.batterySaving = !battery.charging && battery.level < 0.2;
        } catch {
          caps.batterySaving = false;
        }
      }

      setCapabilities(prev => ({ ...prev, ...caps }));
      setIsReady(true);
    };

    detectCapabilities();
  }, []);

  // Calculate performance level based on capabilities
  const performanceLevel = useMemo((): 'low' | 'medium' | 'high' => {
    if (!isReady) return 'medium';

    let score = 0;

    // Graphics capabilities (40% weight)
    if (capabilities.webgpu) score += 40;
    else if (capabilities.webgl) score += 30;
    else score += 10;

    // Hardware capabilities (30% weight)
    if (capabilities.deviceMemory >= 8) score += 20;
    else if (capabilities.deviceMemory >= 4) score += 15;
    else score += 5;

    if (capabilities.hardwareConcurrency >= 8) score += 10;
    else if (capabilities.hardwareConcurrency >= 4) score += 7;
    else score += 3;

    // Network capabilities (20% weight)
    if (capabilities.effectiveType === '4g' && capabilities.downlink >= 10) score += 20;
    else if (capabilities.effectiveType === '4g') score += 15;
    else if (capabilities.effectiveType === '3g') score += 8;
    else score += 3;

    // User preferences (10% weight)
    if (capabilities.saveData) score -= 10;
    if (capabilities.batterySaving) score -= 10;
    if (capabilities.reducedMotion) score -= 5;

    if (score >= 70) return 'high';
    if (score >= 40) return 'medium';
    return 'low';
  }, [capabilities, isReady]);

  // Feature flags based on capabilities and performance level
  const features = useMemo((): FeatureFlags => {
    const baseFeatures: FeatureFlags = {
      enableWebGL: capabilities.webgl,
      enableWebGPU: capabilities.webgpu && performanceLevel === 'high',
      enableParticleEffects: performanceLevel !== 'low' && !capabilities.batterySaving,
      enable3DVisualizations: capabilities.webgl && performanceLevel !== 'low',
      enableComplexAnimations: !capabilities.reducedMotion && performanceLevel !== 'low',
      enableHighResolution: performanceLevel === 'high' && !capabilities.saveData,
      enablePrecalculation: capabilities.webworkers && performanceLevel !== 'low',
      enableOfflineMode: 'serviceWorker' in navigator,
      enableAnalytics: !capabilities.saveData,
      enableErrorReporting: true
    };

    return baseFeatures;
  }, [capabilities, performanceLevel]);

  return {
    capabilities,
    features,
    performanceLevel,
    isReady
  };
}

export function PerformanceDashboard() {
  const { capabilities, features, performanceLevel, isReady } = useProgressiveEnhancement();

  if (!isReady) {
    return (
      <Card className="bg-surface/90 backdrop-blur-sm border-muted/30">
        <CardContent className="p-6">
          <div className="flex items-center gap-3">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-interactive" />
            <span className="text-textSecondary">Detecting device capabilities...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  const performanceColors = {
    high: 'text-green-600 border-green-200 bg-green-50/50',
    medium: 'text-yellow-600 border-yellow-200 bg-yellow-50/50',
    low: 'text-red-600 border-red-200 bg-red-50/50'
  };

  const performanceIcons = {
    high: <Zap className="h-4 w-4 text-green-600" />,
    medium: <Cpu className="h-4 w-4 text-yellow-600" />,
    low: <Battery className="h-4 w-4 text-red-600" />
  };

  return (
    <div className="space-y-6">
      {/* Performance Level Indicator */}
      <Card className={`border-2 ${performanceColors[performanceLevel]}`}>
        <CardHeader className="pb-4">
          <CardTitle className="flex items-center gap-2">
            {performanceIcons[performanceLevel]}
            Performance Level: {performanceLevel.charAt(0).toUpperCase() + performanceLevel.slice(1)}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm">
            {performanceLevel === 'high' && 'All advanced features enabled. Your device can handle complex simulations with high-quality graphics.'}
            {performanceLevel === 'medium' && 'Most features enabled. Some advanced graphics features may be disabled for optimal performance.'}
            {performanceLevel === 'low' && 'Basic features enabled. Graphics and animations are simplified to ensure smooth operation.'}
          </p>
        </CardContent>
      </Card>

      {/* Device Capabilities */}
      <Card className="bg-surface/90 backdrop-blur-sm border-muted/30">
        <CardHeader>
          <CardTitle className="text-textPrimary">Device Capabilities</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-textSecondary">WebGL</span>
              <Badge variant={capabilities.webgl ? "default" : "secondary"}>
                {capabilities.webgl ? 'Supported' : 'Not Available'}
              </Badge>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-textSecondary">WebGPU</span>
              <Badge variant={capabilities.webgpu ? "default" : "secondary"}>
                {capabilities.webgpu ? 'Supported' : 'Not Available'}
              </Badge>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-textSecondary">Web Workers</span>
              <Badge variant={capabilities.webworkers ? "default" : "secondary"}>
                {capabilities.webworkers ? 'Supported' : 'Not Available'}
              </Badge>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-textSecondary">Memory</span>
              <Badge variant="outline">
                {capabilities.deviceMemory} GB
              </Badge>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-textSecondary">CPU Cores</span>
              <Badge variant="outline">
                {capabilities.hardwareConcurrency}
              </Badge>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-textSecondary">Touch</span>
              <Badge variant={capabilities.touchSupport ? "default" : "secondary"}>
                {capabilities.touchSupport ? 'Supported' : 'Not Available'}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Network Status */}
      <Card className="bg-surface/90 backdrop-blur-sm border-muted/30">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-textPrimary">
            {capabilities.connectionType === 'wifi' ? <Wifi className="h-4 w-4" /> : <WifiOff className="h-4 w-4" />}
            Network Status
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-textSecondary">Connection</span>
              <Badge variant="outline">
                {capabilities.effectiveType.toUpperCase()}
              </Badge>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-textSecondary">Speed</span>
              <Badge variant="outline">
                {capabilities.downlink} Mbps
              </Badge>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-textSecondary">Latency</span>
              <Badge variant="outline">
                {capabilities.rtt} ms
              </Badge>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-textSecondary">Data Saver</span>
              <Badge variant={capabilities.saveData ? "destructive" : "default"}>
                {capabilities.saveData ? 'Enabled' : 'Disabled'}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Feature Status */}
      <Card className="bg-surface/90 backdrop-blur-sm border-muted/30">
        <CardHeader>
          <CardTitle className="text-textPrimary">Enabled Features</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {Object.entries(features).map(([feature, enabled]) => (
              <div key={feature} className="flex items-center justify-between">
                <span className="text-sm text-textSecondary">
                  {feature.replace(/([A-Z])/g, ' $1').replace(/^enable /, '').trim()}
                </span>
                <Badge variant={enabled ? "default" : "secondary"}>
                  {enabled ? 'Enabled' : 'Disabled'}
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Performance Recommendations */}
      {performanceLevel === 'low' && (
        <Alert className="bg-yellow-50/50 border-yellow-200">
          <Settings className="h-4 w-4 text-yellow-600" />
          <AlertDescription className="text-yellow-800">
            <strong>Performance Tips:</strong> Close other tabs, enable hardware acceleration in your browser, 
            or try using a more powerful device for the best SimCore experience.
          </AlertDescription>
        </Alert>
      )}

      {capabilities.saveData && (
        <Alert className="bg-blue-50/50 border-blue-200">
          <Wifi className="h-4 w-4 text-blue-600" />
          <AlertDescription className="text-blue-800">
            <strong>Data Saver Mode:</strong> Some features are disabled to reduce data usage. 
            Disable data saver mode in your browser for the full experience.
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}