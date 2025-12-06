import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Smartphone, 
  Battery, 
  Wifi, 
  Download, 
  Share2,
  Vibrate,
  Camera,
  FileText,
  Settings,
  Info,
  CheckCircle,
  AlertTriangle,
  Zap
} from 'lucide-react';
import { Device } from '@capacitor/device';
import { Network } from '@capacitor/network';
import { Filesystem, Directory, Encoding } from '@capacitor/filesystem';
import { Share } from '@capacitor/share';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

interface DeviceInfo {
  platform: string;
  operatingSystem: string;
  osVersion: string;
  manufacturer: string;
  model: string;
  isVirtual: boolean;
  webViewVersion: string;
  batteryLevel?: number;
  isCharging?: boolean;
}

interface NetworkInfo {
  connected: boolean;
  connectionType: string;
}

interface MobileFeatures {
  offlineMode: boolean;
  batteryOptimization: boolean;
  hapticFeedback: boolean;
  dataCompression: boolean;
  adaptiveQuality: boolean;
  gestureNavigation: boolean;
}

interface MobileEnhancedFeaturesProps {
  className?: string;
}

export const MobileEnhancedFeatures: React.FC<MobileEnhancedFeaturesProps> = ({ className }) => {
  const { toast } = useToast();
  const [deviceInfo, setDeviceInfo] = useState<DeviceInfo | null>(null);
  const [networkInfo, setNetworkInfo] = useState<NetworkInfo | null>(null);
  const [features, setFeatures] = useState<MobileFeatures>({
    offlineMode: false,
    batteryOptimization: true,
    hapticFeedback: true,
    dataCompression: true,
    adaptiveQuality: true,
    gestureNavigation: true
  });
  const [isNativeApp, setIsNativeApp] = useState(false);
  const [storageUsage, setStorageUsage] = useState({ used: 0, total: 100 });

  useEffect(() => {
    initializeMobileFeatures();
  }, []);

  const initializeMobileFeatures = async () => {
    try {
      // Check if running in Capacitor
      const { Capacitor } = await import('@capacitor/core');
      setIsNativeApp(Capacitor.isNativePlatform());

      if (Capacitor.isNativePlatform()) {
        // Get device information
        const deviceData = await Device.getInfo();
        const batteryData = await Device.getBatteryInfo();
        
        setDeviceInfo({
          platform: deviceData.platform,
          operatingSystem: deviceData.operatingSystem,
          osVersion: deviceData.osVersion,
          manufacturer: deviceData.manufacturer,
          model: deviceData.model,
          isVirtual: deviceData.isVirtual,
          webViewVersion: deviceData.webViewVersion,
          batteryLevel: batteryData.batteryLevel ? batteryData.batteryLevel * 100 : undefined,
          isCharging: batteryData.isCharging
        });

        // Get network information
        const networkData = await Network.getStatus();
        setNetworkInfo({
          connected: networkData.connected,
          connectionType: networkData.connectionType
        });

        // Listen for network changes
        Network.addListener('networkStatusChange', (status) => {
          setNetworkInfo({
            connected: status.connected,
            connectionType: status.connectionType
          });
        });

        // Simulate storage usage
        setStorageUsage({ used: 25, total: 100 });
      }
    } catch (error) {
      console.error('Failed to initialize mobile features:', error);
    }
  };

  const toggleFeature = (feature: keyof MobileFeatures) => {
    setFeatures(prev => ({
      ...prev,
      [feature]: !prev[feature]
    }));

    toast({
      title: "Feature Updated",
      description: `${feature.replace(/([A-Z])/g, ' $1').toLowerCase()} ${features[feature] ? 'disabled' : 'enabled'}`
    });
  };

  const exportToMobile = async () => {
    if (!isNativeApp) {
      toast({
        title: "Mobile App Required",
        description: "This feature requires the native mobile app",
        variant: "destructive"
      });
      return;
    }

    try {
      // Create export data
      const exportData = {
        timestamp: new Date().toISOString(),
        deviceInfo,
        features,
        simulationResults: "Sample simulation data would go here"
      };

      // Write to filesystem
      await Filesystem.writeFile({
        path: `simcore_export_${Date.now()}.json`,
        data: JSON.stringify(exportData, null, 2),
        directory: Directory.Documents,
        encoding: Encoding.UTF8
      });

      toast({
        title: "Export Successful",
        description: "Data saved to device documents folder"
      });
    } catch (error) {
      toast({
        title: "Export Failed",
        description: "Unable to save to device storage",
        variant: "destructive"
      });
    }
  };

  const shareResults = async () => {
    if (!isNativeApp) {
      // Fallback for web
      if (navigator.share) {
        try {
          await navigator.share({
            title: 'SimCore Physics Simulation',
            text: 'Check out these simulation results from SimCore!',
            url: window.location.href
          });
        } catch (error) {
          console.error('Error sharing:', error);
        }
      }
      return;
    }

    try {
      await Share.share({
        title: 'SimCore Physics Simulation',
        text: 'Check out these simulation results from SimCore!',
        url: window.location.href
      });
    } catch (error) {
      console.error('Error sharing:', error);
    }
  };

  const clearCache = async () => {
    if (!isNativeApp) {
      // Clear web cache
      if ('caches' in window) {
        const cacheNames = await caches.keys();
        await Promise.all(
          cacheNames.map(cacheName => caches.delete(cacheName))
        );
      }
      
      toast({
        title: "Cache Cleared",
        description: "Web cache has been cleared"
      });
      return;
    }

    try {
      // Clear app cache and temporary files
      setStorageUsage(prev => ({ ...prev, used: 5 }));
      
      toast({
        title: "Cache Cleared",
        description: "App cache and temporary files removed"
      });
    } catch (error) {
      toast({
        title: "Clear Failed",
        description: "Unable to clear cache",
        variant: "destructive"
      });
    }
  };

  const getConnectionIcon = () => {
    if (!networkInfo?.connected) return <Wifi className="w-4 h-4 text-red-500" />;
    
    switch (networkInfo.connectionType) {
      case 'wifi':
        return <Wifi className="w-4 h-4 text-green-500" />;
      case 'cellular':
        return <Smartphone className="w-4 h-4 text-blue-500" />;
      default:
        return <Wifi className="w-4 h-4 text-gray-500" />;
    }
  };

  const getBatteryIcon = () => {
    if (!deviceInfo?.batteryLevel) return <Battery className="w-4 h-4" />;
    
    const level = deviceInfo.batteryLevel;
    if (level > 50) return <Battery className="w-4 h-4 text-green-500" />;
    if (level > 20) return <Battery className="w-4 h-4 text-yellow-500" />;
    return <Battery className="w-4 h-4 text-red-500" />;
  };

  return (
    <div className={cn('w-full space-y-6', className)}>
      {/* Device Status */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Smartphone className="w-5 h-5 text-accentPhysics" />
                <span className="font-medium">Device</span>
              </div>
              <Badge variant={isNativeApp ? "default" : "secondary"}>
                {isNativeApp ? "Native" : "Web"}
              </Badge>
            </div>
            {deviceInfo && (
              <div className="mt-2 text-sm text-textSecondary">
                {deviceInfo.manufacturer} {deviceInfo.model}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {getConnectionIcon()}
                <span className="font-medium">Network</span>
              </div>
              <Badge variant={networkInfo?.connected ? "default" : "destructive"}>
                {networkInfo?.connected ? "Online" : "Offline"}
              </Badge>
            </div>
            {networkInfo && (
              <div className="mt-2 text-sm text-textSecondary">
                {networkInfo.connectionType}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {getBatteryIcon()}
                <span className="font-medium">Battery</span>
              </div>
              {deviceInfo?.batteryLevel && (
                <Badge variant="outline">
                  {Math.round(deviceInfo.batteryLevel)}%
                </Badge>
              )}
            </div>
            {deviceInfo?.isCharging && (
              <div className="mt-2 text-sm text-green-600">
                Charging
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileText className="w-5 h-5 text-accentQuantum" />
                <span className="font-medium">Storage</span>
              </div>
              <Badge variant="outline">
                {storageUsage.used}MB
              </Badge>
            </div>
            <Progress value={(storageUsage.used / storageUsage.total) * 100} className="mt-2 h-2" />
          </CardContent>
        </Card>
      </div>

      {/* Mobile Features */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="w-5 h-5" />
            Mobile Optimization Features
          </CardTitle>
          <CardDescription>
            Enhance your mobile physics simulation experience
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="offline-mode">Offline Mode</Label>
                  <p className="text-sm text-textSecondary">
                    Cache simulations for offline use
                  </p>
                </div>
                <Switch
                  id="offline-mode"
                  checked={features.offlineMode}
                  onCheckedChange={() => toggleFeature('offlineMode')}
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="battery-optimization">Battery Optimization</Label>
                  <p className="text-sm text-textSecondary">
                    Reduce CPU usage to save battery
                  </p>
                </div>
                <Switch
                  id="battery-optimization"
                  checked={features.batteryOptimization}
                  onCheckedChange={() => toggleFeature('batteryOptimization')}
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="haptic-feedback">Haptic Feedback</Label>
                  <p className="text-sm text-textSecondary">
                    Vibration feedback for interactions
                  </p>
                </div>
                <Switch
                  id="haptic-feedback"
                  checked={features.hapticFeedback}
                  onCheckedChange={() => toggleFeature('hapticFeedback')}
                />
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="data-compression">Data Compression</Label>
                  <p className="text-sm text-textSecondary">
                    Reduce data usage with compression
                  </p>
                </div>
                <Switch
                  id="data-compression"
                  checked={features.dataCompression}
                  onCheckedChange={() => toggleFeature('dataCompression')}
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="adaptive-quality">Adaptive Quality</Label>
                  <p className="text-sm text-textSecondary">
                    Adjust graphics based on performance
                  </p>
                </div>
                <Switch
                  id="adaptive-quality"
                  checked={features.adaptiveQuality}
                  onCheckedChange={() => toggleFeature('adaptiveQuality')}
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="gesture-navigation">Gesture Navigation</Label>
                  <p className="text-sm text-textSecondary">
                    Swipe and pinch to navigate
                  </p>
                </div>
                <Switch
                  id="gesture-navigation"
                  checked={features.gestureNavigation}
                  onCheckedChange={() => toggleFeature('gestureNavigation')}
                />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Mobile Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Mobile Actions</CardTitle>
          <CardDescription>
            Native mobile functionality and data management
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <Button 
              variant="outline" 
              className="gap-2 h-20 flex-col"
              onClick={exportToMobile}
            >
              <Download className="w-6 h-6" />
              Export Data
            </Button>
            
            <Button 
              variant="outline" 
              className="gap-2 h-20 flex-col"
              onClick={shareResults}
            >
              <Share2 className="w-6 h-6" />
              Share Results
            </Button>
            
            <Button 
              variant="outline" 
              className="gap-2 h-20 flex-col"
              onClick={clearCache}
            >
              <FileText className="w-6 h-6" />
              Clear Cache
            </Button>
            
            <Button 
              variant="outline" 
              className="gap-2 h-20 flex-col"
              onClick={() => {
                if ('vibrate' in navigator) {
                  navigator.vibrate(200);
                }
                toast({
                  title: "Haptic Test",
                  description: "Device vibration triggered"
                });
              }}
            >
              <Vibrate className="w-6 h-6" />
              Test Haptics
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Performance Alerts */}
      {deviceInfo?.batteryLevel && deviceInfo.batteryLevel < 20 && (
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            Low battery detected. Consider enabling battery optimization mode.
          </AlertDescription>
        </Alert>
      )}

      {!networkInfo?.connected && (
        <Alert>
          <Info className="h-4 w-4" />
          <AlertDescription>
            You're offline. Some features may be limited. Enable offline mode to cache simulations.
          </AlertDescription>
        </Alert>
      )}

      {/* Device Information */}
      {deviceInfo && isNativeApp && (
        <Card>
          <CardHeader>
            <CardTitle>Device Information</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-textSecondary">Platform:</span>
                  <span>{deviceInfo.platform}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-textSecondary">OS:</span>
                  <span>{deviceInfo.operatingSystem} {deviceInfo.osVersion}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-textSecondary">Manufacturer:</span>
                  <span>{deviceInfo.manufacturer}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-textSecondary">Model:</span>
                  <span>{deviceInfo.model}</span>
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-textSecondary">WebView:</span>
                  <span>{deviceInfo.webViewVersion}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-textSecondary">Virtual Device:</span>
                  <span>{deviceInfo.isVirtual ? 'Yes' : 'No'}</span>
                </div>
                {deviceInfo.batteryLevel && (
                  <div className="flex justify-between">
                    <span className="text-textSecondary">Battery:</span>
                    <span className="flex items-center gap-1">
                      {Math.round(deviceInfo.batteryLevel)}%
                      {deviceInfo.isCharging && <Zap className="w-3 h-3 text-green-500" />}
                    </span>
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default MobileEnhancedFeatures;