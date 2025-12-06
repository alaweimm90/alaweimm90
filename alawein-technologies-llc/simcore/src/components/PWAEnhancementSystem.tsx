/**
 * Progressive Web App Enhancement System
 * Provides offline capabilities and app-like experience
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Download, 
  Wifi, 
  WifiOff, 
  Smartphone,
  RefreshCw,
  HardDrive,
  Globe,
  Share
} from 'lucide-react';
import { toast } from 'sonner';

interface PWAEnhancementSystemProps {
  className?: string;
}

interface InstallPromptEvent extends Event {
  prompt(): Promise<void>;
  userChoice: Promise<{outcome: 'accepted' | 'dismissed'}>;
}

export function PWAEnhancementSystem({ className }: PWAEnhancementSystemProps) {
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [installPrompt, setInstallPrompt] = useState<InstallPromptEvent | null>(null);
  const [isInstalled, setIsInstalled] = useState(false);
  const [cacheStatus, setCacheStatus] = useState<{
    size: number;
    entries: number;
    updating: boolean;
  }>({ size: 0, entries: 0, updating: false });
  const [updateAvailable, setUpdateAvailable] = useState(false);

  useEffect(() => {
    // Listen for online/offline events
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);
    
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    // Listen for app install prompt
    const handleBeforeInstallPrompt = (e: Event) => {
      e.preventDefault();
      setInstallPrompt(e as InstallPromptEvent);
    };

    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);

    // Check if app is already installed
    window.addEventListener('appinstalled', () => {
      setIsInstalled(true);
      setInstallPrompt(null);
      toast.success('App installed successfully!');
    });

    // Check for service worker updates
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.addEventListener('message', (event) => {
        if (event.data?.type === 'UPDATE_AVAILABLE') {
          setUpdateAvailable(true);
        }
      });
    }

    // Initialize cache status
    updateCacheStatus();

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
    };
  }, []);

  const updateCacheStatus = async () => {
    if ('caches' in window) {
      try {
        const cacheNames = await caches.keys();
        let totalSize = 0;
        let totalEntries = 0;

        for (const cacheName of cacheNames) {
          const cache = await caches.open(cacheName);
          const keys = await cache.keys();
          totalEntries += keys.length;

          // Estimate cache size (rough calculation)
          for (const request of keys) {
            const response = await cache.match(request);
            if (response) {
              const clone = response.clone();
              const buffer = await clone.arrayBuffer();
              totalSize += buffer.byteLength;
            }
          }
        }

        setCacheStatus({
          size: totalSize,
          entries: totalEntries,
          updating: false
        });
      } catch (error) {
        console.warn('Failed to calculate cache status:', error);
      }
    }
  };

  const handleInstallApp = async () => {
    if (!installPrompt) return;

    try {
      await installPrompt.prompt();
      const choiceResult = await installPrompt.userChoice;
      
      if (choiceResult.outcome === 'accepted') {
        toast.success('Installing app...');
      } else {
        toast.info('App installation cancelled');
      }
      
      setInstallPrompt(null);
    } catch (error) {
      toast.error('Failed to install app');
    }
  };

  const handleUpdateApp = async () => {
    if ('serviceWorker' in navigator) {
      try {
        const registration = await navigator.serviceWorker.getRegistration();
        if (registration?.waiting) {
          registration.waiting.postMessage({ type: 'SKIP_WAITING' });
          window.location.reload();
        }
      } catch (error) {
        toast.error('Failed to update app');
      }
    }
  };

  const handleClearCache = async () => {
    if ('caches' in window) {
      try {
        setCacheStatus(prev => ({ ...prev, updating: true }));
        const cacheNames = await caches.keys();
        
        await Promise.all(
          cacheNames.map(cacheName => caches.delete(cacheName))
        );
        
        toast.success('Cache cleared successfully');
        await updateCacheStatus();
      } catch (error) {
        toast.error('Failed to clear cache');
        setCacheStatus(prev => ({ ...prev, updating: false }));
      }
    }
  };

  const handleShareApp = async () => {
    if ('share' in navigator && navigator.share) {
      try {
        await navigator.share({
          title: 'Physics Simulation Platform',
          text: 'Explore interactive scientific simulations',
          url: window.location.origin
        });
      } catch (error) {
        // Fallback to clipboard
        if ('clipboard' in navigator && navigator.clipboard) {
          await navigator.clipboard.writeText(window.location.origin);
          toast.success('Link copied to clipboard');
        } else {
          toast.info('Share cancelled');
        }
      }
    } else if ('clipboard' in navigator && navigator.clipboard) {
      await navigator.clipboard.writeText(window.location.origin);
      toast.success('Link copied to clipboard');
    } else {
      toast.info('Sharing not supported');
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Status Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              {isOnline ? <Wifi className="h-4 w-4 text-green-600" /> : <WifiOff className="h-4 w-4 text-red-600" />}
              Connection
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Badge variant={isOnline ? "default" : "destructive"}>
              {isOnline ? "Online" : "Offline"}
            </Badge>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Smartphone className="h-4 w-4" />
              Installation
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Badge variant={isInstalled || !installPrompt ? "default" : "secondary"}>
              {isInstalled ? "Installed" : installPrompt ? "Available" : "Browser"}
            </Badge>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <HardDrive className="h-4 w-4" />
              Cache Size
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-sm font-medium">{formatBytes(cacheStatus.size)}</div>
            <div className="text-xs text-muted-foreground">{cacheStatus.entries} items</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <RefreshCw className="h-4 w-4" />
              Updates
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Badge variant={updateAvailable ? "destructive" : "default"}>
              {updateAvailable ? "Available" : "Current"}
            </Badge>
          </CardContent>
        </Card>
      </div>

      {/* Update Banner */}
      {updateAvailable && (
        <Card className="border-orange-200 bg-orange-50 dark:bg-orange-950">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="font-medium">App Update Available</h4>
                <p className="text-sm text-muted-foreground">A new version is ready to install</p>
              </div>
              <Button onClick={handleUpdateApp} className="flex items-center gap-2">
                <Download className="h-4 w-4" />
                Update Now
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      <Tabs defaultValue="installation" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="installation">Installation</TabsTrigger>
          <TabsTrigger value="offline">Offline</TabsTrigger>
          <TabsTrigger value="cache">Cache</TabsTrigger>
          <TabsTrigger value="features">Features</TabsTrigger>
        </TabsList>

        <TabsContent value="installation" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Smartphone className="h-5 w-5" />
                App Installation
              </CardTitle>
              <CardDescription>
                Install the app for a native-like experience
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {installPrompt && (
                <div className="p-4 border rounded-lg bg-blue-50 dark:bg-blue-950">
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="font-medium">Install Physics Simulation Platform</h4>
                      <p className="text-sm text-muted-foreground">
                        Add to your home screen for quick access and offline capabilities
                      </p>
                    </div>
                    <Button onClick={handleInstallApp} className="flex items-center gap-2">
                      <Download className="h-4 w-4" />
                      Install
                    </Button>
                  </div>
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <h4 className="font-medium">Benefits of Installing:</h4>
                  <ul className="text-sm space-y-1 text-muted-foreground">
                    <li>• Offline access to simulations</li>
                    <li>• Faster loading times</li>
                    <li>• Native app experience</li>
                    <li>• Push notifications</li>
                    <li>• Home screen shortcut</li>
                  </ul>
                </div>

                <div className="space-y-2">
                  <h4 className="font-medium">System Requirements:</h4>
                  <ul className="text-sm space-y-1 text-muted-foreground">
                    <li>• Modern web browser</li>
                    <li>• 50MB free storage</li>
                    <li>• Internet connection (initial)</li>
                    <li>• JavaScript enabled</li>
                  </ul>
                </div>
              </div>

              <div className="flex gap-2">
                <Button
                  variant="outline"
                  onClick={handleShareApp}
                  className="flex items-center gap-2"
                >
                  <Share className="h-4 w-4" />
                  Share App
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="offline" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <WifiOff className="h-5 w-5" />
                Offline Capabilities
              </CardTitle>
              <CardDescription>
                Access scientific simulations without internet connection
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="p-4 border rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">Current Status</span>
                  <Badge variant={isOnline ? "default" : "destructive"}>
                    {isOnline ? "Online" : "Offline"}
                  </Badge>
                </div>
                <p className="text-sm text-muted-foreground">
                  {isOnline 
                    ? "You're connected to the internet. The app is downloading resources for offline use."
                    : "You're offline. Using cached resources for simulation access."
                  }
                </p>
              </div>

              <div className="space-y-2">
                <h4 className="font-medium">Available Offline:</h4>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    'Graphene Band Structure',
                    'Crystal Visualizer',
                    'Quantum Tunneling',
                    'Ising Model',
                    'LLG Dynamics',
                    'Documentation'
                  ].map((feature) => (
                    <div key={feature} className="flex items-center gap-2 text-sm">
                      <div className="w-2 h-2 bg-green-500 rounded-full" />
                      {feature}
                    </div>
                  ))}
                </div>
              </div>

              <div className="p-4 border rounded-lg bg-yellow-50 dark:bg-yellow-950">
                <h4 className="font-medium text-yellow-800 dark:text-yellow-200">Note</h4>
                <p className="text-sm text-yellow-700 dark:text-yellow-300">
                  Some advanced features may require an internet connection for optimal performance.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="cache" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <HardDrive className="h-5 w-5" />
                Cache Management
              </CardTitle>
              <CardDescription>
                Manage stored data and offline resources
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium mb-2">Storage Usage</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Cache Size:</span>
                      <span className="font-medium">{formatBytes(cacheStatus.size)}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Cached Items:</span>
                      <span className="font-medium">{cacheStatus.entries}</span>
                    </div>
                    <Progress value={Math.min((cacheStatus.size / (10 * 1024 * 1024)) * 100, 100)} />
                    <p className="text-xs text-muted-foreground">
                      Estimated storage limit: 10MB
                    </p>
                  </div>
                </div>

                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium mb-2">Cache Actions</h4>
                  <div className="space-y-2">
                    <Button
                      variant="outline"
                      onClick={updateCacheStatus}
                      disabled={cacheStatus.updating}
                      className="w-full flex items-center gap-2"
                    >
                      <RefreshCw className={`h-4 w-4 ${cacheStatus.updating ? 'animate-spin' : ''}`} />
                      Refresh Status
                    </Button>
                    <Button
                      variant="outline"
                      onClick={handleClearCache}
                      disabled={cacheStatus.updating}
                      className="w-full text-red-600"
                    >
                      Clear All Cache
                    </Button>
                  </div>
                </div>
              </div>

              <div className="p-4 border rounded-lg bg-blue-50 dark:bg-blue-950">
                <h4 className="font-medium text-blue-800 dark:text-blue-200">Cache Information</h4>
                <p className="text-sm text-blue-700 dark:text-blue-300">
                  The cache stores physics simulations, assets, and data for offline access. 
                  Clearing cache will require re-downloading content when online.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="features" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Globe className="h-5 w-5" />
                PWA Features
              </CardTitle>
              <CardDescription>
                Progressive Web App capabilities and features
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {[
                  {
                    title: 'Offline Access',
                    description: 'Use simulations without internet',
                    status: 'Available',
                    icon: WifiOff
                  },
                  {
                    title: 'App Installation',
                    description: 'Install as native app',
                    status: installPrompt ? 'Available' : 'Installed',
                    icon: Smartphone
                  },
                  {
                    title: 'Background Sync',
                    description: 'Sync data when online',
                    status: 'Active',
                    icon: RefreshCw
                  },
                  {
                    title: 'Push Notifications',
                    description: 'Get simulation updates',
                    status: 'Coming Soon',
                    icon: Download
                  }
                ].map((feature) => (
                  <div key={feature.title} className="p-4 border rounded-lg">
                    <div className="flex items-start gap-3">
                      <feature.icon className="h-5 w-5 mt-1 text-blue-600" />
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <h4 className="font-medium">{feature.title}</h4>
                          <Badge variant={feature.status === 'Available' ? "default" : "secondary"}>
                            {feature.status}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">{feature.description}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="p-4 border rounded-lg bg-green-50 dark:bg-green-950">
                <h4 className="font-medium text-green-800 dark:text-green-200">PWA Ready</h4>
                <p className="text-sm text-green-700 dark:text-green-300">
                  This app meets all Progressive Web App standards for reliability, 
                  fast loading, and engaging user experience.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default PWAEnhancementSystem;