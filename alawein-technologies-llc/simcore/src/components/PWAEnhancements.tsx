import React, { useState, useEffect, useCallback } from 'react';
import { Download, RefreshCw, Wifi, WifiOff, Smartphone, Share } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

// PWA Installation Hook
export function usePWAInstall() {
  const [installPrompt, setInstallPrompt] = useState<any>(null);
  const [isInstalled, setIsInstalled] = useState(false);
  const [isInstallable, setIsInstallable] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    // Check if already installed
    const checkInstalled = () => {
      if (window.matchMedia && window.matchMedia('(display-mode: standalone)').matches) {
        setIsInstalled(true);
      } else if ((window.navigator as any).standalone === true) {
        setIsInstalled(true);
      }
    };

    // Listen for install prompt
    const handleBeforeInstallPrompt = (e: Event) => {
      e.preventDefault();
      setInstallPrompt(e);
      setIsInstallable(true);
    };

    // Listen for app installed
    const handleAppInstalled = () => {
      setIsInstalled(true);
      setIsInstallable(false);
      setInstallPrompt(null);
      toast({
        title: "App Installed!",
        description: "SimCore has been installed successfully.",
      });
    };

    checkInstalled();
    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
    window.addEventListener('appinstalled', handleAppInstalled);

    return () => {
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
      window.removeEventListener('appinstalled', handleAppInstalled);
    };
  }, [toast]);

  const install = useCallback(async () => {
    if (!installPrompt) return false;

    try {
      const result = await installPrompt.prompt();
      const outcome = await result.userChoice;
      
      if (outcome === 'accepted') {
        setInstallPrompt(null);
        setIsInstallable(false);
        return true;
      }
      return false;
    } catch (error) {
      console.error('Installation failed:', error);
      return false;
    }
  }, [installPrompt]);

  return { install, isInstalled, isInstallable };
}

// Network Status Hook
export function useNetworkStatus() {
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [connectionType, setConnectionType] = useState<string>('unknown');

  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    // Check connection type if available
    const connection = (navigator as any).connection || (navigator as any).mozConnection || (navigator as any).webkitConnection;
    if (connection) {
      setConnectionType(connection.effectiveType || 'unknown');
      
      const handleConnectionChange = () => {
        setConnectionType(connection.effectiveType || 'unknown');
      };
      
      connection.addEventListener('change', handleConnectionChange);
      
      return () => {
        window.removeEventListener('online', handleOnline);
        window.removeEventListener('offline', handleOffline);
        connection.removeEventListener('change', handleConnectionChange);
      };
    }

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return { isOnline, connectionType };
}

// PWA Install Banner
export function PWAInstallBanner() {
  const { install, isInstalled, isInstallable } = usePWAInstall();
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    const wasDismissed = sessionStorage.getItem('pwa-banner-dismissed');
    if (wasDismissed) setDismissed(true);
  }, []);

  const handleInstall = async () => {
    const success = await install();
    if (success) {
      setDismissed(true);
    }
  };

  const handleDismiss = () => {
    setDismissed(true);
    sessionStorage.setItem('pwa-banner-dismissed', 'true');
  };

  if (isInstalled || dismissed || !isInstallable) return null;

  return (
    <Alert className="fixed bottom-4 left-4 right-4 z-50 md:left-auto md:right-4 md:w-96">
      <Smartphone className="h-4 w-4" />
      <div className="flex-1">
        <AlertDescription className="font-medium mb-2">
          Install SimCore for the best experience
        </AlertDescription>
        <div className="flex gap-2">
          <Button onClick={handleInstall} size="sm">
            <Download className="h-4 w-4 mr-1" />
            Install
          </Button>
          <Button onClick={handleDismiss} variant="ghost" size="sm">
            Later
          </Button>
        </div>
      </div>
    </Alert>
  );
}

// Network Status Indicator
export function NetworkStatusIndicator() {
  const { isOnline, connectionType } = useNetworkStatus();

  return (
    <div className="flex items-center gap-2">
      {isOnline ? (
        <div className="flex items-center gap-1 text-green-600">
          <Wifi className="h-4 w-4" />
          <span className="text-xs">Online</span>
          {connectionType !== 'unknown' && (
            <Badge variant="secondary" className="text-xs">
              {connectionType}
            </Badge>
          )}
        </div>
      ) : (
        <div className="flex items-center gap-1 text-red-600">
          <WifiOff className="h-4 w-4" />
          <span className="text-xs">Offline</span>
        </div>
      )}
    </div>
  );
}

// App Share Component
export function AppShareButton() {
  const { toast } = useToast();

  const handleShare = async () => {
    const shareData = {
      title: 'SimCore - Physics Simulation Platform',
      text: 'Explore advanced scientific simulations with SimCore',
      url: window.location.href
    };

    try {
      if (navigator.share && navigator.canShare(shareData)) {
        await navigator.share(shareData);
      } else {
        // Fallback to clipboard
        await navigator.clipboard.writeText(shareData.url);
        toast({
          title: "Link Copied!",
          description: "Share link has been copied to clipboard.",
        });
      }
    } catch (error) {
      console.error('Share failed:', error);
    }
  };

  return (
    <Button onClick={handleShare} variant="ghost" size="sm">
      <Share className="h-4 w-4 mr-1" />
      Share
    </Button>
  );
}

// PWA Features Card
export function PWAFeaturesCard() {
  const { isInstalled } = usePWAInstall();
  const { isOnline } = useNetworkStatus();

  const features = [
    { name: 'Offline Access', available: isInstalled, icon: WifiOff },
    { name: 'Push Notifications', available: isInstalled && 'Notification' in window, icon: RefreshCw },
    { name: 'Install to Home Screen', available: isInstalled, icon: Download },
    { name: 'Background Sync', available: isInstalled && 'serviceWorker' in navigator, icon: RefreshCw }
  ];

  return (
    <Card className="w-full max-w-md">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Smartphone className="h-5 w-5" />
          App Features
        </CardTitle>
        <CardDescription>
          {isInstalled ? 'Installed as app' : 'Available features when installed'}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {features.map((feature) => (
            <div key={feature.name} className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <feature.icon className="h-4 w-4" />
                <span className="text-sm">{feature.name}</span>
              </div>
              <Badge 
                variant={feature.available ? "default" : "secondary"}
                className={cn(
                  feature.available ? "bg-accent" : "bg-muted"
                )}
              >
                {feature.available ? "Available" : "Inactive"}
              </Badge>
            </div>
          ))}
        </div>
        
        <div className="mt-4 pt-4 border-t">
          <NetworkStatusIndicator />
        </div>
      </CardContent>
    </Card>
  );
}