/**
 * PWA Enhanced Features
 * 
 * Progressive Web App enhancements for SimCore including offline support,
 * install prompts, background sync, and performance optimizations.
 */

import { useEffect, useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Download, Wifi, WifiOff, RefreshCw, Info } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface BeforeInstallPromptEvent extends Event {
  prompt(): Promise<void>;
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>;
}

export function PWAInstallPrompt() {
  const [deferredPrompt, setDeferredPrompt] = useState<BeforeInstallPromptEvent | null>(null);
  const [isInstallable, setIsInstallable] = useState(false);
  const [isInstalled, setIsInstalled] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    const handleBeforeInstallPrompt = (e: Event) => {
      e.preventDefault();
      setDeferredPrompt(e as BeforeInstallPromptEvent);
      setIsInstallable(true);
    };

    const handleAppInstalled = () => {
      setIsInstalled(true);
      setIsInstallable(false);
      toast({
        title: "SimCore Installed!",
        description: "You can now access SimCore from your home screen."
      });
    };

    // Check if already installed
    const isStandalone = window.matchMedia('(display-mode: standalone)').matches;
    if (isStandalone || (window.navigator as any).standalone) {
      setIsInstalled(true);
    }

    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
    window.addEventListener('appinstalled', handleAppInstalled);

    return () => {
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
      window.removeEventListener('appinstalled', handleAppInstalled);
    };
  }, [toast]);

  const handleInstall = async () => {
    if (!deferredPrompt) return;

    try {
      await deferredPrompt.prompt();
      const { outcome } = await deferredPrompt.userChoice;
      
      if (outcome === 'accepted') {
        setIsInstallable(false);
        toast({
          title: "Installing SimCore...",
          description: "SimCore is being added to your device."
        });
      }
      
      setDeferredPrompt(null);
    } catch (error) {
      console.error('Install failed:', error);
      toast({
        title: "Installation Failed",
        description: "Could not install SimCore. Please try again.",
        variant: "destructive"
      });
    }
  };

  if (isInstalled) {
    return (
      <Alert className="bg-green-50/50 border-green-200">
        <Download className="h-4 w-4 text-green-600" />
        <AlertDescription className="text-green-800">
          SimCore is installed and ready to use offline!
        </AlertDescription>
      </Alert>
    );
  }

  if (!isInstallable) return null;

  return (
    <Card className="bg-blue-50/50 border-blue-200">
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2 text-blue-800">
          <Download className="h-5 w-5" />
          Install SimCore
        </CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-blue-700 mb-4">
          Install SimCore for faster loading, offline access, and a native app experience.
        </p>
        <Button onClick={handleInstall} className="w-full">
          <Download className="w-4 h-4 mr-2" />
          Install App
        </Button>
      </CardContent>
    </Card>
  );
}

export function OfflineIndicator() {
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [wasOffline, setWasOffline] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      if (wasOffline) {
        toast({
          title: "Back Online",
          description: "Connection restored. All features available."
        });
        setWasOffline(false);
      }
    };

    const handleOffline = () => {
      setIsOnline(false);
      setWasOffline(true);
      toast({
        title: "You're Offline",
        description: "Some features may be limited. Saved simulations are still available.",
        variant: "destructive"
      });
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [wasOffline, toast]);

  return (
    <div className="fixed top-4 right-4 z-50">
      <Badge 
        variant={isOnline ? "default" : "destructive"}
        className="flex items-center gap-1"
      >
        {isOnline ? (
          <>
            <Wifi className="w-3 h-3" />
            Online
          </>
        ) : (
          <>
            <WifiOff className="w-3 h-3" />
            Offline
          </>
        )}
      </Badge>
    </div>
  );
}

export function ServiceWorkerUpdater() {
  const [updateAvailable, setUpdateAvailable] = useState(false);
  const [isUpdating, setIsUpdating] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.ready.then(registration => {
        registration.addEventListener('updatefound', () => {
          const newWorker = registration.installing;
          if (newWorker) {
            newWorker.addEventListener('statechange', () => {
              if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                setUpdateAvailable(true);
              }
            });
          }
        });
      });

      navigator.serviceWorker.addEventListener('controllerchange', () => {
        window.location.reload();
      });
    }
  }, []);

  const handleUpdate = async () => {
    setIsUpdating(true);
    
    try {
      const registration = await navigator.serviceWorker.ready;
      if (registration.waiting) {
        registration.waiting.postMessage({ type: 'SKIP_WAITING' });
        toast({
          title: "Updating...",
          description: "SimCore is being updated to the latest version."
        });
      }
    } catch (error) {
      console.error('Update failed:', error);
      toast({
        title: "Update Failed",
        description: "Could not update SimCore. Please refresh manually.",
        variant: "destructive"
      });
    } finally {
      setIsUpdating(false);
    }
  };

  if (!updateAvailable) return null;

  return (
    <Alert className="fixed bottom-4 right-4 max-w-sm z-50 bg-blue-50 border-blue-200">
      <RefreshCw className="h-4 w-4 text-blue-600" />
      <AlertDescription className="text-blue-800">
        <div className="flex items-center justify-between">
          <span>New version available!</span>
          <Button 
            size="sm" 
            onClick={handleUpdate}
            disabled={isUpdating}
            className="ml-2"
          >
            {isUpdating ? (
              <RefreshCw className="w-3 h-3 animate-spin" />
            ) : (
              'Update'
            )}
          </Button>
        </div>
      </AlertDescription>
    </Alert>
  );
}

export function PWAStatus() {
  const [installPromptEvent, setInstallPromptEvent] = useState<BeforeInstallPromptEvent | null>(null);
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [isInstalled, setIsInstalled] = useState(false);

  useEffect(() => {
    // Check installation status
    const isStandalone = window.matchMedia('(display-mode: standalone)').matches;
    setIsInstalled(isStandalone || (window.navigator as any).standalone);

    // Listen for install prompt
    const handleBeforeInstallPrompt = (e: Event) => {
      e.preventDefault();
      setInstallPromptEvent(e as BeforeInstallPromptEvent);
    };

    // Listen for network status
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return (
    <div className="space-y-4">
      <PWAInstallPrompt />
      <OfflineIndicator />
      <ServiceWorkerUpdater />
    </div>
  );
}