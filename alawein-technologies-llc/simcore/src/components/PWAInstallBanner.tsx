import React, { useState, useEffect } from 'react';
import { usePWAInstall, useNetworkStatus } from './PWAEnhancements';
import { useResponsiveEnhanced } from '@/hooks/use-responsive-enhanced';
import { cn } from '@/lib/utils';
import { Download, X, RefreshCw, Wifi, WifiOff } from 'lucide-react';

export function PWAInstallBanner() {
  const { install, isInstalled, isInstallable } = usePWAInstall();
  const { isOnline } = useNetworkStatus();
  const [updateAvailable] = useState(false); // Simplified for now
  const { isMobile } = useResponsiveEnhanced();
  
  const [showBanner, setShowBanner] = useState(false);
  const [showUpdateBanner, setShowUpdateBanner] = useState(false);
  const [dismissed, setDismissed] = useState(false);

  // Show install banner after delay if installable and not dismissed
  useEffect(() => {
    if (isInstallable && !isInstalled && !dismissed) {
      const timer = setTimeout(() => setShowBanner(true), 3000);
      return () => clearTimeout(timer);
    }
  }, [isInstallable, isInstalled, dismissed]);

  // Show update banner when update is available
  useEffect(() => {
    if (updateAvailable) {
      setShowUpdateBanner(true);
    }
  }, [updateAvailable]);

  const handleInstall = async () => {
    const success = await install();
    if (success) {
      setShowBanner(false);
    }
  };

  const handleDismiss = () => {
    setShowBanner(false);
    setDismissed(true);
    // Remember dismissal for current session
    sessionStorage.setItem('pwa-banner-dismissed', 'true');
  };

  const handleUpdate = () => {
    // Simplified update logic
    setShowUpdateBanner(false);
  };

  // Check if banner was dismissed in current session
  useEffect(() => {
    const wasDismissed = sessionStorage.getItem('pwa-banner-dismissed');
    if (wasDismissed) {
      setDismissed(true);
    }
  }, []);

  // If already installed, show simple online/offline status
  if (isInstalled) {
    return (
      <div className={cn(
        'fixed top-4 right-4 z-50 transition-all duration-300',
        isMobile ? 'top-2 right-2' : 'top-4 right-4'
      )}>
        <div className={cn(
          'flex items-center gap-2 px-3 py-2 rounded-lg shadow-lg backdrop-blur-sm border',
          isOnline 
            ? 'bg-success/10 border-success/20 text-success' 
            : 'bg-destructive/10 border-destructive/20 text-destructive'
        )}>
          {isOnline ? (
            <Wifi className="w-4 h-4" />
          ) : (
            <WifiOff className="w-4 h-4" />
          )}
          <span className="text-xs font-medium">
            {isOnline ? 'Online' : 'Offline'}
          </span>
        </div>
      </div>
    );
  }

  return (
    <>
      {/* PWA Install Banner */}
      {showBanner && (
        <div className={cn(
          'fixed bottom-4 left-4 right-4 z-50 transform transition-all duration-300',
          isMobile ? 'bottom-20' : 'bottom-4 left-4 right-auto w-96'
        )}>
          <div className="bg-card border border-border rounded-lg shadow-lg p-4">
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
                <Download className="w-5 h-5 text-primary" />
              </div>
              
              <div className="flex-1 min-w-0">
                <h3 className="font-semibold text-sm text-foreground mb-1">
                  Install SimCore
                </h3>
                <p className="text-xs text-muted-foreground mb-3">
                  Get the full app experience with offline access and faster loading.
                </p>
                
                <div className="flex gap-2">
                  <button
                    onClick={handleInstall}
                    className="px-3 py-2 bg-primary text-primary-foreground text-xs font-medium rounded-md hover:bg-primary/90 transition-colors"
                  >
                    Install App
                  </button>
                  <button
                    onClick={handleDismiss}
                    className="px-3 py-2 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
                  >
                    Later
                  </button>
                </div>
              </div>
              
              <button
                onClick={handleDismiss}
                className="flex-shrink-0 w-6 h-6 flex items-center justify-center text-muted-foreground hover:text-foreground transition-colors"
                aria-label="Dismiss"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* PWA Update Banner */}
      {showUpdateBanner && (
        <div className={cn(
          'fixed top-4 left-4 right-4 z-50 transform transition-all duration-300',
          isMobile ? 'top-2 left-2 right-2' : 'top-4 left-4 right-auto w-96'
        )}>
          <div className="bg-card border border-border rounded-lg shadow-lg p-4">
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
                <RefreshCw className="w-5 h-5 text-primary" />
              </div>
              
              <div className="flex-1 min-w-0">
                <h3 className="font-semibold text-sm text-foreground mb-1">
                  Update Available
                </h3>
                <p className="text-xs text-muted-foreground mb-3">
                  A new version of SimCore is ready with improvements and fixes.
                </p>
                
                <div className="flex gap-2">
                  <button
                    onClick={handleUpdate}
                    className="px-3 py-2 bg-primary text-primary-foreground text-xs font-medium rounded-md hover:bg-primary/90 transition-colors"
                  >
                    Update Now
                  </button>
                  <button
                    onClick={() => setShowUpdateBanner(false)}
                    className="px-3 py-2 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
                  >
                    Later
                  </button>
                </div>
              </div>
              
              <button
                onClick={() => setShowUpdateBanner(false)}
                className="flex-shrink-0 w-6 h-6 flex items-center justify-center text-muted-foreground hover:text-foreground transition-colors"
                aria-label="Dismiss"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Offline Banner */}
      {!isOnline && (
        <div className={cn(
          'fixed top-0 left-0 right-0 z-50 bg-destructive text-destructive-foreground',
          'border-b border-destructive/20'
        )}>
          <div className="container mx-auto px-4 py-2">
            <div className="flex items-center justify-center gap-2 text-sm">
              <WifiOff className="w-4 h-4" />
              <span>You're offline. Some features may be limited.</span>
            </div>
          </div>
        </div>
      )}
    </>
  );
}