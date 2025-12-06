import { useState, useEffect, useCallback } from 'react';
import { trackQuantumEvents } from '@/lib/analytics';
import { logger } from '@/lib/logger';

interface BeforeInstallPromptEvent extends Event {
  prompt(): Promise<void>;
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>;
}

interface PWAInstallState {
  isInstallable: boolean;
  isInstalled: boolean;
  canInstall: boolean;
  installPrompt: BeforeInstallPromptEvent | null;
}

export const usePWAInstall = () => {
  const [state, setState] = useState<PWAInstallState>({
    isInstallable: false,
    isInstalled: false,
    canInstall: false,
    installPrompt: null
  });

  const checkInstallationStatus = useCallback(() => {
    // Check if PWA is already installed
    const isStandalone = window.matchMedia('(display-mode: standalone)').matches;
    const isInWebAppiOS = (window.navigator as any).standalone === true;
    const isInstalled = isStandalone || isInWebAppiOS;

    setState(prev => ({
      ...prev,
      isInstalled
    }));
  }, []);

  const showInstallPrompt = useCallback(async () => {
    if (!state.installPrompt) {
      trackQuantumEvents.featureDiscovery('pwa_install_no_prompt');
      return false;
    }

    try {
      // Show the install prompt
      await state.installPrompt.prompt();
      
      // Wait for user choice
      const choiceResult = await state.installPrompt.userChoice;
      
      if (choiceResult.outcome === 'accepted') {
        trackQuantumEvents.featureDiscovery('pwa_install_accepted');
        setState(prev => ({
          ...prev,
          isInstalled: true,
          canInstall: false,
          installPrompt: null
        }));
        return true;
      } else {
        trackQuantumEvents.featureDiscovery('pwa_install_dismissed');
        return false;
      }
    } catch (error) {
      logger.error('Error showing install prompt', { error });
      trackQuantumEvents.errorBoundary(
        'PWA install prompt failed',
        (error as Error).stack || 'No stack trace',
        'pwa-install'
      );
      return false;
    }
  }, [state.installPrompt]);

  useEffect(() => {
    checkInstallationStatus();

    const handleBeforeInstallPrompt = (e: Event) => {
      // Prevent the default mini-infobar from appearing
      e.preventDefault();
      
      const installEvent = e as BeforeInstallPromptEvent;
      
      setState(prev => ({
        ...prev,
        isInstallable: true,
        canInstall: true,
        installPrompt: installEvent
      }));

      trackQuantumEvents.featureDiscovery('pwa_install_prompt_available');
    };

    const handleAppInstalled = () => {
      setState(prev => ({
        ...prev,
        isInstalled: true,
        canInstall: false,
        installPrompt: null
      }));

      trackQuantumEvents.featureDiscovery('pwa_install_completed');
    };

    // Listen for install prompt
    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
    window.addEventListener('appinstalled', handleAppInstalled);

    // Check display mode changes
    const displayModeQuery = window.matchMedia('(display-mode: standalone)');
    const handleDisplayModeChange = () => checkInstallationStatus();
    displayModeQuery.addEventListener('change', handleDisplayModeChange);

    return () => {
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
      window.removeEventListener('appinstalled', handleAppInstalled);
      displayModeQuery.removeEventListener('change', handleDisplayModeChange);
    };
  }, [checkInstallationStatus]);

  return {
    ...state,
    showInstallPrompt,
    checkInstallationStatus
  };
};

// Offline support hook
export const useOfflineSupport = () => {
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [hasBeenOffline, setHasBeenOffline] = useState(false);

  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      if (hasBeenOffline) {
        trackQuantumEvents.preferenceChange('connectivity', 'online', 'offline');
      }
    };

    const handleOffline = () => {
      setIsOnline(false);
      setHasBeenOffline(true);
      trackQuantumEvents.preferenceChange('connectivity', 'offline', 'online');
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [hasBeenOffline]);

  return {
    isOnline,
    hasBeenOffline
  };
};