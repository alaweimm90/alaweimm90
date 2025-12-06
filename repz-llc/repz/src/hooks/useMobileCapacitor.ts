import { useEffect, useState } from 'react';
import { Capacitor } from '@capacitor/core';
import { SplashScreen } from '@capacitor/splash-screen';
import { StatusBar, Style } from '@capacitor/status-bar';
import { Haptics, ImpactStyle, NotificationType } from '@capacitor/haptics';
import { Device } from '@capacitor/device';
import { Network } from '@capacitor/network';

export interface DeviceInfo {
  platform: string;
  isNative: boolean;
  model: string;
  operatingSystem: string;
  osVersion: string;
  manufacturer: string;
  isVirtual: boolean;
}

export interface NetworkStatus {
  connected: boolean;
  connectionType: string;
}

export const useMobileCapacitor = () => {
  const [deviceInfo, setDeviceInfo] = useState<DeviceInfo | null>(null);
  const [networkStatus, setNetworkStatus] = useState<NetworkStatus>({
    connected: true,
    connectionType: 'unknown'
  });

  useEffect(() => {
    const initCapacitor = async () => {
      if (Capacitor.isNativePlatform()) {
        try {
          // Initialize splash screen
          await SplashScreen.hide();

          // Configure status bar
          await StatusBar.setStyle({ style: Style.Dark });
          await StatusBar.setBackgroundColor({ color: '#1a1a1a' });

          // Get device info
          const info = await Device.getInfo();
          setDeviceInfo({
            platform: Capacitor.getPlatform(),
            isNative: Capacitor.isNativePlatform(),
            model: info.model,
            operatingSystem: info.operatingSystem,
            osVersion: info.osVersion,
            manufacturer: info.manufacturer,
            isVirtual: info.isVirtual
          });

          // Get network status
          const status = await Network.getStatus();
          setNetworkStatus({
            connected: status.connected,
            connectionType: status.connectionType
          });

          // Listen for network changes
          const networkListener = await Network.addListener('networkStatusChange', (status) => {
            setNetworkStatus({
              connected: status.connected,
              connectionType: status.connectionType
            });
          });

          return () => {
            networkListener.remove();
          };
        } catch (error) {
          console.error('Error initializing Capacitor:', error);
        }
      }
    };

    initCapacitor();
  }, []);

  const hapticFeedback = {
    light: () => {
      if (Capacitor.isNativePlatform()) {
        Haptics.impact({ style: ImpactStyle.Light });
      }
    },
    medium: () => {
      if (Capacitor.isNativePlatform()) {
        Haptics.impact({ style: ImpactStyle.Medium });
      }
    },
    heavy: () => {
      if (Capacitor.isNativePlatform()) {
        Haptics.impact({ style: ImpactStyle.Heavy });
      }
    },
    success: () => {
      if (Capacitor.isNativePlatform()) {
        Haptics.notification({ type: NotificationType.Success });
      }
    },
    warning: () => {
      if (Capacitor.isNativePlatform()) {
        Haptics.notification({ type: NotificationType.Warning });
      }
    },
    error: () => {
      if (Capacitor.isNativePlatform()) {
        Haptics.notification({ type: NotificationType.Error });
      }
    }
  };

  const statusBar = {
    show: async () => {
      if (Capacitor.isNativePlatform()) {
        await StatusBar.show();
      }
    },
    hide: async () => {
      if (Capacitor.isNativePlatform()) {
        await StatusBar.hide();
      }
    },
    setLight: async () => {
      if (Capacitor.isNativePlatform()) {
        await StatusBar.setStyle({ style: Style.Light });
      }
    },
    setDark: async () => {
      if (Capacitor.isNativePlatform()) {
        await StatusBar.setStyle({ style: Style.Dark });
      }
    }
  };

  return {
    deviceInfo,
    networkStatus,
    isNative: Capacitor.isNativePlatform(),
    platform: Capacitor.getPlatform(),
    hapticFeedback,
    statusBar
  };
};