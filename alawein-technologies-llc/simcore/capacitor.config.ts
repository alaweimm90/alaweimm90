import { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'app.lovable.f5c4348eaff94eb4bf030c2e06c7822a',
  appName: 'simcore-explorer',
  webDir: 'dist',
  server: {
    url: 'https://f5c4348e-aff9-4eb4-bf03-0c2e06c7822a.lovableproject.com?forceHideBadge=true',
    cleartext: true
  },
  plugins: {
    SplashScreen: {
      launchShowDuration: 2000,
      launchAutoHide: true,
      backgroundColor: "#000000",
      androidSplashResourceName: "splash",
      androidScaleType: "CENTER_CROP",
      showSpinner: false,
      androidSpinnerStyle: "large",
      iosSpinnerStyle: "small",
      spinnerColor: "#999999",
      splashFullScreen: true,
      splashImmersive: true,
      layoutName: "launch_screen",
      useDialog: true,
    },
    StatusBar: {
      style: "dark",
      backgroundColor: "#000000"
    }
  }
};

export default config;