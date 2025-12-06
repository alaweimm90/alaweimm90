/**
 * Accessibility Enhancement System
 * Provides comprehensive accessibility features for scientific simulations
 */

import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Eye, 
  Ear, 
  MousePointer, 
  Keyboard,
  Volume2,
  VolumeX,
  ZoomIn,
  ZoomOut,
  Contrast,
  Settings
} from 'lucide-react';

interface AccessibilitySettings {
  highContrast: boolean;
  largeText: boolean;
  reducedMotion: boolean;
  soundEnabled: boolean;
  keyboardNavigation: boolean;
  screenReaderOptimized: boolean;
  focusIndicators: boolean;
  colorBlindAssist: boolean;
  fontSize: number;
  zoomLevel: number;
}

interface AccessibilityEnhancementSystemProps {
  className?: string;
  onSettingsChange?: (settings: AccessibilitySettings) => void;
}

export function AccessibilityEnhancementSystem({ 
  className, 
  onSettingsChange 
}: AccessibilityEnhancementSystemProps) {
  const [settings, setSettings] = useState<AccessibilitySettings>({
    highContrast: false,
    largeText: false,
    reducedMotion: false,
    soundEnabled: true,
    keyboardNavigation: true,
    screenReaderOptimized: false,
    focusIndicators: true,
    colorBlindAssist: false,
    fontSize: 100,
    zoomLevel: 100
  });

  const [isActive, setIsActive] = useState(false);
  const announcementRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Load settings from localStorage
    const savedSettings = localStorage.getItem('accessibility-settings');
    if (savedSettings) {
      const parsed = JSON.parse(savedSettings);
      setSettings(parsed);
      applySettings(parsed);
    }

    // Check for system preferences
    checkSystemPreferences();
  }, []);

  useEffect(() => {
    if (isActive) {
      applySettings(settings);
      localStorage.setItem('accessibility-settings', JSON.stringify(settings));
      onSettingsChange?.(settings);
    }
  }, [settings, isActive, onSettingsChange]);

  const checkSystemPreferences = () => {
    // Check for reduced motion preference
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setSettings(prev => ({ ...prev, reducedMotion: true }));
    }

    // Check for high contrast preference
    if (window.matchMedia('(prefers-contrast: high)').matches) {
      setSettings(prev => ({ ...prev, highContrast: true }));
    }

    // Check for color scheme preference
    if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
      document.documentElement.classList.add('dark');
    }
  };

  const applySettings = (newSettings: AccessibilitySettings) => {
    const root = document.documentElement;

    // High contrast
    if (newSettings.highContrast) {
      root.classList.add('high-contrast');
    } else {
      root.classList.remove('high-contrast');
    }

    // Large text
    if (newSettings.largeText) {
      root.classList.add('large-text');
    } else {
      root.classList.remove('large-text');
    }

    // Reduced motion
    if (newSettings.reducedMotion) {
      root.classList.add('reduce-motion');
    } else {
      root.classList.remove('reduce-motion');
    }

    // Focus indicators
    if (newSettings.focusIndicators) {
      root.classList.add('enhanced-focus');
    } else {
      root.classList.remove('enhanced-focus');
    }

    // Color blind assistance
    if (newSettings.colorBlindAssist) {
      root.classList.add('colorblind-assist');
    } else {
      root.classList.remove('colorblind-assist');
    }

    // Font size
    root.style.setProperty('--font-scale', `${newSettings.fontSize / 100}`);

    // Zoom level
    root.style.setProperty('--zoom-scale', `${newSettings.zoomLevel / 100}`);
  };

  const updateSetting = <K extends keyof AccessibilitySettings>(
    key: K, 
    value: AccessibilitySettings[K]
  ) => {
    setSettings(prev => ({ ...prev, [key]: value }));
    announceChange(`${key} ${value ? 'enabled' : 'disabled'}`);
  };

  const announceChange = (message: string) => {
    if (announcementRef.current) {
      announcementRef.current.textContent = message;
    }
  };

  const resetSettings = () => {
    const defaultSettings: AccessibilitySettings = {
      highContrast: false,
      largeText: false,
      reducedMotion: false,
      soundEnabled: true,
      keyboardNavigation: true,
      screenReaderOptimized: false,
      focusIndicators: true,
      colorBlindAssist: false,
      fontSize: 100,
      zoomLevel: 100
    };
    setSettings(defaultSettings);
    announceChange('Settings reset to defaults');
  };

  return (
    <div className={`space-y-[var(--semantic-spacing-card)] ${className}`}>
      {/* Screen reader announcements */}
      <div 
        ref={announcementRef}
        className="sr-only"
        aria-live="polite"
        aria-atomic="true"
      />

      {/* Activation Toggle */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-[var(--semantic-spacing-inline)]">
            <Eye className="h-5 w-5" />
            Accessibility Enhancement System
          </CardTitle>
          <CardDescription>
            Customize accessibility features for better simulation experience
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <span className="font-medium">Enable Accessibility Features</span>
            <Switch
              checked={isActive}
              onCheckedChange={setIsActive}
              aria-label="Toggle accessibility features"
            />
          </div>
        </CardContent>
      </Card>

      {isActive && (
        <Tabs defaultValue="visual" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="visual">Visual</TabsTrigger>
            <TabsTrigger value="motor">Motor</TabsTrigger>
            <TabsTrigger value="cognitive">Cognitive</TabsTrigger>
            <TabsTrigger value="audio">Audio</TabsTrigger>
          </TabsList>

          <TabsContent value="visual" className="space-y-[var(--semantic-spacing-inline)]">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-[var(--semantic-spacing-inline)]">
                  <Eye className="h-4 w-4" />
                  Visual Accessibility
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-[var(--semantic-spacing-card)]">
                <div className="flex items-center justify-between">
                  <div>
                    <label className="font-medium">High Contrast Mode</label>
                    <p className="text-sm text-muted-foreground">Increases contrast for better visibility</p>
                  </div>
                  <Switch
                    checked={settings.highContrast}
                    onCheckedChange={(value) => updateSetting('highContrast', value)}
                    aria-label="Toggle high contrast mode"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <label className="font-medium">Large Text</label>
                    <p className="text-sm text-muted-foreground">Increases text size throughout the application</p>
                  </div>
                  <Switch
                    checked={settings.largeText}
                    onCheckedChange={(value) => updateSetting('largeText', value)}
                    aria-label="Toggle large text"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <label className="font-medium">Color Blind Assistance</label>
                    <p className="text-sm text-muted-foreground">Adds patterns and symbols to distinguish colors</p>
                  </div>
                  <Switch
                    checked={settings.colorBlindAssist}
                    onCheckedChange={(value) => updateSetting('colorBlindAssist', value)}
                    aria-label="Toggle color blind assistance"
                  />
                </div>

                <div className="space-y-2">
                  <label className="font-medium flex items-center gap-2">
                    <ZoomIn className="h-4 w-4" />
                    Font Size: {settings.fontSize}%
                  </label>
                  <Slider
                    value={[settings.fontSize]}
                    onValueChange={([value]) => updateSetting('fontSize', value)}
                    min={75}
                    max={200}
                    step={25}
                    className="w-full"
                    aria-label="Adjust font size"
                  />
                </div>

                <div className="space-y-2">
                  <label className="font-medium flex items-center gap-2">
                    <ZoomOut className="h-4 w-4" />
                    Zoom Level: {settings.zoomLevel}%
                  </label>
                  <Slider
                    value={[settings.zoomLevel]}
                    onValueChange={([value]) => updateSetting('zoomLevel', value)}
                    min={50}
                    max={200}
                    step={25}
                    className="w-full"
                    aria-label="Adjust zoom level"
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="motor" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MousePointer className="h-4 w-4" />
                  Motor Accessibility
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <label className="font-medium">Enhanced Keyboard Navigation</label>
                    <p className="text-sm text-muted-foreground">Improved keyboard shortcuts and navigation</p>
                  </div>
                  <Switch
                    checked={settings.keyboardNavigation}
                    onCheckedChange={(value) => updateSetting('keyboardNavigation', value)}
                    aria-label="Toggle enhanced keyboard navigation"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <label className="font-medium">Enhanced Focus Indicators</label>
                    <p className="text-sm text-muted-foreground">Makes focus states more visible</p>
                  </div>
                  <Switch
                    checked={settings.focusIndicators}
                    onCheckedChange={(value) => updateSetting('focusIndicators', value)}
                    aria-label="Toggle enhanced focus indicators"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <label className="font-medium">Reduced Motion</label>
                    <p className="text-sm text-muted-foreground">Minimizes animations and transitions</p>
                  </div>
                  <Switch
                    checked={settings.reducedMotion}
                    onCheckedChange={(value) => updateSetting('reducedMotion', value)}
                    aria-label="Toggle reduced motion"
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="cognitive" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="h-4 w-4" />
                  Cognitive Accessibility
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <label className="font-medium">Screen Reader Optimization</label>
                    <p className="text-sm text-muted-foreground">Optimizes content for screen readers</p>
                  </div>
                  <Switch
                    checked={settings.screenReaderOptimized}
                    onCheckedChange={(value) => updateSetting('screenReaderOptimized', value)}
                    aria-label="Toggle screen reader optimization"
                  />
                </div>

                <div className="space-y-2">
                  <h4 className="font-medium">Keyboard Shortcuts</h4>
                  <div className="space-y-1 text-sm text-muted-foreground">
                    <div>• <kbd className="px-2 py-1 bg-muted rounded">Tab</kbd> - Navigate forward</div>
                    <div>• <kbd className="px-2 py-1 bg-muted rounded">Shift + Tab</kbd> - Navigate backward</div>
                    <div>• <kbd className="px-2 py-1 bg-muted rounded">Enter</kbd> - Activate button/link</div>
                    <div>• <kbd className="px-2 py-1 bg-muted rounded">Space</kbd> - Toggle checkbox/switch</div>
                    <div>• <kbd className="px-2 py-1 bg-muted rounded">Escape</kbd> - Close dialog/menu</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="audio" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Ear className="h-4 w-4" />
                  Audio Accessibility
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <label className="font-medium">Sound Feedback</label>
                    <p className="text-sm text-muted-foreground">Audio feedback for interactions</p>
                  </div>
                  <Switch
                    checked={settings.soundEnabled}
                    onCheckedChange={(value) => updateSetting('soundEnabled', value)}
                    aria-label="Toggle sound feedback"
                  />
                </div>

                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      const utterance = new SpeechSynthesisUtterance('Accessibility system is working correctly');
                      speechSynthesis.speak(utterance);
                    }}
                    className="flex items-center gap-2"
                  >
                    <Volume2 className="h-4 w-4" />
                    Test Speech
                  </Button>
                  <Badge variant="outline">Screen reader compatible</Badge>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      )}

      {isActive && (
        <Card>
          <CardContent className="pt-6">
            <Button
              onClick={resetSettings}
              variant="outline"
              className="w-full"
            >
              Reset to Defaults
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

export default AccessibilityEnhancementSystem;