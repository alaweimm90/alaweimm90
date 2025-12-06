import { memo, useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { 
  Accessibility,
  Eye,
  EyeOff,
  Type,
  MousePointer,
  Contrast,
  Volume2,
  VolumeX,
  X,
  Settings,
  Palette,
  Move,
  ZoomIn,
  ZoomOut,
  RotateCcw,
} from "lucide-react";

interface AccessibilityState {
  highContrast: boolean;
  largeText: boolean;
  reducedMotion: boolean;
  screenReaderMode: boolean;
  focusIndicators: boolean;
  soundEnabled: boolean;
  cursorSize: 'normal' | 'large' | 'extra-large';
  textSize: number; // 100 = normal, 150 = 1.5x, 200 = 2x
  colorBlindnessFilter: 'none' | 'protanopia' | 'deuteranopia' | 'tritanopia';
}

const defaultState: AccessibilityState = {
  highContrast: false,
  largeText: false,
  reducedMotion: false,
  screenReaderMode: false,
  focusIndicators: true,
  soundEnabled: true,
  cursorSize: 'normal',
  textSize: 100,
  colorBlindnessFilter: 'none',
};

export const AccessibilityToolbar = memo(() => {
  const [isOpen, setIsOpen] = useState(false);
  const [state, setState] = useState<AccessibilityState>(defaultState);

  // Load saved preferences on mount
  useEffect(() => {
    const saved = localStorage.getItem('accessibility-preferences');
    if (saved) {
      try {
        const parsedState = JSON.parse(saved);
        setState({ ...defaultState, ...parsedState });
      } catch (error) {
        console.warn('[A11y] Failed to parse saved preferences:', error);
      }
    }
  }, []);

  // Save preferences and apply changes
  useEffect(() => {
    localStorage.setItem('accessibility-preferences', JSON.stringify(state));
    applyAccessibilitySettings(state);
  }, [state]);

  // Apply accessibility settings to DOM
  const applyAccessibilitySettings = (settings: AccessibilityState) => {
    const root = document.documentElement;
    const body = document.body;

    // High contrast mode
    if (settings.highContrast) {
      root.classList.add('a11y-high-contrast');
    } else {
      root.classList.remove('a11y-high-contrast');
    }

    // Large text
    if (settings.largeText || settings.textSize > 100) {
      const scale = settings.largeText ? 1.2 : (settings.textSize / 100);
      root.style.setProperty('--a11y-text-scale', scale.toString());
      root.classList.add('a11y-large-text');
    } else {
      root.style.removeProperty('--a11y-text-scale');
      root.classList.remove('a11y-large-text');
    }

    // Reduced motion
    if (settings.reducedMotion) {
      root.classList.add('a11y-reduced-motion');
    } else {
      root.classList.remove('a11y-reduced-motion');
    }

    // Screen reader mode
    if (settings.screenReaderMode) {
      root.classList.add('a11y-screen-reader');
    } else {
      root.classList.remove('a11y-screen-reader');
    }

    // Enhanced focus indicators
    if (settings.focusIndicators) {
      root.classList.add('a11y-enhanced-focus');
    } else {
      root.classList.remove('a11y-enhanced-focus');
    }

    // Cursor size
    root.classList.remove('a11y-cursor-large', 'a11y-cursor-extra-large');
    if (settings.cursorSize === 'large') {
      root.classList.add('a11y-cursor-large');
    } else if (settings.cursorSize === 'extra-large') {
      root.classList.add('a11y-cursor-extra-large');
    }

    // Color blindness filters
    root.classList.remove('a11y-protanopia', 'a11y-deuteranopia', 'a11y-tritanopia');
    if (settings.colorBlindnessFilter !== 'none') {
      root.classList.add(`a11y-${settings.colorBlindnessFilter}`);
    }

    // Announce changes to screen readers
    if (settings.screenReaderMode) {
      announceChange('Accessibility settings updated');
    }
  };

  // Screen reader announcements
  const announceChange = (message: string) => {
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', 'polite');
    announcement.setAttribute('aria-atomic', 'true');
    announcement.className = 'sr-only';
    announcement.textContent = message;
    document.body.appendChild(announcement);
    
    setTimeout(() => {
      document.body.removeChild(announcement);
    }, 1000);
  };

  const updateState = (updates: Partial<AccessibilityState>) => {
    setState(prev => ({ ...prev, ...updates }));
  };

  const resetToDefaults = () => {
    setState(defaultState);
    announceChange('Accessibility settings reset to defaults');
  };

  const playClickSound = () => {
    if (state.soundEnabled) {
      // Simple click sound using Web Audio API
      try {
        const AudioContextClass = window.AudioContext || (window as Window & typeof globalThis & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
        if (!AudioContextClass) return;
        const audioContext = new AudioContextClass();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator.frequency.value = 800;
        oscillator.type = 'sine';
        
        gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);
        
        oscillator.start();
        oscillator.stop(audioContext.currentTime + 0.1);
      } catch (error) {
        // Fallback - no sound
      }
    }
  };

  const handleButtonClick = (action: () => void) => {
    playClickSound();
    action();
  };

  return (
    <div className="fixed bottom-4 right-4 z-50">
      {/* Accessibility Toggle Button */}
      <Button
        variant="default"
        size="lg"
        className="rounded-full shadow-lg bg-primary hover:bg-primary/90 transition-all duration-200"
        onClick={() => handleButtonClick(() => setIsOpen(!isOpen))}
        aria-label={isOpen ? "Close accessibility toolbar" : "Open accessibility toolbar"}
        aria-expanded={isOpen}
        aria-controls="accessibility-toolbar"
      >
        <Accessibility className="h-6 w-6" />
      </Button>

      {/* Accessibility Toolbar Panel */}
      {isOpen && (
        <Card 
          id="accessibility-toolbar"
          className="absolute bottom-16 right-0 w-80 max-w-[calc(100vw-2rem)] p-4 shadow-xl border-2"
          role="dialog"
          aria-label="Accessibility Settings"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Accessibility className="h-5 w-5 text-primary" />
              <h2 className="text-lg font-semibold">Accessibility</h2>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => handleButtonClick(() => setIsOpen(false))}
              aria-label="Close accessibility toolbar"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>

          <div className="space-y-4 max-h-96 overflow-y-auto">
            {/* Vision Section */}
            <div>
              <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
                <Eye className="h-4 w-4" />
                Vision
              </h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label htmlFor="high-contrast" className="text-sm">High Contrast</Label>
                  <Switch
                    id="high-contrast"
                    checked={state.highContrast}
                    onCheckedChange={(checked) => 
                      handleButtonClick(() => updateState({ highContrast: checked }))
                    }
                    aria-describedby="high-contrast-desc"
                  />
                </div>
                <p id="high-contrast-desc" className="text-xs text-muted-foreground">
                  Increases contrast for better visibility
                </p>

                <div className="flex items-center justify-between">
                  <Label htmlFor="large-text" className="text-sm">Large Text</Label>
                  <Switch
                    id="large-text"
                    checked={state.largeText}
                    onCheckedChange={(checked) => 
                      handleButtonClick(() => updateState({ largeText: checked }))
                    }
                  />
                </div>

                <div className="space-y-2">
                  <Label className="text-sm">Text Size: {state.textSize}%</Label>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleButtonClick(() => 
                        updateState({ textSize: Math.max(75, state.textSize - 25) })
                      )}
                      disabled={state.textSize <= 75}
                      aria-label="Decrease text size"
                    >
                      <ZoomOut className="h-3 w-3" />
                    </Button>
                    <div className="flex-1 text-center text-sm">
                      {state.textSize}%
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleButtonClick(() => 
                        updateState({ textSize: Math.min(200, state.textSize + 25) })
                      )}
                      disabled={state.textSize >= 200}
                      aria-label="Increase text size"
                    >
                      <ZoomIn className="h-3 w-3" />
                    </Button>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label className="text-sm">Color Blindness Filter</Label>
                  <select
                    value={state.colorBlindnessFilter}
                    onChange={(e) => handleButtonClick(() =>
                      updateState({ colorBlindnessFilter: e.target.value as 'none' | 'protanopia' | 'deuteranopia' | 'tritanopia' })
                    )}
                    className="w-full p-2 text-sm rounded border border-border bg-background"
                    aria-describedby="colorblind-desc"
                  >
                    <option value="none">None</option>
                    <option value="protanopia">Protanopia (Red-blind)</option>
                    <option value="deuteranopia">Deuteranopia (Green-blind)</option>
                    <option value="tritanopia">Tritanopia (Blue-blind)</option>
                  </select>
                  <p id="colorblind-desc" className="text-xs text-muted-foreground">
                    Simulates color vision deficiencies
                  </p>
                </div>
              </div>
            </div>

            <Separator />

            {/* Motor Section */}
            <div>
              <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
                <MousePointer className="h-4 w-4" />
                Motor
              </h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label htmlFor="reduced-motion" className="text-sm">Reduce Motion</Label>
                  <Switch
                    id="reduced-motion"
                    checked={state.reducedMotion}
                    onCheckedChange={(checked) => 
                      handleButtonClick(() => updateState({ reducedMotion: checked }))
                    }
                  />
                </div>

                <div className="space-y-2">
                  <Label className="text-sm">Cursor Size</Label>
                  <select
                    value={state.cursorSize}
                    onChange={(e) => handleButtonClick(() =>
                      updateState({ cursorSize: e.target.value as 'normal' | 'large' | 'extra-large' })
                    )}
                    className="w-full p-2 text-sm rounded border border-border bg-background"
                  >
                    <option value="normal">Normal</option>
                    <option value="large">Large</option>
                    <option value="extra-large">Extra Large</option>
                  </select>
                </div>

                <div className="flex items-center justify-between">
                  <Label htmlFor="focus-indicators" className="text-sm">Enhanced Focus</Label>
                  <Switch
                    id="focus-indicators"
                    checked={state.focusIndicators}
                    onCheckedChange={(checked) => 
                      handleButtonClick(() => updateState({ focusIndicators: checked }))
                    }
                  />
                </div>
              </div>
            </div>

            <Separator />

            {/* Cognitive Section */}
            <div>
              <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
                <Settings className="h-4 w-4" />
                Cognitive
              </h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label htmlFor="screen-reader" className="text-sm">Screen Reader Mode</Label>
                  <Switch
                    id="screen-reader"
                    checked={state.screenReaderMode}
                    onCheckedChange={(checked) => 
                      handleButtonClick(() => updateState({ screenReaderMode: checked }))
                    }
                  />
                </div>

                <div className="flex items-center justify-between">
                  <Label htmlFor="sound-enabled" className="text-sm flex items-center gap-2">
                    {state.soundEnabled ? <Volume2 className="h-3 w-3" /> : <VolumeX className="h-3 w-3" />}
                    Sound Feedback
                  </Label>
                  <Switch
                    id="sound-enabled"
                    checked={state.soundEnabled}
                    onCheckedChange={(checked) => 
                      handleButtonClick(() => updateState({ soundEnabled: checked }))
                    }
                  />
                </div>
              </div>
            </div>

            <Separator />

            {/* Reset Button */}
            <Button
              variant="outline"
              size="sm"
              onClick={() => handleButtonClick(resetToDefaults)}
              className="w-full"
              aria-describedby="reset-desc"
            >
              <RotateCcw className="h-4 w-4 mr-2" />
              Reset to Defaults
            </Button>
            <p id="reset-desc" className="text-xs text-muted-foreground">
              Restore all accessibility settings to their default values
            </p>
          </div>
        </Card>
      )}
    </div>
  );
});

AccessibilityToolbar.displayName = 'AccessibilityToolbar';