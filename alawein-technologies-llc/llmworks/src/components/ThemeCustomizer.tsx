import { memo, useState, useEffect } from 'react';
import { Settings, Palette, Eye, Moon, Sun, Monitor, Zap, Sparkles, Atom } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';

interface ThemeSettings {
  colorScheme: 'tactical' | 'neural' | 'cyber' | 'stealth';
  intensity: number; // 0-100
  glassMorphism: number; // 0-100
  animations: boolean;
  particleEffects: boolean;
  dynamicBackground: boolean;
  soundEffects: boolean;
  accessibility: {
    reducedMotion: boolean;
    highContrast: boolean;
    largeText: boolean;
  };
}

interface ThemeCustomizerProps {
  className?: string;
  isOpen?: boolean;
  onToggle?: () => void;
}

const ThemeCustomizerComponent = ({ 
  className = '', 
  isOpen = false, 
  onToggle 
}: ThemeCustomizerProps) => {
  const [settings, setSettings] = useState<ThemeSettings>({
    colorScheme: 'tactical',
    intensity: 75,
    glassMorphism: 60,
    animations: true,
    particleEffects: true,
    dynamicBackground: true,
    soundEffects: false,
    accessibility: {
      reducedMotion: false,
      highContrast: false,
      largeText: false
    }
  });

  const [activePreview, setActivePreview] = useState<string | null>(null);

  const colorSchemes = {
    tactical: {
      name: 'Tactical Blue',
      description: 'Strategic intelligence theme with tactical blues and cyan accents',
      colors: {
        primary: '#2563EB',
        secondary: '#0891B2', 
        accent: '#B45309',
        background: '#0A0A0B'
      },
      icon: Eye
    },
    neural: {
      name: 'Neural Network',
      description: 'AI-inspired theme with purple and pink neural pathways',
      colors: {
        primary: '#8B5CF6',
        secondary: '#EC4899',
        accent: '#10B981',
        background: '#0B0A0F'
      },
      icon: Atom
    },
    cyber: {
      name: 'Cyber Matrix',
      description: 'Cyberpunk theme with neon greens and electric blues',
      colors: {
        primary: '#06B6D4',
        secondary: '#84CC16',
        accent: '#F97316',
        background: '#0A0F0B'
      },
      icon: Zap
    },
    stealth: {
      name: 'Stealth Mode',
      description: 'Dark operational theme for low-light environments',
      colors: {
        primary: '#6B7280',
        secondary: '#374151',
        accent: '#F59E0B',
        background: '#030303'
      },
      icon: Moon
    }
  };

  // Load settings from localStorage
  useEffect(() => {
    const savedSettings = localStorage.getItem('themeSettings');
    if (savedSettings) {
      try {
        const parsed = JSON.parse(savedSettings);
        setSettings({ ...settings, ...parsed });
      } catch (error) {
        console.warn('Failed to load theme settings:', error);
      }
    }
  }, []);

  // Save settings to localStorage
  useEffect(() => {
    localStorage.setItem('themeSettings', JSON.stringify(settings));
    applyTheme(settings);
  }, [settings]);

  // Add keyboard and click outside handling
  useEffect(() => {
    if (!isOpen) return;

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && onToggle) {
        onToggle();
      }
    };

    const handleClickOutside = (e: MouseEvent) => {
      const sidebar = document.getElementById('theme-customizer-sidebar');
      if (sidebar && !sidebar.contains(e.target as Node) && onToggle) {
        onToggle();
      }
    };

    document.addEventListener('keydown', handleEscape);
    document.addEventListener('mousedown', handleClickOutside);

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen, onToggle]);

  const applyTheme = (themeSettings: ThemeSettings) => {
    const root = document.documentElement;
    const scheme = colorSchemes[themeSettings.colorScheme];

    // Apply color scheme
    root.style.setProperty('--color-primary', scheme.colors.primary);
    root.style.setProperty('--color-secondary', scheme.colors.secondary);
    root.style.setProperty('--color-accent', scheme.colors.accent);
    root.style.setProperty('--color-background', scheme.colors.background);

    // Apply intensity (affects opacity and glow effects)
    const intensityFactor = themeSettings.intensity / 100;
    root.style.setProperty('--theme-intensity', intensityFactor.toString());

    // Apply glass morphism level
    const glassFactor = themeSettings.glassMorphism / 100;
    root.style.setProperty('--glass-opacity', (glassFactor * 0.2).toString());
    root.style.setProperty('--glass-blur', `${glassFactor * 20}px`);

    // Apply accessibility settings
    if (themeSettings.accessibility.reducedMotion) {
      root.style.setProperty('--animation-duration', '0s');
    } else {
      root.style.removeProperty('--animation-duration');
    }

    if (themeSettings.accessibility.largeText) {
      root.style.setProperty('--font-scale', '1.2');
    } else {
      root.style.removeProperty('--font-scale');
    }

    // Apply feature toggles to body classes
    document.body.classList.toggle('animations-disabled', !themeSettings.animations);
    document.body.classList.toggle('particles-disabled', !themeSettings.particleEffects);
    document.body.classList.toggle('dynamic-bg-disabled', !themeSettings.dynamicBackground);
    document.body.classList.toggle('high-contrast', themeSettings.accessibility.highContrast);
  };

  const updateSetting = <K extends keyof ThemeSettings>(
    key: K, 
    value: ThemeSettings[K]
  ) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  const updateAccessibility = <K extends keyof ThemeSettings['accessibility']>(
    key: K,
    value: ThemeSettings['accessibility'][K]
  ) => {
    setSettings(prev => ({
      ...prev,
      accessibility: { ...prev.accessibility, [key]: value }
    }));
  };

  const resetToDefaults = () => {
    const defaultSettings: ThemeSettings = {
      colorScheme: 'tactical',
      intensity: 75,
      glassMorphism: 60,
      animations: true,
      particleEffects: true,
      dynamicBackground: true,
      soundEffects: false,
      accessibility: {
        reducedMotion: false,
        highContrast: false,
        largeText: false
      }
    };
    setSettings(defaultSettings);
  };

  const previewScheme = (scheme: string) => {
    setActivePreview(scheme);
    setTimeout(() => setActivePreview(null), 2000);
  };

  if (!isOpen) {
    return (
      <Button
        onClick={onToggle}
        variant="outline"
        size="sm"
        className="glass-minimal border-primary/30 hover:bg-primary/10 fixed bottom-4 left-4 z-40"
        aria-label="Open theme customizer"
      >
        <Settings className="h-4 w-4 mr-2" />
        Customize
      </Button>
    );
  }

  return (
    <>
      {/* Overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40"
          onClick={onToggle}
          aria-hidden="true"
        />
      )}
      
      {/* Sidebar */}
      <div 
        id="theme-customizer-sidebar"
        className={`fixed inset-y-0 right-0 w-96 bg-background/95 backdrop-blur-xl border-l border-primary/20 shadow-2xl z-50 overflow-y-auto transform transition-transform duration-300 ${
          isOpen ? 'translate-x-0' : 'translate-x-full'
        } ${className}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="theme-customizer-title"
      >
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="glass-subtle p-2 rounded-xl">
              <Palette className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h2 id="theme-customizer-title" className="heading-refined text-lg">Theme Customizer</h2>
              <p className="text-xs text-muted-foreground">
                Personalize your Strategic Command Center
              </p>
            </div>
          </div>
          <Button
            onClick={onToggle}
            variant="outline"
            size="sm"
            className="glass-minimal border-primary/30"
            aria-label="Close theme customizer"
          >
            <span className="text-lg font-bold">×</span>
          </Button>
        </div>

        {/* Color Schemes */}
        <div className="space-y-4">
          <h3 className="heading-refined text-sm text-primary">Color Scheme</h3>
          <div className="grid grid-cols-1 gap-3">
            {Object.entries(colorSchemes).map(([key, scheme]) => {
              const IconComponent = scheme.icon;
              const isActive = settings.colorScheme === key;
              const isPreview = activePreview === key;
              
              return (
                <button
                  key={key}
                  onClick={() => {
                    updateSetting('colorScheme', key as keyof typeof colorSchemes);
                    previewScheme(key);
                  }}
                  className={`glass-minimal p-4 rounded-lg text-left transition-all duration-300 ${
                    isActive ? 'border border-primary/50 bg-primary/5' : 'hover:bg-muted/10'
                  } ${isPreview ? 'animate-pulse' : ''}`}
                >
                  <div className="flex items-center gap-3 mb-2">
                    <IconComponent className="h-5 w-5 text-primary" />
                    <span className="font-medium">{scheme.name}</span>
                  </div>
                  <p className="text-xs text-muted-foreground mb-3">
                    {scheme.description}
                  </p>
                  <div className="flex gap-2">
                    {Object.values(scheme.colors).map((color, index) => (
                      <div
                        key={index}
                        className="w-4 h-4 rounded-full border border-white/20"
                        style={{ backgroundColor: color }}
                      />
                    ))}
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Intensity Control */}
        <div className="space-y-4">
          <h3 className="heading-refined text-sm text-primary">Visual Intensity</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label className="text-sm">Overall Intensity</Label>
              <span className="text-xs text-muted-foreground">{settings.intensity}%</span>
            </div>
            <Slider
              value={[settings.intensity]}
              onValueChange={([value]) => updateSetting('intensity', value)}
              min={0}
              max={100}
              step={5}
              className="w-full"
            />
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label className="text-sm">Glass Morphism</Label>
              <span className="text-xs text-muted-foreground">{settings.glassMorphism}%</span>
            </div>
            <Slider
              value={[settings.glassMorphism]}
              onValueChange={([value]) => updateSetting('glassMorphism', value)}
              min={0}
              max={100}
              step={5}
              className="w-full"
            />
          </div>
        </div>

        {/* Feature Toggles */}
        <div className="space-y-4">
          <h3 className="heading-refined text-sm text-primary">Visual Features</h3>
          <div className="space-y-4">
            {[
              { key: 'animations', label: 'Animations', description: 'Smooth transitions and hover effects' },
              { key: 'particleEffects', label: 'Particle Effects', description: 'Victory celebrations and notifications' },
              { key: 'dynamicBackground', label: 'Dynamic Background', description: 'Floating elements and ambient effects' },
              { key: 'soundEffects', label: 'Sound Effects', description: 'Audio feedback for actions' }
            ].map(({ key, label, description }) => (
              <div key={key} className="flex items-center justify-between">
                <div className="flex-1">
                  <Label className="text-sm font-medium">{label}</Label>
                  <p className="text-xs text-muted-foreground">{description}</p>
                </div>
                <Switch
                  checked={settings[key as keyof ThemeSettings] as boolean}
                  onCheckedChange={(checked) => updateSetting(key as keyof ThemeSettings, checked)}
                />
              </div>
            ))}
          </div>
        </div>

        {/* Accessibility */}
        <div className="space-y-4">
          <h3 className="heading-refined text-sm text-primary">Accessibility</h3>
          <div className="space-y-4">
            {[
              { key: 'reducedMotion', label: 'Reduce Motion', description: 'Minimize animations for comfort' },
              { key: 'highContrast', label: 'High Contrast', description: 'Increase color contrast ratios' },
              { key: 'largeText', label: 'Large Text', description: 'Increase font sizes for readability' }
            ].map(({ key, label, description }) => (
              <div key={key} className="flex items-center justify-between">
                <div className="flex-1">
                  <Label className="text-sm font-medium">{label}</Label>
                  <p className="text-xs text-muted-foreground">{description}</p>
                </div>
                <Switch
                  checked={settings.accessibility[key as keyof ThemeSettings['accessibility']]}
                  onCheckedChange={(checked) => updateAccessibility(key as keyof ThemeSettings['accessibility'], checked)}
                />
              </div>
            ))}
          </div>
        </div>

        {/* Preset Actions */}
        <div className="space-y-3">
          <h3 className="heading-refined text-sm text-primary">Presets</h3>
          <div className="grid grid-cols-2 gap-3">
            <Button
              onClick={resetToDefaults}
              variant="outline"
              size="sm"
              className="glass-minimal border-primary/30"
            >
              <Settings className="h-4 w-4 mr-2" />
              Reset
            </Button>
            <Button
              onClick={() => {
                updateSetting('intensity', 100);
                updateSetting('glassMorphism', 80);
                updateSetting('animations', true);
                updateSetting('particleEffects', true);
                updateSetting('dynamicBackground', true);
              }}
              size="sm"
              className="bg-primary hover:bg-primary/90"
            >
              <Sparkles className="h-4 w-4 mr-2" />
              Maximum
            </Button>
          </div>
        </div>

        {/* Current Theme Info */}
        <div className="glass-minimal p-4 rounded-lg border border-primary/20">
          <div className="flex items-center gap-2 mb-2">
            <Monitor className="h-4 w-4 text-primary" />
            <span className="text-sm font-medium">Current Configuration</span>
          </div>
          <div className="text-xs text-muted-foreground space-y-1">
            <div>Scheme: {colorSchemes[settings.colorScheme].name}</div>
            <div>Intensity: {settings.intensity}%</div>
            <div>Glass: {settings.glassMorphism}%</div>
            <div>
              Features: {[
                settings.animations && 'Animations',
                settings.particleEffects && 'Particles',
                settings.dynamicBackground && 'Background'
              ].filter(Boolean).join(', ') || 'None'}
            </div>
          </div>
        </div>
      </div>
      </div>
    </>
  );
};

export const ThemeCustomizer = memo(ThemeCustomizerComponent);