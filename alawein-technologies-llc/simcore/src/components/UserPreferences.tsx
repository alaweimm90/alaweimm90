import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { 
  Settings,
  Monitor,
  Moon,
  Sun,
  Volume2,
  VolumeX,
  Eye,
  Palette,
  Accessibility,
  Zap,
  Save,
  RotateCcw,
  Bell,
  Globe
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { useTheme } from 'next-themes';

interface UserPreferences {
  theme: 'light' | 'dark' | 'system';
  reducedMotion: boolean;
  fontSize: number;
  soundEnabled: boolean;
  notificationsEnabled: boolean;
  autoSave: boolean;
  language: string;
  contrastMode: 'normal' | 'high';
  animationSpeed: number;
  tutorialMode: boolean;
  expertMode: boolean;
  colorScheme: 'default' | 'deuteranopia' | 'protanopia' | 'tritanopia';
}

interface UserPreferencesProps {
  className?: string;
  onClose?: () => void;
}

const defaultPreferences: UserPreferences = {
  theme: 'system',
  reducedMotion: false,
  fontSize: 16,
  soundEnabled: true,
  notificationsEnabled: true,
  autoSave: true,
  language: 'en',
  contrastMode: 'normal',
  animationSpeed: 1,
  tutorialMode: true,
  expertMode: false,
  colorScheme: 'default'
};

export const UserPreferences: React.FC<UserPreferencesProps> = ({
  className = '',
  onClose
}) => {
  const [preferences, setPreferences] = useState<UserPreferences>(defaultPreferences);
  const [hasChanges, setHasChanges] = useState(false);
  const { toast } = useToast();
  const { theme, setTheme } = useTheme();

  // Load preferences from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('simcore-preferences');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setPreferences({ ...defaultPreferences, ...parsed });
      } catch (error) {
        console.warn('Failed to parse saved preferences:', error);
      }
    }
  }, []);

  // Apply preferences to document
  useEffect(() => {
    const root = document.documentElement;
    
    // Font size
    root.style.fontSize = `${preferences.fontSize}px`;
    
    // High contrast mode
    if (preferences.contrastMode === 'high') {
      root.classList.add('high-contrast');
    } else {
      root.classList.remove('high-contrast');
    }
    
    // Reduced motion
    if (preferences.reducedMotion) {
      root.classList.add('reduce-motion');
    } else {
      root.classList.remove('reduce-motion');
    }
    
    // Animation speed
    root.style.setProperty('--animation-speed', preferences.animationSpeed.toString());
    
    // Color scheme for accessibility
    root.setAttribute('data-color-scheme', preferences.colorScheme);
    
  }, [preferences]);

  const updatePreference = <K extends keyof UserPreferences>(
    key: K,
    value: UserPreferences[K]
  ) => {
    setPreferences(prev => ({ ...prev, [key]: value }));
    setHasChanges(true);
    
    // Special handling for theme
    if (key === 'theme') {
      setTheme(value as string);
    }
  };

  const savePreferences = () => {
    try {
      localStorage.setItem('simcore-preferences', JSON.stringify(preferences));
      setHasChanges(false);
      toast({
        title: "Preferences Saved",
        description: "Your settings have been saved successfully.",
      });
    } catch (error) {
      toast({
        title: "Save Failed",
        description: "Could not save preferences. Please try again.",
        variant: "destructive"
      });
    }
  };

  const resetPreferences = () => {
    setPreferences(defaultPreferences);
    setHasChanges(true);
    toast({
      title: "Preferences Reset",
      description: "All settings have been reset to defaults.",
    });
  };

  const accessibilityPresets = [
    {
      name: 'High Contrast',
      description: 'Improved visibility for low vision users',
      apply: () => {
        updatePreference('contrastMode', 'high');
        updatePreference('fontSize', 18);
      }
    },
    {
      name: 'Motion Sensitive',
      description: 'Reduced animations and motion',
      apply: () => {
        updatePreference('reducedMotion', true);
        updatePreference('animationSpeed', 0.5);
      }
    },
    {
      name: 'Deuteranopia',
      description: 'Color scheme for red-green color blindness',
      apply: () => {
        updatePreference('colorScheme', 'deuteranopia');
      }
    }
  ];

  return (
    <Card className={`w-full max-w-4xl mx-auto border-border/50 ${className}`}>
      <CardHeader>
        <CardTitle className="flex items-center gap-[--spacing-sm]">
          <Settings className="w-6 h-6 text-primary" />
          User Preferences
        </CardTitle>
      </CardHeader>

      <CardContent>
        <Tabs defaultValue="appearance" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="appearance" className="min-h-[--touch-target-min]">Appearance</TabsTrigger>
            <TabsTrigger value="accessibility" className="min-h-[--touch-target-min]">Accessibility</TabsTrigger>
            <TabsTrigger value="interaction" className="min-h-[--touch-target-min]">Interaction</TabsTrigger>
            <TabsTrigger value="advanced" className="min-h-[--touch-target-min]">Advanced</TabsTrigger>
          </TabsList>

          {/* Appearance Tab */}
          <TabsContent value="appearance" className="space-y-6 mt-6">
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="theme">Theme</Label>
                <Select 
                  value={preferences.theme} 
                  onValueChange={(value: any) => updatePreference('theme', value)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="light">
                      <div className="flex items-center gap-2">
                        <Sun className="w-4 h-4" />
                        Light
                      </div>
                    </SelectItem>
                    <SelectItem value="dark">
                      <div className="flex items-center gap-2">
                        <Moon className="w-4 h-4" />
                        Dark
                      </div>
                    </SelectItem>
                    <SelectItem value="system">
                      <div className="flex items-center gap-2">
                        <Monitor className="w-4 h-4" />
                        System
                      </div>
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="font-size">Font Size: {preferences.fontSize}px</Label>
                <Slider
                  id="font-size"
                  min={12}
                  max={24}
                  step={1}
                  value={[preferences.fontSize]}
                  onValueChange={([value]) => updatePreference('fontSize', value)}
                  className="w-full"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="contrast">Contrast Mode</Label>
                <Select 
                  value={preferences.contrastMode} 
                  onValueChange={(value: any) => updatePreference('contrastMode', value)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="normal">Normal</SelectItem>
                    <SelectItem value="high">High Contrast</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="color-scheme">Color Scheme</Label>
                <Select 
                  value={preferences.colorScheme} 
                  onValueChange={(value: any) => updatePreference('colorScheme', value)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="default">Default</SelectItem>
                    <SelectItem value="deuteranopia">Deuteranopia Friendly</SelectItem>
                    <SelectItem value="protanopia">Protanopia Friendly</SelectItem>
                    <SelectItem value="tritanopia">Tritanopia Friendly</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </TabsContent>

          {/* Accessibility Tab */}
          <TabsContent value="accessibility" className="space-y-6 mt-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label htmlFor="reduced-motion">Reduce Motion</Label>
                  <p className="text-sm text-muted-foreground">
                    Minimize animations and transitions
                  </p>
                </div>
                <Switch
                  id="reduced-motion"
                  checked={preferences.reducedMotion}
                  onCheckedChange={(checked) => updatePreference('reducedMotion', checked)}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="animation-speed">Animation Speed: {preferences.animationSpeed}x</Label>
                <Slider
                  id="animation-speed"
                  min={0.1}
                  max={2}
                  step={0.1}
                  value={[preferences.animationSpeed]}
                  onValueChange={([value]) => updatePreference('animationSpeed', value)}
                  className="w-full"
                />
              </div>

              <Separator />

              <div>
                <h4 className="font-semibold mb-3">Accessibility Presets</h4>
                <div className="space-y-2">
                  {accessibilityPresets.map((preset) => (
                    <Card key={preset.name} className="p-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <h5 className="font-medium">{preset.name}</h5>
                          <p className="text-sm text-muted-foreground">{preset.description}</p>
                        </div>
                        <Button size="sm" onClick={preset.apply}>
                          Apply
                        </Button>
                      </div>
                    </Card>
                  ))}
                </div>
              </div>
            </div>
          </TabsContent>

          {/* Interaction Tab */}
          <TabsContent value="interaction" className="space-y-6 mt-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label htmlFor="sound">Sound Effects</Label>
                  <p className="text-sm text-muted-foreground">
                    Play audio feedback for interactions
                  </p>
                </div>
                <Switch
                  id="sound"
                  checked={preferences.soundEnabled}
                  onCheckedChange={(checked) => updatePreference('soundEnabled', checked)}
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label htmlFor="notifications">Notifications</Label>
                  <p className="text-sm text-muted-foreground">
                    Show system notifications and updates
                  </p>
                </div>
                <Switch
                  id="notifications"
                  checked={preferences.notificationsEnabled}
                  onCheckedChange={(checked) => updatePreference('notificationsEnabled', checked)}
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label htmlFor="auto-save">Auto Save</Label>
                  <p className="text-sm text-muted-foreground">
                    Automatically save simulation states
                  </p>
                </div>
                <Switch
                  id="auto-save"
                  checked={preferences.autoSave}
                  onCheckedChange={(checked) => updatePreference('autoSave', checked)}
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label htmlFor="tutorial-mode">Tutorial Mode</Label>
                  <p className="text-sm text-muted-foreground">
                    Show helpful tips and guidance
                  </p>
                </div>
                <Switch
                  id="tutorial-mode"
                  checked={preferences.tutorialMode}
                  onCheckedChange={(checked) => updatePreference('tutorialMode', checked)}
                />
              </div>
            </div>
          </TabsContent>

          {/* Advanced Tab */}
          <TabsContent value="advanced" className="space-y-6 mt-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label htmlFor="expert-mode">Expert Mode</Label>
                  <p className="text-sm text-muted-foreground">
                    Show advanced controls and options
                  </p>
                </div>
                <Switch
                  id="expert-mode"
                  checked={preferences.expertMode}
                  onCheckedChange={(checked) => updatePreference('expertMode', checked)}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="language">Language</Label>
                <Select 
                  value={preferences.language} 
                  onValueChange={(value: any) => updatePreference('language', value)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="en">English</SelectItem>
                    <SelectItem value="es">Español</SelectItem>
                    <SelectItem value="fr">Français</SelectItem>
                    <SelectItem value="de">Deutsch</SelectItem>
                    <SelectItem value="zh">中文</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <Separator />

              <Card className="p-4 bg-muted/30">
                <h4 className="font-semibold mb-2">Data & Storage</h4>
                <div className="space-y-2 text-sm text-muted-foreground">
                  <p>• Preferences are stored locally in your browser</p>
                  <p>• Simulation data is cached for performance</p>
                  <p>• No personal data is sent to external servers</p>
                </div>
              </Card>
            </div>
          </TabsContent>
        </Tabs>

        {/* Action Buttons */}
        <div className="flex items-center justify-between pt-6 border-t">
          <Button
            variant="outline"
            onClick={resetPreferences}
            className="flex items-center gap-2"
          >
            <RotateCcw className="w-4 h-4" />
            Reset to Defaults
          </Button>

          <div className="flex items-center gap-2">
            {onClose && (
              <Button variant="outline" onClick={onClose}>
                Cancel
              </Button>
            )}
            <Button
              onClick={savePreferences}
              disabled={!hasChanges}
              className="flex items-center gap-2"
            >
              <Save className="w-4 h-4" />
              Save Preferences
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};