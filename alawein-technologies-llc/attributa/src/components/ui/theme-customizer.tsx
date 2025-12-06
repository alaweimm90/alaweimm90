import React, { useState } from 'react';
import { Monitor, Moon, Sun, Palette } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';

interface ThemeCustomizerProps {
  onThemeChange?: (theme: { theme: string }) => void;
  className?: string;
}

export function ThemeCustomizer({ onThemeChange, className }: ThemeCustomizerProps) {
  const [theme, setTheme] = useState('dark');
  const [reducedMotion, setReducedMotion] = useState(false);
  const [highContrast, setHighContrast] = useState(false);
  const [fontSize, setFontSize] = useState([14]);
  const [spacing, setSpacing] = useState([1]);

  const handleThemeChange = (newTheme: string) => {
    setTheme(newTheme);
    document.documentElement.classList.remove('light', 'dark', 'system');
    document.documentElement.classList.add(newTheme);
    onThemeChange?.({ theme: newTheme });
  };

  const handleReducedMotion = (enabled: boolean) => {
    setReducedMotion(enabled);
    document.documentElement.classList.toggle('reduce-motion', enabled);
  };

  const handleHighContrast = (enabled: boolean) => {
    setHighContrast(enabled);
    document.documentElement.classList.toggle('high-contrast', enabled);
  };

  const handleFontSizeChange = (value: number[]) => {
    setFontSize(value);
    document.documentElement.style.fontSize = `${value[0]}px`;
  };

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 font-mono">
          <Palette className="h-5 w-5 text-primary" />
          Developer Preferences
        </CardTitle>
        <CardDescription>
          Customize your development environment for optimal productivity
        </CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {/* Theme Selection */}
        <div className="space-y-3">
          <label className="text-sm font-medium font-mono">Color Theme</label>
          <div className="grid grid-cols-3 gap-2">
            <Button
              variant={theme === 'light' ? 'default' : 'outline'}
              size="sm"
              onClick={() => handleThemeChange('light')}
              className="flex items-center gap-2"
            >
              <Sun className="h-4 w-4" />
              Light
            </Button>
            <Button
              variant={theme === 'dark' ? 'default' : 'outline'}
              size="sm"
              onClick={() => handleThemeChange('dark')}
              className="flex items-center gap-2"
            >
              <Moon className="h-4 w-4" />
              Dark
            </Button>
            <Button
              variant={theme === 'system' ? 'default' : 'outline'}
              size="sm"
              onClick={() => handleThemeChange('system')}
              className="flex items-center gap-2"
            >
              <Monitor className="h-4 w-4" />
              System
            </Button>
          </div>
        </div>

        <Separator />

        {/* Accessibility Options */}
        <div className="space-y-4">
          <h4 className="text-sm font-medium font-mono">Accessibility</h4>
          
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <label className="text-sm font-medium">Reduced Motion</label>
              <p className="text-xs text-muted-foreground">
                Minimize animations and transitions
              </p>
            </div>
            <Switch
              checked={reducedMotion}
              onCheckedChange={handleReducedMotion}
            />
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <label className="text-sm font-medium">High Contrast</label>
              <p className="text-xs text-muted-foreground">
                Increase color contrast for better visibility
              </p>
            </div>
            <Switch
              checked={highContrast}
              onCheckedChange={handleHighContrast}
            />
          </div>
        </div>

        <Separator />

        {/* Typography Settings */}
        <div className="space-y-4">
          <h4 className="text-sm font-medium font-mono">Typography</h4>
          
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">Font Size</label>
              <Badge variant="outline" className="font-mono text-xs">
                {fontSize[0]}px
              </Badge>
            </div>
            <Slider
              value={fontSize}
              onValueChange={handleFontSizeChange}
              min={12}
              max={18}
              step={1}
              className="w-full"
            />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">Spacing</label>
              <Badge variant="outline" className="font-mono text-xs">
                {spacing[0]}x
              </Badge>
            </div>
            <Slider
              value={spacing}
              onValueChange={setSpacing}
              min={0.5}
              max={2}
              step={0.25}
              className="w-full"
            />
          </div>
        </div>

        <Separator />

        {/* Code Preferences */}
        <div className="space-y-4">
          <h4 className="text-sm font-medium font-mono">Code Display</h4>
          
          <div className="space-y-2">
            <label className="text-sm font-medium">Syntax Theme</label>
            <Select defaultValue="github-dark">
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="github-dark">GitHub Dark</SelectItem>
                <SelectItem value="dracula">Dracula</SelectItem>
                <SelectItem value="monokai">Monokai</SelectItem>
                <SelectItem value="nord">Nord</SelectItem>
                <SelectItem value="one-dark">One Dark</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <label className="text-sm font-medium">Line Numbers</label>
              <p className="text-xs text-muted-foreground">
                Show line numbers in code blocks
              </p>
            </div>
            <Switch defaultChecked />
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <label className="text-sm font-medium">Word Wrap</label>
              <p className="text-xs text-muted-foreground">
                Wrap long lines in code blocks
              </p>
            </div>
            <Switch />
          </div>
        </div>

        <div className="pt-4 border-t">
          <Button
            variant="outline"
            size="sm"
            className="w-full font-mono"
            onClick={() => {
              // Reset to defaults
              handleThemeChange('dark');
              setReducedMotion(false);
              setHighContrast(false);
              setFontSize([14]);
              setSpacing([1]);
            }}
          >
            Reset to Defaults
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}