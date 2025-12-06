import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Palette, Download, Eye, RotateCcw } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface ThemeColors {
  primary: string;
  secondary: string;
  accent: string;
  background: string;
  surface: string;
  text: string;
}

export const CustomThemeBuilder: React.FC = () => {
  const { toast } = useToast();
  const [colors, setColors] = useState<ThemeColors>({
    primary: '#3b82f6',
    secondary: '#6366f1', 
    accent: '#8b5cf6',
    background: '#0a0a0a',
    surface: '#141414',
    text: '#ffffff'
  });

  const [previewMode, setPreviewMode] = useState(false);

  const updateColor = (key: keyof ThemeColors, value: string) => {
    setColors(prev => ({ ...prev, [key]: value }));
  };

  const applyTheme = () => {
    if (previewMode) {
      // Apply theme to CSS variables
      const root = document.documentElement;
      Object.entries(colors).forEach(([key, value]) => {
        root.style.setProperty(`--theme-${key}`, value);
      });
      
      toast({
        title: "Theme Applied",
        description: "Custom theme is now active"
      });
    }
  };

  const exportTheme = () => {
    const themeData = {
      name: "Custom SimCore Theme",
      colors,
      timestamp: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(themeData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'simcore-custom-theme.json';
    a.click();
    URL.revokeObjectURL(url);

    toast({
      title: "Theme Exported",
      description: "Theme configuration saved to file"
    });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Palette className="w-5 h-5" />
          Custom Theme Builder
        </CardTitle>
        <CardDescription>
          Create custom color themes for SimCore physics simulations
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <h4 className="font-medium">Theme Colors</h4>
            {Object.entries(colors).map(([key, value]) => (
              <div key={key} className="flex items-center gap-3">
                <Label className="w-20 capitalize">{key}:</Label>
                <Input
                  type="color"
                  value={value}
                  onChange={(e) => updateColor(key as keyof ThemeColors, e.target.value)}
                  className="w-16 h-8"
                />
                <Input
                  value={value}
                  onChange={(e) => updateColor(key as keyof ThemeColors, e.target.value)}
                  className="font-mono text-sm"
                />
              </div>
            ))}
          </div>

          <div className="space-y-4">
            <h4 className="font-medium">Preview</h4>
            <div 
              className="p-4 rounded-lg border"
              style={{
                backgroundColor: colors.surface,
                color: colors.text,
                borderColor: colors.accent
              }}
            >
              <h5 style={{ color: colors.primary }}>Physics Simulation</h5>
              <p style={{ color: colors.text }}>Sample simulation content with custom theme</p>
              <button 
                className="px-3 py-1 rounded mt-2"
                style={{ 
                  backgroundColor: colors.accent,
                  color: colors.background 
                }}
              >
                Start Simulation
              </button>
            </div>
          </div>
        </div>

        <div className="flex gap-4 pt-4 border-t">
          <Button onClick={() => setPreviewMode(!previewMode)} variant="outline" className="gap-2">
            <Eye className="w-4 h-4" />
            {previewMode ? 'Exit Preview' : 'Preview Theme'}
          </Button>
          <Button onClick={applyTheme} disabled={!previewMode}>
            Apply Theme
          </Button>
          <Button onClick={exportTheme} variant="outline" className="gap-2">
            <Download className="w-4 h-4" />
            Export
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default CustomThemeBuilder;