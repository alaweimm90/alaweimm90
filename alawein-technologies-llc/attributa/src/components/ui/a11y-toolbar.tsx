import React, { useState } from 'react';
import { Button } from './button';
import { Card } from './card';
import { Settings, Eye, MousePointer, Type, Volume2 } from 'lucide-react';

interface A11yToolbarProps {
  className?: string;
}

export function A11yToolbar({ className }: A11yToolbarProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [settings, setSettings] = useState({
    highContrast: false,
    largeText: false,
    reducedMotion: false,
    focusIndicators: false
  });

  const toggleSetting = (key: keyof typeof settings) => {
    const newSettings = { ...settings, [key]: !settings[key] };
    setSettings(newSettings);
    
    // Apply settings to document
    document.documentElement.classList.toggle(`a11y-${key.replace(/([A-Z])/g, '-$1').toLowerCase()}`, newSettings[key]);
    
    // Save to localStorage
    localStorage.setItem('a11y-preferences', JSON.stringify(newSettings));
  };

  React.useEffect(() => {
    // Load saved preferences
    const saved = localStorage.getItem('a11y-preferences');
    if (saved) {
      const preferences = JSON.parse(saved);
      setSettings(preferences);
      
      // Apply saved settings
      Object.entries(preferences).forEach(([key, value]) => {
        if (value) {
          document.documentElement.classList.add(`a11y-${key.replace(/([A-Z])/g, '-$1').toLowerCase()}`);
        }
      });
    }
  }, []);

  if (!isOpen) {
    return (
      <Button
        variant="outline"
        size="sm"
        onClick={() => setIsOpen(true)}
        className={`fixed top-4 right-4 z-50 ${className}`}
        aria-label="Open accessibility tools"
      >
        <Settings className="h-4 w-4" />
      </Button>
    );
  }

  return (
    <Card className={`fixed top-4 right-4 z-50 p-4 w-80 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold">Accessibility Tools</h3>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setIsOpen(false)}
          aria-label="Close accessibility tools"
        >
          ×
        </Button>
      </div>
      
      <div className="space-y-3">
        <Button
          variant={settings.highContrast ? "default" : "outline"}
          size="sm"
          className="w-full justify-start"
          onClick={() => toggleSetting('highContrast')}
        >
          <Eye className="h-4 w-4 mr-2" />
          High Contrast
        </Button>
        
        <Button
          variant={settings.largeText ? "default" : "outline"}
          size="sm"
          className="w-full justify-start"
          onClick={() => toggleSetting('largeText')}
        >
          <Type className="h-4 w-4 mr-2" />
          Large Text
        </Button>
        
        <Button
          variant={settings.reducedMotion ? "default" : "outline"}
          size="sm"
          className="w-full justify-start"
          onClick={() => toggleSetting('reducedMotion')}
        >
          <Volume2 className="h-4 w-4 mr-2" />
          Reduce Motion
        </Button>
        
        <Button
          variant={settings.focusIndicators ? "default" : "outline"}
          size="sm"
          className="w-full justify-start"
          onClick={() => toggleSetting('focusIndicators')}
        >
          <MousePointer className="h-4 w-4 mr-2" />
          Enhanced Focus
        </Button>
      </div>
    </Card>
  );
}