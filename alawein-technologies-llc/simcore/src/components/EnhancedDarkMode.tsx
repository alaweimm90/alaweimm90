import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { Moon, Sun, Monitor, Palette } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

type Theme = 'dark' | 'light' | 'system';
type ColorScheme = 'default' | 'physics' | 'research' | 'minimal';

interface ThemeContextType {
  theme: Theme;
  colorScheme: ColorScheme;
  setTheme: (theme: Theme) => void;
  setColorScheme: (scheme: ColorScheme) => void;
  systemTheme: 'dark' | 'light';
  effectiveTheme: 'dark' | 'light';
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}

interface ThemeProviderProps {
  children: React.ReactNode;
  defaultTheme?: Theme;
  defaultColorScheme?: ColorScheme;
}

export function ThemeProvider({ 
  children, 
  defaultTheme = 'system',
  defaultColorScheme = 'physics'
}: ThemeProviderProps) {
  const [theme, setThemeState] = useState<Theme>(defaultTheme);
  const [colorScheme, setColorSchemeState] = useState<ColorScheme>(defaultColorScheme);
  const [systemTheme, setSystemTheme] = useState<'dark' | 'light'>('light');

  // Listen for system theme changes
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    
    const handleChange = (e: MediaQueryListEvent) => {
      setSystemTheme(e.matches ? 'dark' : 'light');
    };
    
    setSystemTheme(mediaQuery.matches ? 'dark' : 'light');
    mediaQuery.addEventListener('change', handleChange);
    
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  // Load saved preferences
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') as Theme;
    const savedScheme = localStorage.getItem('colorScheme') as ColorScheme;
    
    if (savedTheme) setThemeState(savedTheme);
    if (savedScheme) setColorSchemeState(savedScheme);
  }, []);

  const effectiveTheme = theme === 'system' ? systemTheme : theme;

  // Apply theme to document
  useEffect(() => {
    const root = window.document.documentElement;
    
    root.classList.remove('light', 'dark');
    root.classList.add(effectiveTheme);
    
    // Apply color scheme
    root.setAttribute('data-color-scheme', colorScheme);
    
    // Update meta theme-color for mobile browsers
    const metaThemeColor = document.querySelector('meta[name="theme-color"]');
    if (metaThemeColor) {
      const color = effectiveTheme === 'dark' ? '#0a0a0a' : '#ffffff';
      metaThemeColor.setAttribute('content', color);
    }
  }, [effectiveTheme, colorScheme]);

  const setTheme = useCallback((newTheme: Theme) => {
    setThemeState(newTheme);
    localStorage.setItem('theme', newTheme);
  }, []);

  const setColorScheme = useCallback((newScheme: ColorScheme) => {
    setColorSchemeState(newScheme);
    localStorage.setItem('colorScheme', newScheme);
  }, []);

  const value = {
    theme,
    colorScheme,
    setTheme,
    setColorScheme,
    systemTheme,
    effectiveTheme
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}

// Theme switcher component
export function ThemeToggle() {
  const { theme, setTheme, colorScheme, setColorScheme, effectiveTheme } = useTheme();

  const themeIcons = {
    light: Sun,
    dark: Moon,
    system: Monitor
  };

  const colorSchemes = {
    default: { name: 'Default', color: 'bg-blue-500' },
    physics: { name: 'Physics', color: 'bg-purple-500' },
    research: { name: 'Research', color: 'bg-green-500' },
    minimal: { name: 'Minimal', color: 'bg-gray-500' }
  };

  const CurrentIcon = themeIcons[theme];

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button 
          variant="ghost" 
          size="sm"
          className="relative h-9 w-9 p-0"
        >
          <CurrentIcon className="h-4 w-4" />
          <span className="sr-only">Toggle theme</span>
        </Button>
      </DropdownMenuTrigger>
      
      <DropdownMenuContent align="end" className="w-56">
        <div className="px-2 py-1.5">
          <div className="text-sm font-medium">Theme</div>
        </div>
        
        {Object.entries(themeIcons).map(([key, Icon]) => (
          <DropdownMenuItem
            key={key}
            onClick={() => setTheme(key as Theme)}
            className={cn(
              "flex items-center gap-2",
              theme === key && "bg-accent"
            )}
          >
            <Icon className="h-4 w-4" />
            <span className="capitalize">{key}</span>
            {theme === key && <Badge variant="secondary" className="ml-auto">Active</Badge>}
          </DropdownMenuItem>
        ))}
        
        <DropdownMenuSeparator />
        
        <div className="px-2 py-1.5">
          <div className="text-sm font-medium">Color Scheme</div>
        </div>
        
        {Object.entries(colorSchemes).map(([key, { name, color }]) => (
          <DropdownMenuItem
            key={key}
            onClick={() => setColorScheme(key as ColorScheme)}
            className={cn(
              "flex items-center gap-2",
              colorScheme === key && "bg-accent"
            )}
          >
            <div className={cn("h-3 w-3 rounded-full", color)} />
            <span>{name}</span>
            {colorScheme === key && <Badge variant="secondary" className="ml-auto">Active</Badge>}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

// Theme-aware container
interface ThemedContainerProps {
  children: React.ReactNode;
  className?: string;
  variant?: 'default' | 'elevated' | 'outlined';
}

export function ThemedContainer({ 
  children, 
  className,
  variant = 'default' 
}: ThemedContainerProps) {
  const { effectiveTheme, colorScheme } = useTheme();
  
  const variantClasses = {
    default: 'bg-background text-foreground',
    elevated: 'bg-card text-card-foreground shadow-md',
    outlined: 'bg-background text-foreground border border-border'
  };
  
  return (
    <div 
      className={cn(
        variantClasses[variant],
        'transition-colors duration-200',
        className
      )}
      data-theme={effectiveTheme}
      data-color-scheme={colorScheme}
    >
      {children}
    </div>
  );
}