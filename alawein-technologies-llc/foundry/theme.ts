/**
 * Theme Configuration
 * Handles light/dark mode switching and theme persistence
 */

export type ThemeMode = 'light' | 'dark' | 'system';

export interface ThemeColors {
  // Backgrounds
  background: string;
  surface: string;
  surfaceHover: string;

  // Text
  textPrimary: string;
  textSecondary: string;
  textMuted: string;
  textInverse: string;

  // Borders
  border: string;
  borderHover: string;
  borderFocus: string;

  // Primary colors
  primary: string;
  primaryHover: string;
  primaryLight: string;
  primaryDark: string;
  primaryText: string;

  // Secondary colors
  secondary: string;
  secondaryHover: string;
  secondaryLight: string;
  secondaryDark: string;
  secondaryText: string;

  // Semantic colors
  success: string;
  successHover: string;
  successLight: string;
  successDark: string;
  successText: string;

  warning: string;
  warningHover: string;
  warningLight: string;
  warningDark: string;
  warningText: string;

  danger: string;
  dangerHover: string;
  dangerLight: string;
  dangerDark: string;
  dangerText: string;

  // Neutral colors
  gray50: string;
  gray100: string;
  gray200: string;
  gray300: string;
  gray400: string;
  gray500: string;
  gray600: string;
  gray700: string;
  gray800: string;
  gray900: string;
  gray950: string;
}

export const lightTheme: ThemeColors = {
  // Backgrounds
  background: '#ffffff',
  surface: '#f9fafb',
  surfaceHover: '#f3f4f6',

  // Text
  textPrimary: '#111827',
  textSecondary: '#4b5563',
  textMuted: '#9ca3af',
  textInverse: '#ffffff',

  // Borders
  border: '#e5e7eb',
  borderHover: '#d1d5db',
  borderFocus: '#6366f1',

  // Primary colors
  primary: '#6366f1',
  primaryHover: '#4f46e5',
  primaryLight: '#e0e7ff',
  primaryDark: '#4338ca',
  primaryText: '#ffffff',

  // Secondary colors
  secondary: '#a855f7',
  secondaryHover: '#9333ea',
  secondaryLight: '#f3e8ff',
  secondaryDark: '#7e22ce',
  secondaryText: '#ffffff',

  // Success colors
  success: '#22c55e',
  successHover: '#16a34a',
  successLight: '#dcfce7',
  successDark: '#15803d',
  successText: '#ffffff',

  // Warning colors
  warning: '#f59e0b',
  warningHover: '#d97706',
  warningLight: '#fef3c7',
  warningDark: '#b45309',
  warningText: '#ffffff',

  // Danger colors
  danger: '#ef4444',
  dangerHover: '#dc2626',
  dangerLight: '#fee2e2',
  dangerDark: '#b91c1c',
  dangerText: '#ffffff',

  // Neutral colors
  gray50: '#f9fafb',
  gray100: '#f3f4f6',
  gray200: '#e5e7eb',
  gray300: '#d1d5db',
  gray400: '#9ca3af',
  gray500: '#6b7280',
  gray600: '#4b5563',
  gray700: '#374151',
  gray800: '#1f2937',
  gray900: '#111827',
  gray950: '#030712',
};

export const darkTheme: ThemeColors = {
  // Backgrounds
  background: '#030712',
  surface: '#111827',
  surfaceHover: '#1f2937',

  // Text
  textPrimary: '#f9fafb',
  textSecondary: '#d1d5db',
  textMuted: '#6b7280',
  textInverse: '#111827',

  // Borders
  border: '#374151',
  borderHover: '#4b5563',
  borderFocus: '#818cf8',

  // Primary colors
  primary: '#818cf8',
  primaryHover: '#a5b4fc',
  primaryLight: '#312e81',
  primaryDark: '#6366f1',
  primaryText: '#111827',

  // Secondary colors
  secondary: '#c084fc',
  secondaryHover: '#d8b4fe',
  secondaryLight: '#581c87',
  secondaryDark: '#a855f7',
  secondaryText: '#111827',

  // Success colors
  success: '#4ade80',
  successHover: '#86efac',
  successLight: '#14532d',
  successDark: '#22c55e',
  successText: '#111827',

  // Warning colors
  warning: '#fbbf24',
  warningHover: '#fcd34d',
  warningLight: '#78350f',
  warningDark: '#f59e0b',
  warningText: '#111827',

  // Danger colors
  danger: '#f87171',
  dangerHover: '#fca5a5',
  dangerLight: '#7f1d1d',
  dangerDark: '#ef4444',
  dangerText: '#111827',

  // Neutral colors
  gray50: '#030712',
  gray100: '#111827',
  gray200: '#1f2937',
  gray300: '#374151',
  gray400: '#4b5563',
  gray500: '#6b7280',
  gray600: '#9ca3af',
  gray700: '#d1d5db',
  gray800: '#e5e7eb',
  gray900: '#f3f4f6',
  gray950: '#f9fafb',
};

export interface ThemeConfig {
  mode: ThemeMode;
  colors: ThemeColors;
  typography: {
    fontFamily: {
      sans: string;
      mono: string;
    };
    fontSize: {
      xs: string;
      sm: string;
      base: string;
      lg: string;
      xl: string;
      '2xl': string;
      '3xl': string;
      '4xl': string;
      '5xl': string;
      '6xl': string;
      '7xl': string;
      '8xl': string;
      '9xl': string;
    };
    fontWeight: {
      thin: number;
      extralight: number;
      light: number;
      normal: number;
      medium: number;
      semibold: number;
      bold: number;
      extrabold: number;
      black: number;
    };
    lineHeight: {
      none: number;
      tight: number;
      snug: number;
      normal: number;
      relaxed: number;
      loose: number;
    };
  };
  spacing: {
    xs: string;
    sm: string;
    md: string;
    lg: string;
    xl: string;
    '2xl': string;
    '3xl': string;
    '4xl': string;
  };
  borderRadius: {
    none: string;
    sm: string;
    DEFAULT: string;
    md: string;
    lg: string;
    xl: string;
    '2xl': string;
    '3xl': string;
    full: string;
  };
  shadows: {
    xs: string;
    sm: string;
    md: string;
    lg: string;
    xl: string;
    '2xl': string;
    '3xl': string;
    inner: string;
  };
  transitions: {
    duration: {
      fast: number;
      normal: number;
      slow: number;
    };
    easing: {
      linear: string;
      in: string;
      out: string;
      inOut: string;
      spring: string;
    };
  };
  breakpoints: {
    xs: string;
    sm: string;
    md: string;
    lg: string;
    xl: string;
    '2xl': string;
  };
  zIndex: {
    dropdown: number;
    sticky: number;
    fixed: number;
    backdrop: number;
    modal: number;
    popover: number;
    tooltip: number;
    toast: number;
  };
}

export const theme: ThemeConfig = {
  mode: 'system',
  colors: lightTheme,
  typography: {
    fontFamily: {
      sans: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      mono: '"Fira Code", Consolas, Monaco, "Andale Mono", monospace',
    },
    fontSize: {
      xs: '0.75rem',
      sm: '0.875rem',
      base: '1rem',
      lg: '1.125rem',
      xl: '1.25rem',
      '2xl': '1.5rem',
      '3xl': '1.875rem',
      '4xl': '2.25rem',
      '5xl': '3rem',
      '6xl': '3.75rem',
      '7xl': '4.5rem',
      '8xl': '6rem',
      '9xl': '8rem',
    },
    fontWeight: {
      thin: 100,
      extralight: 200,
      light: 300,
      normal: 400,
      medium: 500,
      semibold: 600,
      bold: 700,
      extrabold: 800,
      black: 900,
    },
    lineHeight: {
      none: 1,
      tight: 1.25,
      snug: 1.375,
      normal: 1.5,
      relaxed: 1.625,
      loose: 2,
    },
  },
  spacing: {
    xs: '0.25rem',
    sm: '0.5rem',
    md: '1rem',
    lg: '1.5rem',
    xl: '2rem',
    '2xl': '3rem',
    '3xl': '4rem',
    '4xl': '5rem',
  },
  borderRadius: {
    none: '0px',
    sm: '0.125rem',
    DEFAULT: '0.25rem',
    md: '0.375rem',
    lg: '0.5rem',
    xl: '0.75rem',
    '2xl': '1rem',
    '3xl': '1.5rem',
    full: '9999px',
  },
  shadows: {
    xs: '0 1px 2px 0 rgb(0 0 0 / 0.05)',
    sm: '0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)',
    md: '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)',
    lg: '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)',
    xl: '0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1)',
    '2xl': '0 25px 50px -12px rgb(0 0 0 / 0.25)',
    '3xl': '0 35px 60px -15px rgb(0 0 0 / 0.3)',
    inner: 'inset 0 2px 4px 0 rgb(0 0 0 / 0.05)',
  },
  transitions: {
    duration: {
      fast: 150,
      normal: 300,
      slow: 500,
    },
    easing: {
      linear: 'linear',
      in: 'cubic-bezier(0.4, 0, 1, 1)',
      out: 'cubic-bezier(0, 0, 0.2, 1)',
      inOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
      spring: 'cubic-bezier(0.175, 0.885, 0.32, 1.275)',
    },
  },
  breakpoints: {
    xs: '320px',
    sm: '640px',
    md: '768px',
    lg: '1024px',
    xl: '1280px',
    '2xl': '1536px',
  },
  zIndex: {
    dropdown: 1000,
    sticky: 1020,
    fixed: 1030,
    backdrop: 1040,
    modal: 1050,
    popover: 1060,
    tooltip: 1070,
    toast: 1080,
  },
};

/**
 * Theme Provider Hook
 * Usage: const { theme, setTheme, toggleTheme } = useTheme();
 */
export class ThemeManager {
  private static instance: ThemeManager;
  private currentMode: ThemeMode = 'system';
  private listeners: Set<(theme: ThemeMode) => void> = new Set();

  private constructor() {
    this.init();
  }

  static getInstance(): ThemeManager {
    if (!ThemeManager.instance) {
      ThemeManager.instance = new ThemeManager();
    }
    return ThemeManager.instance;
  }

  private init() {
    // Check localStorage for saved preference
    const savedTheme = localStorage.getItem('theme') as ThemeMode | null;
    if (savedTheme) {
      this.currentMode = savedTheme;
    } else {
      // Default to system preference
      this.currentMode = 'system';
    }
    this.applyTheme();

    // Listen for system theme changes
    if (window.matchMedia) {
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
        if (this.currentMode === 'system') {
          this.applyTheme();
        }
      });
    }
  }

  private applyTheme() {
    const root = document.documentElement;
    const isDark = this.isDarkMode();

    if (isDark) {
      root.classList.add('dark');
      this.applyColors(darkTheme);
    } else {
      root.classList.remove('dark');
      this.applyColors(lightTheme);
    }

    // Notify listeners
    this.listeners.forEach(listener => listener(this.currentMode));
  }

  private applyColors(colors: ThemeColors) {
    const root = document.documentElement;
    Object.entries(colors).forEach(([key, value]) => {
      // Convert camelCase to kebab-case
      const cssVar = `--${key.replace(/([A-Z])/g, '-$1').toLowerCase()}`;
      root.style.setProperty(cssVar, value);
    });
  }

  private isDarkMode(): boolean {
    if (this.currentMode === 'dark') {
      return true;
    }
    if (this.currentMode === 'light') {
      return false;
    }
    // System mode - check system preference
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  }

  setTheme(mode: ThemeMode) {
    this.currentMode = mode;
    localStorage.setItem('theme', mode);
    this.applyTheme();
  }

  toggleTheme() {
    const newMode = this.isDarkMode() ? 'light' : 'dark';
    this.setTheme(newMode);
  }

  getTheme(): ThemeMode {
    return this.currentMode;
  }

  getColors(): ThemeColors {
    return this.isDarkMode() ? darkTheme : lightTheme;
  }

  subscribe(listener: (theme: ThemeMode) => void) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
}

// Export singleton instance
export const themeManager = ThemeManager.getInstance();

// React Hook
export function useTheme() {
  const [mode, setMode] = React.useState<ThemeMode>(themeManager.getTheme());
  const [colors, setColors] = React.useState<ThemeColors>(themeManager.getColors());

  React.useEffect(() => {
    const unsubscribe = themeManager.subscribe((newMode) => {
      setMode(newMode);
      setColors(themeManager.getColors());
    });
    return unsubscribe;
  }, []);

  return {
    mode,
    colors,
    setTheme: (mode: ThemeMode) => themeManager.setTheme(mode),
    toggleTheme: () => themeManager.toggleTheme(),
  };
}