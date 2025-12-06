/**
 * QMLab Theme Configuration
 * Central theme configuration for easy customization
 * Edit this file to change the entire app's theme
 */

export interface ThemeConfig {
  name: string;
  colors: {
    primary: string;
    primaryHover: string;
    primaryActive: string;
    secondary: string;
    secondaryHover: string;
    secondaryActive: string;
    success: string;
    warning: string;
    danger: string;
    info: string;
  };
  typography: {
    fontFamily: {
      sans: string;
      mono: string;
    };
    fontSize: {
      base: string;
      scale: number;
    };
  };
  spacing: {
    base: number; // in pixels
    scale: number[];
  };
  borderRadius: {
    base: string;
    scale: Record<'sm' | 'md' | 'lg' | 'xl' | 'full', string>;
  };
  animation: {
    duration: {
      fast: string;
      normal: string;
      slow: string;
    };
    easing: string;
  };
  effects: {
    blur: {
      subtle: string;
      medium: string;
      strong: string;
    };
    glow: {
      subtle: string;
      medium: string;
      strong: string;
    };
  };
}

export const defaultTheme: ThemeConfig = {
  name: 'QMLab Quantum',
  colors: {
    primary: 'var(--quantum-blue-500)',
    primaryHover: 'var(--quantum-blue-600)',
    primaryActive: 'var(--quantum-blue-700)',
    secondary: 'var(--quantum-purple-500)',
    secondaryHover: 'var(--quantum-purple-600)',
    secondaryActive: 'var(--quantum-purple-700)',
    success: '#059669',
    warning: '#d97706',
    danger: '#dc2626',
    info: 'var(--quantum-blue-600)',
  },
  typography: {
    fontFamily: {
      sans: 'Inter, system-ui, -apple-system, sans-serif',
      mono: '"Fira Code", "JetBrains Mono", monospace',
    },
    fontSize: {
      base: '16px',
      scale: 1.25, // Major third scale
    },
  },
  spacing: {
    base: 4, // 4px base unit
    scale: [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24],
  },
  borderRadius: {
    base: '0.5rem',
    scale: {
      sm: '0.375rem',
      md: '0.5rem',
      lg: '0.75rem',
      xl: '1rem',
      full: '9999px',
    },
  },
  animation: {
    duration: {
      fast: '150ms',
      normal: '300ms',
      slow: '500ms',
    },
    easing: 'cubic-bezier(0.25, 0.46, 0.45, 0.94)',
  },
  effects: {
    blur: {
      subtle: '8px',
      medium: '12px',
      strong: '20px',
    },
    glow: {
      subtle: '0 0 20px hsl(217 91% 60% / 0.2)',
      medium: '0 0 40px hsl(217 91% 60% / 0.3)',
      strong: '0 0 60px hsl(217 91% 60% / 0.4)',
    },
  },
};

// Export active theme (switch here to change theme globally)
export const activeTheme = defaultTheme;
