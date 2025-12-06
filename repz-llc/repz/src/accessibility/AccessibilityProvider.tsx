import React, { createContext, useContext, useEffect, useState } from 'react';

interface AccessibilityConfig {
  reducedMotion: boolean;
  highContrast: boolean;
  fontSize: 'small' | 'medium' | 'large' | 'extra-large';
  screenReaderOptimized: boolean;
  keyboardNavigation: boolean;
}

interface AccessibilityContextType {
  config: AccessibilityConfig;
  updateConfig: (updates: Partial<AccessibilityConfig>) => void;
  announceToScreenReader: (message: string) => void;
  focusManagement: {
    trapFocus: (element: HTMLElement) => () => void;
    restoreFocus: (element: HTMLElement) => void;
  };
}

const AccessibilityContext = createContext<AccessibilityContextType | null>(null);

export const useAccessibilityContext = () => {
  const context = useContext(AccessibilityContext);
  if (!context) {
    throw new Error('useAccessibilityContext must be used within AccessibilityProvider');
  }
  return context;
};

interface AccessibilityProviderProps {
  children: React.ReactNode;
}

export const AccessibilityProvider: React.FC<AccessibilityProviderProps> = ({ children }) => {
  const [config, setConfig] = useState<AccessibilityConfig>({
    reducedMotion: false,
    highContrast: false,
    fontSize: 'medium',
    screenReaderOptimized: false,
    keyboardNavigation: true,
  });

  useEffect(() => {
    // Detect user preferences
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const contrastQuery = window.matchMedia('(prefers-contrast: high)');
    
    setConfig(prev => ({
      ...prev,
      reducedMotion: mediaQuery.matches,
      highContrast: contrastQuery.matches,
    }));

    // Listen for changes
    const handleMotionChange = (e: MediaQueryListEvent) => {
      setConfig(prev => ({ ...prev, reducedMotion: e.matches }));
    };

    const handleContrastChange = (e: MediaQueryListEvent) => {
      setConfig(prev => ({ ...prev, highContrast: e.matches }));
    };

    mediaQuery.addEventListener('change', handleMotionChange);
    contrastQuery.addEventListener('change', handleContrastChange);

    // Apply CSS custom properties
    document.documentElement.style.setProperty(
      '--animation-speed',
      config.reducedMotion ? '0s' : '0.3s'
    );

    return () => {
      mediaQuery.removeEventListener('change', handleMotionChange);
      contrastQuery.removeEventListener('change', handleContrastChange);
    };
  }, [config.reducedMotion]);

  useEffect(() => {
    // Apply accessibility settings to document
    const root = document.documentElement;
    
    // Font size
    const fontSizeMap = {
      small: '14px',
      medium: '16px',
      large: '18px',
      'extra-large': '20px',
    };
    root.style.setProperty('--base-font-size', fontSizeMap[config.fontSize]);

    // High contrast mode
    if (config.highContrast) {
      root.classList.add('high-contrast');
    } else {
      root.classList.remove('high-contrast');
    }

    // Reduced motion
    if (config.reducedMotion) {
      root.classList.add('reduced-motion');
    } else {
      root.classList.remove('reduced-motion');
    }

    // Screen reader optimization
    if (config.screenReaderOptimized) {
      root.classList.add('screen-reader-optimized');
    } else {
      root.classList.remove('screen-reader-optimized');
    }
  }, [config]);

  const updateConfig = (updates: Partial<AccessibilityConfig>) => {
    setConfig(prev => ({ ...prev, ...updates }));
    
    // Save preferences to localStorage
    localStorage.setItem('accessibility-preferences', JSON.stringify({ ...config, ...updates }));
  };

  const announceToScreenReader = (message: string) => {
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', 'polite');
    announcement.setAttribute('aria-atomic', 'true');
    announcement.style.position = 'absolute';
    announcement.style.left = '-10000px';
    announcement.style.width = '1px';
    announcement.style.height = '1px';
    announcement.style.overflow = 'hidden';
    
    document.body.appendChild(announcement);
    announcement.textContent = message;
    
    setTimeout(() => {
      document.body.removeChild(announcement);
    }, 1000);
  };

  const trapFocus = (element: HTMLElement): (() => void) => {
    const focusableElements = element.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    const firstElement = focusableElements[0] as HTMLElement;
    const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

    const handleTabKey = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;

      if (e.shiftKey) {
        if (document.activeElement === firstElement) {
          lastElement.focus();
          e.preventDefault();
        }
      } else {
        if (document.activeElement === lastElement) {
          firstElement.focus();
          e.preventDefault();
        }
      }
    };

    const handleEscapeKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        const closeButton = element.querySelector('[data-close-button]') as HTMLElement;
        if (closeButton) {
          closeButton.click();
        }
      }
    };

    element.addEventListener('keydown', handleTabKey);
    element.addEventListener('keydown', handleEscapeKey);
    
    // Focus the first element
    if (firstElement) {
      firstElement.focus();
    }

    return () => {
      element.removeEventListener('keydown', handleTabKey);
      element.removeEventListener('keydown', handleEscapeKey);
    };
  };

  const restoreFocus = (element: HTMLElement) => {
    if (element && typeof element.focus === 'function') {
      element.focus();
    }
  };

  const value: AccessibilityContextType = {
    config,
    updateConfig,
    announceToScreenReader,
    focusManagement: {
      trapFocus,
      restoreFocus,
    },
  };

  return (
    <AccessibilityContext.Provider value={value}>
      {children}
    </AccessibilityContext.Provider>
  );
};