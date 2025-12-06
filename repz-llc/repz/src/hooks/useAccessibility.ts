import { useState, useEffect, useCallback, useRef } from 'react';

interface AccessibilityPreferences {
  prefersReducedMotion: boolean;
  prefersHighContrast: boolean;
  prefersLargeText: boolean;
  prefersColorScheme: 'light' | 'dark' | 'no-preference';
  screenReaderDetected: boolean;
  keyboardNavigation: boolean;
}

interface FocusManagement {
  trapFocus: (container: HTMLElement) => () => void;
  restoreFocus: (element?: HTMLElement) => void;
  announceLiveRegion: (message: string, priority?: 'polite' | 'assertive') => void;
  setAriaLive: (element: HTMLElement, value: 'polite' | 'assertive' | 'off') => void;
}

interface KeyboardNavigation {
  currentFocusIndex: number;
  focusableElements: HTMLElement[];
  moveToNext: () => void;
  moveToPrevious: () => void;
  moveToFirst: () => void;
  moveToLast: () => void;
}

export function useAccessibility() {
  const [preferences, setPreferences] = useState<AccessibilityPreferences>({
    prefersReducedMotion: false,
    prefersHighContrast: false,
    prefersLargeText: false,
    prefersColorScheme: 'no-preference',
    screenReaderDetected: false,
    keyboardNavigation: false,
  });

  const [keyboardNav, setKeyboardNav] = useState<KeyboardNavigation>({
    currentFocusIndex: -1,
    focusableElements: [],
    moveToNext: () => {},
    moveToPrevious: () => {},
    moveToFirst: () => {},
    moveToLast: () => {},
  });

  const liveRegionRef = useRef<HTMLDivElement>(null);
  const lastFocusedElementRef = useRef<HTMLElement | null>(null);

  // Detect accessibility preferences
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const detectPreferences = () => {
      const mediaQueries = {
        prefersReducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)'),
        prefersHighContrast: window.matchMedia('(prefers-contrast: high)'),
        prefersLargeText: window.matchMedia('(min-resolution: 2dppx)'), // Proxy for large text
        prefersColorScheme: window.matchMedia('(prefers-color-scheme: dark)'),
      };

      // Detect screen reader
      const screenReaderDetected = 
        navigator.userAgent.includes('NVDA') ||
        navigator.userAgent.includes('JAWS') ||
        window.speechSynthesis?.getVoices().length > 0 ||
        'speechSynthesis' in window;

      setPreferences({
        prefersReducedMotion: mediaQueries.prefersReducedMotion.matches,
        prefersHighContrast: mediaQueries.prefersHighContrast.matches,
        prefersLargeText: mediaQueries.prefersLargeText.matches,
        prefersColorScheme: mediaQueries.prefersColorScheme.matches ? 'dark' : 'light',
        screenReaderDetected,
        keyboardNavigation: false, // Will be set on first tab keypress
      });

      // Listen for changes
      Object.values(mediaQueries).forEach(mq => {
        mq.addEventListener('change', detectPreferences);
      });

      return () => {
        Object.values(mediaQueries).forEach(mq => {
          mq.removeEventListener('change', detectPreferences);
        });
      };
    };

    const cleanup = detectPreferences();
    return cleanup;
  }, []);

  // Detect keyboard navigation
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Tab') {
        setPreferences(prev => ({ ...prev, keyboardNavigation: true }));
      }
    };

    const handleMouseDown = () => {
      setPreferences(prev => ({ ...prev, keyboardNavigation: false }));
    };

    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('mousedown', handleMouseDown);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('mousedown', handleMouseDown);
    };
  }, []);

  // Create live region for announcements
  useEffect(() => {
    if (typeof window === 'undefined') return;

    if (!liveRegionRef.current) {
      const liveRegion = document.createElement('div');
      liveRegion.setAttribute('aria-live', 'polite');
      liveRegion.setAttribute('aria-atomic', 'true');
      liveRegion.style.position = 'absolute';
      liveRegion.style.left = '-10000px';
      liveRegion.style.width = '1px';
      liveRegion.style.height = '1px';
      liveRegion.style.overflow = 'hidden';
      document.body.appendChild(liveRegion);
      liveRegionRef.current = liveRegion;
    }

    return () => {
      if (liveRegionRef.current) {
        document.body.removeChild(liveRegionRef.current);
        liveRegionRef.current = null;
      }
    };
  }, []);

  // Focus management utilities
  const focusManagement: FocusManagement = {
    trapFocus: useCallback((container: HTMLElement) => {
      const focusableElements = Array.from(
        container.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        )
      ).filter(el => !el.hasAttribute('disabled') && !el.getAttribute('aria-hidden'));

      const firstElement = focusableElements[0];
      const lastElement = focusableElements[focusableElements.length - 1];

      const handleTabKey = (event: KeyboardEvent) => {
        if (event.key !== 'Tab') return;

        if (event.shiftKey) {
          if (document.activeElement === firstElement) {
            event.preventDefault();
            lastElement?.focus();
          }
        } else {
          if (document.activeElement === lastElement) {
            event.preventDefault();
            firstElement?.focus();
          }
        }
      };

      const handleEscapeKey = (event: KeyboardEvent) => {
        if (event.key === 'Escape') {
          const closeButton = container.querySelector('[data-dismiss]') as HTMLElement;
          closeButton?.click();
        }
      };

      container.addEventListener('keydown', handleTabKey);
      container.addEventListener('keydown', handleEscapeKey);
      
      // Focus first element
      firstElement?.focus();

      return () => {
        container.removeEventListener('keydown', handleTabKey);
        container.removeEventListener('keydown', handleEscapeKey);
      };
    }, []),

    restoreFocus: useCallback((element?: HTMLElement) => {
      const targetElement = element || lastFocusedElementRef.current;
      if (targetElement && document.contains(targetElement)) {
        targetElement.focus();
      }
    }, []),

    announceLiveRegion: useCallback((message: string, priority: 'polite' | 'assertive' = 'polite') => {
      if (!liveRegionRef.current) return;

      liveRegionRef.current.setAttribute('aria-live', priority);
      liveRegionRef.current.textContent = message;

      // Clear after announcement
      setTimeout(() => {
        if (liveRegionRef.current) {
          liveRegionRef.current.textContent = '';
        }
      }, 1000);
    }, []),

    setAriaLive: useCallback((element: HTMLElement, value: 'polite' | 'assertive' | 'off') => {
      element.setAttribute('aria-live', value);
    }, []),
  };

  // Keyboard navigation for custom components
  const setupKeyboardNavigation = useCallback((container: HTMLElement) => {
    const updateFocusableElements = () => {
      const elements = Array.from(
        container.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        )
      ).filter(el => !el.hasAttribute('disabled') && !el.getAttribute('aria-hidden'));

      setKeyboardNav(prev => ({
        ...prev,
        focusableElements: elements,
        currentFocusIndex: elements.indexOf(document.activeElement as HTMLElement),
      }));
    };

    updateFocusableElements();

    const handleKeyDown = (event: KeyboardEvent) => {
      const { focusableElements, currentFocusIndex } = keyboardNav;

      switch (event.key) {
        case 'ArrowDown':
        case 'ArrowRight':
          event.preventDefault();
          keyboardNav.moveToNext();
          break;
        case 'ArrowUp':
        case 'ArrowLeft':
          event.preventDefault();
          keyboardNav.moveToPrevious();
          break;
        case 'Home':
          event.preventDefault();
          keyboardNav.moveToFirst();
          break;
        case 'End':
          event.preventDefault();
          keyboardNav.moveToLast();
          break;
      }
    };

    container.addEventListener('keydown', handleKeyDown);
    container.addEventListener('focus', updateFocusableElements, true);

    return () => {
      container.removeEventListener('keydown', handleKeyDown);
      container.removeEventListener('focus', updateFocusableElements, true);
    };
  }, [keyboardNav]);

  // Update keyboard navigation methods
  useEffect(() => {
    setKeyboardNav(prev => ({
      ...prev,
      moveToNext: () => {
        const nextIndex = (prev.currentFocusIndex + 1) % prev.focusableElements.length;
        prev.focusableElements[nextIndex]?.focus();
        setKeyboardNav(current => ({ ...current, currentFocusIndex: nextIndex }));
      },
      moveToPrevious: () => {
        const prevIndex = prev.currentFocusIndex <= 0 
          ? prev.focusableElements.length - 1 
          : prev.currentFocusIndex - 1;
        prev.focusableElements[prevIndex]?.focus();
        setKeyboardNav(current => ({ ...current, currentFocusIndex: prevIndex }));
      },
      moveToFirst: () => {
        prev.focusableElements[0]?.focus();
        setKeyboardNav(current => ({ ...current, currentFocusIndex: 0 }));
      },
      moveToLast: () => {
        const lastIndex = prev.focusableElements.length - 1;
        prev.focusableElements[lastIndex]?.focus();
        setKeyboardNav(current => ({ ...current, currentFocusIndex: lastIndex }));
      },
    }));
  }, [keyboardNav.focusableElements, keyboardNav.currentFocusIndex]);

  // Color contrast checker
  const checkColorContrast = useCallback((
    foreground: string, 
    background: string
  ): { ratio: number; passesAA: boolean; passesAAA: boolean } => {
    // Convert hex to RGB
    const hexToRgb = (hex: string) => {
      const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
      return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
      } : null;
    };

    // Calculate relative luminance
    const getLuminance = (r: number, g: number, b: number) => {
      const [rs, gs, bs] = [r, g, b].map(c => {
        c = c / 255;
        return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
      });
      return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
    };

    const fgRgb = hexToRgb(foreground);
    const bgRgb = hexToRgb(background);

    if (!fgRgb || !bgRgb) {
      return { ratio: 0, passesAA: false, passesAAA: false };
    }

    const fgLum = getLuminance(fgRgb.r, fgRgb.g, fgRgb.b);
    const bgLum = getLuminance(bgRgb.r, bgRgb.g, bgRgb.b);

    const ratio = (Math.max(fgLum, bgLum) + 0.05) / (Math.min(fgLum, bgLum) + 0.05);

    return {
      ratio: Math.round(ratio * 100) / 100,
      passesAA: ratio >= 4.5,
      passesAAA: ratio >= 7,
    };
  }, []);

  // ARIA utilities
  const ariaUtils = {
    setAriaLabel: (element: HTMLElement, label: string) => {
      element.setAttribute('aria-label', label);
    },
    setAriaDescribedBy: (element: HTMLElement, id: string) => {
      element.setAttribute('aria-describedby', id);
    },
    setAriaExpanded: (element: HTMLElement, expanded: boolean) => {
      element.setAttribute('aria-expanded', expanded.toString());
    },
    setAriaSelected: (element: HTMLElement, selected: boolean) => {
      element.setAttribute('aria-selected', selected.toString());
    },
    setAriaPressed: (element: HTMLElement, pressed: boolean) => {
      element.setAttribute('aria-pressed', pressed.toString());
    },
    setRole: (element: HTMLElement, role: string) => {
      element.setAttribute('role', role);
    },
  };

  // Store last focused element
  useEffect(() => {
    const handleFocus = (event: FocusEvent) => {
      if (event.target instanceof HTMLElement) {
        lastFocusedElementRef.current = event.target;
      }
    };

    document.addEventListener('focusin', handleFocus);
    return () => document.removeEventListener('focusin', handleFocus);
  }, []);

  return {
    preferences,
    keyboardNav,
    focusManagement,
    setupKeyboardNavigation,
    checkColorContrast,
    ariaUtils,
    
    // Utility getters
    isScreenReaderActive: preferences.screenReaderDetected,
    shouldReduceMotion: preferences.prefersReducedMotion,
    isHighContrast: preferences.prefersHighContrast,
    isKeyboardUser: preferences.keyboardNavigation,
    isDarkMode: preferences.prefersColorScheme === 'dark',
  };
}