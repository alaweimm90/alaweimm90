import { useEffect, useState } from 'react';

export interface AccessibilityPreferences {
  reduceMotion: boolean;
  highContrast: boolean;
  prefersKeyboard: boolean;
  screenReader: boolean;
}

export const useAccessibility = () => {
  const [preferences, setPreferences] = useState<AccessibilityPreferences>({
    reduceMotion: false,
    highContrast: false,
    prefersKeyboard: false,
    screenReader: false
  });

  useEffect(() => {
    // Check for reduce motion preference
    const reduceMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const highContrastQuery = window.matchMedia('(prefers-contrast: high)');
    
    // Detect screen reader
    const hasScreenReader = () => {
      return !!(
        navigator.userAgent.match(/NVDA|JAWS|VoiceOver|WindowEyes|DRAGON|ZoomText|MagPie/) ||
        window.speechSynthesis ||
        (window.navigator as any)?.userAgentData?.brands?.some((brand: any) => 
          brand.brand.toLowerCase().includes('screen') || 
          brand.brand.toLowerCase().includes('reader')
        )
      );
    };

    const updatePreferences = () => {
      setPreferences({
        reduceMotion: reduceMotionQuery.matches,
        highContrast: highContrastQuery.matches,
        prefersKeyboard: document.activeElement?.tagName !== 'BODY',
        screenReader: hasScreenReader()
      });
    };

    updatePreferences();

    reduceMotionQuery.addEventListener('change', updatePreferences);
    highContrastQuery.addEventListener('change', updatePreferences);

    // Listen for keyboard navigation
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Tab') {
        setPreferences(prev => ({ ...prev, prefersKeyboard: true }));
      }
    };

    document.addEventListener('keydown', handleKeyDown);

    return () => {
      reduceMotionQuery.removeEventListener('change', updatePreferences);
      highContrastQuery.removeEventListener('change', updatePreferences);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  return preferences;
};

// ARIA Live Region Hook
export const useAriaLive = () => {
  const [announcement, setAnnouncement] = useState('');

  const announce = (message: string, priority: 'polite' | 'assertive' = 'polite') => {
    setAnnouncement(`${priority}: ${message}`);
    // Clear after announcement
    setTimeout(() => setAnnouncement(''), 1000);
  };

  return { announcement, announce };
};

// Focus Management Hook
export const useFocusManagement = () => {
  const focusFirst = (container: HTMLElement) => {
    const focusable = container.querySelector<HTMLElement>(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    focusable?.focus();
  };

  const trapFocus = (container: HTMLElement) => {
    const focusableElements = container.querySelectorAll<HTMLElement>(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    
    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    const handleTabKey = (e: KeyboardEvent) => {
      if (e.key === 'Tab') {
        if (e.shiftKey) {
          if (document.activeElement === firstElement) {
            e.preventDefault();
            lastElement.focus();
          }
        } else {
          if (document.activeElement === lastElement) {
            e.preventDefault();
            firstElement.focus();
          }
        }
      }

      if (e.key === 'Escape') {
        container.removeEventListener('keydown', handleTabKey);
      }
    };

    container.addEventListener('keydown', handleTabKey);
    firstElement?.focus();

    return () => container.removeEventListener('keydown', handleTabKey);
  };

  return { focusFirst, trapFocus };
};