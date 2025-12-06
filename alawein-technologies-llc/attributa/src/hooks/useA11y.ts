import { useEffect, useState } from 'react';

export interface A11yPreferences {
  reducedMotion: boolean;
  highContrast: boolean;
  focusVisible: boolean;
  screenReader: boolean;
}

export function useA11y() {
  const [preferences, setPreferences] = useState<A11yPreferences>({
    reducedMotion: false,
    highContrast: false,
    focusVisible: false,
    screenReader: false
  });

  useEffect(() => {
    // Check for reduced motion preference
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    
    // Check for high contrast preference
    const highContrast = window.matchMedia('(prefers-contrast: high)').matches;
    
    // Check for focus-visible support
    const focusVisible = 'focus-visible' in document.createElement('div').style;
    
    // Basic screen reader detection
    const screenReader = window.navigator.userAgent.includes('NVDA') || 
                        window.navigator.userAgent.includes('JAWS') ||
                        window.speechSynthesis !== undefined;

    setPreferences({
      reducedMotion,
      highContrast,
      focusVisible,
      screenReader
    });

    // Add CSS classes based on preferences
    if (reducedMotion) {
      document.documentElement.classList.add('reduced-motion');
    }
    if (highContrast) {
      document.documentElement.classList.add('high-contrast');
    }

    // Listen for preference changes
    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const contrastQuery = window.matchMedia('(prefers-contrast: high)');
    
    const handleMotionChange = (e: MediaQueryListEvent) => {
      setPreferences(prev => ({ ...prev, reducedMotion: e.matches }));
      document.documentElement.classList.toggle('reduced-motion', e.matches);
    };
    
    const handleContrastChange = (e: MediaQueryListEvent) => {
      setPreferences(prev => ({ ...prev, highContrast: e.matches }));
      document.documentElement.classList.toggle('high-contrast', e.matches);
    };

    motionQuery.addEventListener('change', handleMotionChange);
    contrastQuery.addEventListener('change', handleContrastChange);

    return () => {
      motionQuery.removeEventListener('change', handleMotionChange);
      contrastQuery.removeEventListener('change', handleContrastChange);
    };
  }, []);

  return preferences;
}