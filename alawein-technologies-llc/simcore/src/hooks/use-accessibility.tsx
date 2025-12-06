/**
 * Accessibility Hook
 * 
 * Provides comprehensive accessibility utilities for SimCore components
 * including focus management, ARIA announcements, and keyboard navigation.
 */

import { useEffect, useRef, useState, useCallback } from 'react';

export interface AccessibilityOptions {
  announceChanges?: boolean;
  focusOnMount?: boolean;
  trapFocus?: boolean;
  respectReducedMotion?: boolean;
}

export function useAccessibility(options: AccessibilityOptions = {}) {
  const {
    announceChanges = true,
    focusOnMount = false,
    trapFocus = false,
    respectReducedMotion = true
  } = options;

  const elementRef = useRef<HTMLElement>(null);
  const [announcements, setAnnouncements] = useState<string[]>([]);
  const [isReducedMotion, setIsReducedMotion] = useState(false);

  // Check for reduced motion preference
  useEffect(() => {
    if (!respectReducedMotion) return;
    
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setIsReducedMotion(mediaQuery.matches);
    
    const handleChange = (e: MediaQueryListEvent) => {
      setIsReducedMotion(e.matches);
    };
    
    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, [respectReducedMotion]);

  // Focus management
  const focusElement = useCallback(() => {
    if (elementRef.current) {
      elementRef.current.focus();
    }
  }, []);

  useEffect(() => {
    if (focusOnMount) {
      focusElement();
    }
  }, [focusOnMount, focusElement]);

  // ARIA announcements
  const announce = useCallback((message: string, priority: 'polite' | 'assertive' = 'polite') => {
    if (!announceChanges) return;
    
    setAnnouncements(prev => [...prev, message]);
    
    // Clear announcement after delay
    setTimeout(() => {
      setAnnouncements(prev => prev.slice(1));
    }, 1000);
  }, [announceChanges]);

  // Focus trap for modals/dialogs
  useEffect(() => {
    if (!trapFocus || !elementRef.current) return;

    const element = elementRef.current;
    const focusableElements = element.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    
    const firstFocusable = focusableElements[0] as HTMLElement;
    const lastFocusable = focusableElements[focusableElements.length - 1] as HTMLElement;

    const handleTabKey = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;

      if (e.shiftKey) {
        if (document.activeElement === firstFocusable) {
          e.preventDefault();
          lastFocusable.focus();
        }
      } else {
        if (document.activeElement === lastFocusable) {
          e.preventDefault();
          firstFocusable.focus();
        }
      }
    };

    element.addEventListener('keydown', handleTabKey);
    return () => element.removeEventListener('keydown', handleTabKey);
  }, [trapFocus]);

  // Keyboard navigation helpers
  const handleKeyNavigation = useCallback((
    e: React.KeyboardEvent, 
    onEnter?: () => void,
    onEscape?: () => void,
    onArrowUp?: () => void,
    onArrowDown?: () => void,
    onArrowLeft?: () => void,
    onArrowRight?: () => void
  ) => {
    switch (e.key) {
      case 'Enter':
      case ' ':
        e.preventDefault();
        onEnter?.();
        break;
      case 'Escape':
        onEscape?.();
        break;
      case 'ArrowUp':
        e.preventDefault();
        onArrowUp?.();
        break;
      case 'ArrowDown':
        e.preventDefault();
        onArrowDown?.();
        break;
      case 'ArrowLeft':
        e.preventDefault();
        onArrowLeft?.();
        break;
      case 'ArrowRight':
        e.preventDefault();
        onArrowRight?.();
        break;
    }
  }, []);

  return {
    elementRef,
    announcements,
    isReducedMotion,
    announce,
    focusElement,
    handleKeyNavigation,
    // Accessibility props for components
    accessibilityProps: {
      ref: elementRef,
      'aria-live': announceChanges ? 'polite' as const : undefined,
      'aria-atomic': announceChanges ? true : undefined,
    }
  };
}

export function useSkipLinks() {
  const [skipLinks, setSkipLinks] = useState<Array<{ id: string; label: string }>>([]);

  const registerSkipLink = useCallback((id: string, label: string) => {
    setSkipLinks(prev => [...prev, { id, label }]);
  }, []);

  const unregisterSkipLink = useCallback((id: string) => {
    setSkipLinks(prev => prev.filter(link => link.id !== id));
  }, []);

  const skipToContent = useCallback((id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.focus();
      element.scrollIntoView({ behavior: 'smooth' });
    }
  }, []);

  return {
    skipLinks,
    registerSkipLink,
    unregisterSkipLink,
    skipToContent
  };
}

export function useFocusManagement() {
  const previousFocus = useRef<HTMLElement | null>(null);

  const saveFocus = useCallback(() => {
    previousFocus.current = document.activeElement as HTMLElement;
  }, []);

  const restoreFocus = useCallback(() => {
    if (previousFocus.current) {
      previousFocus.current.focus();
      previousFocus.current = null;
    }
  }, []);

  const focusFirst = useCallback((container: HTMLElement) => {
    const focusable = container.querySelector(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    ) as HTMLElement;
    
    if (focusable) {
      focusable.focus();
    }
  }, []);

  return {
    saveFocus,
    restoreFocus,
    focusFirst
  };
}

export function useKeyboardNavigation(onNavigate: (direction: 'up' | 'down' | 'left' | 'right') => void) {
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    switch (e.key) {
      case 'ArrowUp':
        e.preventDefault();
        onNavigate('up');
        break;
      case 'ArrowDown':
        e.preventDefault();
        onNavigate('down');
        break;
      case 'ArrowLeft':
        e.preventDefault();
        onNavigate('left');
        break;
      case 'ArrowRight':
        e.preventDefault();
        onNavigate('right');
        break;
    }
  }, [onNavigate]);

  return { handleKeyDown };
}