import { useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { announceToScreenReader } from "@/lib/screen-reader";

interface KeyboardShortcutConfig {
  key: string;
  altKey?: boolean;
  ctrlKey?: boolean;
  shiftKey?: boolean;
  metaKey?: boolean;
  action: () => void;
  description: string;
  preventDefault?: boolean;
}

// Global keyboard shortcuts for the application
export const useKeyboardShortcuts = () => {
  const navigate = useNavigate();

  // Navigation shortcuts
  const navigationShortcuts: KeyboardShortcutConfig[] = [
    {
      key: 'h',
      altKey: true,
      action: () => {
        navigate('/');
        announceToScreenReader('Navigated to home page');
      },
      description: 'Go to home page'
    },
    {
      key: 'a',
      altKey: true,
      action: () => {
        navigate('/arena');
        announceToScreenReader('Navigated to Arena evaluation');
      },
      description: 'Open Arena evaluation'
    },
    {
      key: 'b',
      altKey: true,
      action: () => {
        navigate('/bench');
        announceToScreenReader('Navigated to Bench evaluation');
      },
      description: 'Open Bench evaluation'
    },
    {
      key: 'd',
      altKey: true,
      action: () => {
        navigate('/dashboard');
        announceToScreenReader('Navigated to Dashboard');
      },
      description: 'Open Dashboard'
    },
    {
      key: 's',
      altKey: true,
      action: () => {
        navigate('/settings');
        announceToScreenReader('Navigated to Settings');
      },
      description: 'Open Settings'
    },
  ];

  // Accessibility shortcuts
  const accessibilityShortcuts: KeyboardShortcutConfig[] = [
    {
      key: 'c',
      altKey: true,
      action: () => {
        document.documentElement.classList.toggle('a11y-high-contrast');
        const isEnabled = document.documentElement.classList.contains('a11y-high-contrast');
        announceToScreenReader(`High contrast mode ${isEnabled ? 'enabled' : 'disabled'}`);
        
        // Save preference
        localStorage.setItem('accessibility-high-contrast', isEnabled.toString());
      },
      description: 'Toggle high contrast mode'
    },
    {
      key: 'l',
      altKey: true,
      action: () => {
        document.documentElement.classList.toggle('a11y-large-text');
        const isEnabled = document.documentElement.classList.contains('a11y-large-text');
        announceToScreenReader(`Large text mode ${isEnabled ? 'enabled' : 'disabled'}`);
        
        // Save preference
        localStorage.setItem('accessibility-large-text', isEnabled.toString());
      },
      description: 'Toggle large text mode'
    },
    {
      key: 'm',
      altKey: true,
      action: () => {
        document.documentElement.classList.toggle('a11y-reduced-motion');
        const isEnabled = document.documentElement.classList.contains('a11y-reduced-motion');
        announceToScreenReader(`Reduced motion ${isEnabled ? 'enabled' : 'disabled'}`);
        
        // Save preference
        localStorage.setItem('accessibility-reduced-motion', isEnabled.toString());
      },
      description: 'Toggle reduced motion'
    },
    {
      key: 'r',
      altKey: true,
      action: () => {
        document.documentElement.classList.toggle('a11y-screen-reader');
        const isEnabled = document.documentElement.classList.contains('a11y-screen-reader');
        announceToScreenReader(`Screen reader mode ${isEnabled ? 'enabled' : 'disabled'}`);
        
        // Save preference
        localStorage.setItem('accessibility-screen-reader', isEnabled.toString());
      },
      description: 'Toggle screen reader mode'
    },
    {
      key: 't',
      altKey: true,
      action: () => {
        // This will be handled by the AccessibilityToolbar component
        const toolbar = document.querySelector('[aria-label*="accessibility toolbar"]') as HTMLButtonElement;
        if (toolbar) {
          toolbar.click();
          announceToScreenReader('Accessibility toolbar toggled');
        }
      },
      description: 'Toggle accessibility toolbar'
    },
  ];

  // Utility shortcuts
  const utilityShortcuts: KeyboardShortcutConfig[] = [
    {
      key: 'k',
      altKey: true,
      action: () => {
        // This will be handled by the KeyboardShortcutsModal component
        announceToScreenReader('Keyboard shortcuts dialog opened');
      },
      description: 'Open keyboard shortcuts'
    },
    {
      key: '/',
      action: () => {
        // Focus search if available
        const searchInput = document.querySelector('input[type="search"], input[placeholder*="search" i]') as HTMLInputElement;
        if (searchInput) {
          searchInput.focus();
          announceToScreenReader('Search field focused');
        }
      },
      description: 'Focus search field',
      preventDefault: true
    },
  ];

  const allShortcuts = [
    ...navigationShortcuts,
    ...accessibilityShortcuts,
    ...utilityShortcuts,
  ];

  // Handle keyboard events
  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    // Don't trigger shortcuts when user is typing in input fields
    const activeElement = document.activeElement;
    const isTyping = activeElement && (
      activeElement.tagName === 'INPUT' ||
      activeElement.tagName === 'TEXTAREA' ||
      activeElement.hasAttribute('contenteditable')
    );

    if (isTyping) return;

    // Find matching shortcut
    const matchingShortcut = allShortcuts.find(shortcut => {
      return (
        shortcut.key.toLowerCase() === event.key.toLowerCase() &&
        !!shortcut.altKey === event.altKey &&
        !!shortcut.ctrlKey === event.ctrlKey &&
        !!shortcut.shiftKey === event.shiftKey &&
        !!shortcut.metaKey === event.metaKey
      );
    });

    if (matchingShortcut) {
      if (matchingShortcut.preventDefault !== false) {
        event.preventDefault();
      }
      matchingShortcut.action();
    }
  }, [allShortcuts]);

  // Skip to main content shortcut (always available)
  const handleSkipToMain = useCallback((event: KeyboardEvent) => {
    // Handle skip link when focused and Enter/Space is pressed
    if (event.target instanceof HTMLAnchorElement && 
        event.target.classList.contains('skip-link') &&
        (event.key === 'Enter' || event.key === ' ')) {
      event.preventDefault();
      const mainContent = document.getElementById('main');
      if (mainContent) {
        mainContent.focus();
        mainContent.scrollIntoView({ behavior: 'smooth', block: 'start' });
        announceToScreenReader('Skipped to main content');
      }
    }
  }, []);

  // Load saved accessibility preferences on mount
  useEffect(() => {
    // Restore accessibility settings
    const highContrast = localStorage.getItem('accessibility-high-contrast') === 'true';
    const largeText = localStorage.getItem('accessibility-large-text') === 'true';
    const reducedMotion = localStorage.getItem('accessibility-reduced-motion') === 'true';
    const screenReader = localStorage.getItem('accessibility-screen-reader') === 'true';

    if (highContrast) {
      document.documentElement.classList.add('a11y-high-contrast');
    }
    if (largeText) {
      document.documentElement.classList.add('a11y-large-text');
    }
    if (reducedMotion) {
      document.documentElement.classList.add('a11y-reduced-motion');
    }
    if (screenReader) {
      document.documentElement.classList.add('a11y-screen-reader');
    }
  }, []);

  // Set up event listeners
  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('keydown', handleSkipToMain);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('keydown', handleSkipToMain);
    };
  }, [handleKeyDown, handleSkipToMain]);

  // Return shortcuts for documentation
  return { shortcuts: allShortcuts };
};

// Hook for component-specific keyboard shortcuts
export const useComponentShortcuts = (shortcuts: KeyboardShortcutConfig[], enabled: boolean = true) => {
  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    if (!enabled) return;

    // Don't trigger shortcuts when user is typing in input fields
    const activeElement = document.activeElement;
    const isTyping = activeElement && (
      activeElement.tagName === 'INPUT' ||
      activeElement.tagName === 'TEXTAREA' ||
      activeElement.hasAttribute('contenteditable')
    );

    if (isTyping) return;

    // Find matching shortcut
    const matchingShortcut = shortcuts.find(shortcut => {
      return (
        shortcut.key.toLowerCase() === event.key.toLowerCase() &&
        !!shortcut.altKey === event.altKey &&
        !!shortcut.ctrlKey === event.ctrlKey &&
        !!shortcut.shiftKey === event.shiftKey &&
        !!shortcut.metaKey === event.metaKey
      );
    });

    if (matchingShortcut) {
      if (matchingShortcut.preventDefault !== false) {
        event.preventDefault();
      }
      matchingShortcut.action();
    }
  }, [shortcuts, enabled]);

  useEffect(() => {
    if (enabled) {
      document.addEventListener('keydown', handleKeyDown);
      return () => document.removeEventListener('keydown', handleKeyDown);
    }
  }, [handleKeyDown, enabled]);
};