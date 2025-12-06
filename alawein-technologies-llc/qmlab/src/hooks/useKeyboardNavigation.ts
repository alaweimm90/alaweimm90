import { useEffect, useCallback, useRef } from 'react';
import { trackQuantumEvents } from '@/lib/analytics';

interface FocusableElement extends Omit<HTMLElement, 'tabIndex'> {
  tabIndex: number;
}

export const useKeyboardNavigation = () => {
  const focusHistoryRef = useRef<HTMLElement[]>([]);
  const currentFocusRef = useRef<HTMLElement | null>(null);

  // Get all focusable elements in order
  const getFocusableElements = useCallback((): FocusableElement[] => {
    const selectors = [
      'a[href]',
      'button:not([disabled])',
      'input:not([disabled])',
      'select:not([disabled])',
      'textarea:not([disabled])',
      '[tabindex]:not([tabindex="-1"])',
      '[role="button"]:not([disabled])',
      '[role="tab"]:not([disabled])',
      '[role="menuitem"]:not([disabled])'
    ].join(', ');

    return Array.from(document.querySelectorAll(selectors))
      .filter((el): el is FocusableElement => {
        const element = el as HTMLElement;
        return (
          element.offsetWidth > 0 &&
          element.offsetHeight > 0 &&
          !element.hidden &&
          window.getComputedStyle(element).visibility !== 'hidden'
        );
      })
      .sort((a, b) => {
        const aTabIndex = a.tabIndex || 0;
        const bTabIndex = b.tabIndex || 0;
        if (aTabIndex !== bTabIndex) {
          return aTabIndex - bTabIndex;
        }
        // DOM order for same tabindex
        return a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1;
      });
  }, []);

  // Roving tabindex for complex widgets
  const setRovingTabIndex = useCallback((container: HTMLElement, activeElement: HTMLElement) => {
    const focusableChildren = container.querySelectorAll('[role="tab"], [role="menuitem"], [role="gridcell"]');
    
    focusableChildren.forEach((child) => {
      (child as HTMLElement).tabIndex = child === activeElement ? 0 : -1;
    });
  }, []);

  // Skip links enhancement
  const enhanceSkipLinks = useCallback(() => {
    const skipLinks = document.querySelectorAll('.skip-to-main, [href^="#"]');
    
    skipLinks.forEach((link) => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        const href = (link as HTMLAnchorElement).getAttribute('href');
        if (href) {
          const target = document.querySelector(href);
          if (target) {
            // Ensure target is focusable
            const targetElement = target as HTMLElement;
            if (!targetElement.hasAttribute('tabindex')) {
              targetElement.tabIndex = -1;
            }
            
            targetElement.focus();
            targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
            
            // Track skip link usage
            trackQuantumEvents.featureDiscovery('skip_link_used', 'keyboard');
          }
        }
      });
    });
  }, []);

  // Focus trap for modals/dialogs
  const createFocusTrap = useCallback((container: HTMLElement) => {
    const focusableElements = container.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    ) as NodeListOf<HTMLElement>;
    
    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    const trapFocus = (e: KeyboardEvent) => {
      if (e.key === 'Tab') {
        if (e.shiftKey) {
          // Shift + Tab
          if (document.activeElement === firstElement) {
            e.preventDefault();
            lastElement?.focus();
          }
        } else {
          // Tab
          if (document.activeElement === lastElement) {
            e.preventDefault();
            firstElement?.focus();
          }
        }
      }
      
      // Escape to close
      if (e.key === 'Escape') {
        const closeButton = container.querySelector('[data-close], [aria-label*="close"]') as HTMLElement;
        closeButton?.click();
      }
    };

    container.addEventListener('keydown', trapFocus);
    firstElement?.focus();

    return () => {
      container.removeEventListener('keydown', trapFocus);
    };
  }, []);

  // Arrow key navigation for complex widgets
  const handleArrowNavigation = useCallback((e: KeyboardEvent, container: HTMLElement, orientation: 'horizontal' | 'vertical' | 'grid' = 'horizontal') => {
    const items = Array.from(container.querySelectorAll('[role="tab"], [role="menuitem"], [role="gridcell"]')) as HTMLElement[];
    const currentIndex = items.indexOf(document.activeElement as HTMLElement);
    
    if (currentIndex === -1) return;
    
    let nextIndex = currentIndex;
    
    switch (e.key) {
      case 'ArrowRight':
        if (orientation === 'horizontal' || orientation === 'grid') {
          nextIndex = (currentIndex + 1) % items.length;
        }
        break;
      case 'ArrowLeft':
        if (orientation === 'horizontal' || orientation === 'grid') {
          nextIndex = currentIndex === 0 ? items.length - 1 : currentIndex - 1;
        }
        break;
      case 'ArrowDown':
        if (orientation === 'vertical' || orientation === 'grid') {
          nextIndex = (currentIndex + 1) % items.length;
        }
        break;
      case 'ArrowUp':
        if (orientation === 'vertical' || orientation === 'grid') {
          nextIndex = currentIndex === 0 ? items.length - 1 : currentIndex - 1;
        }
        break;
      case 'Home':
        nextIndex = 0;
        break;
      case 'End':
        nextIndex = items.length - 1;
        break;
      default:
        return;
    }
    
    e.preventDefault();
    items[nextIndex]?.focus();
    setRovingTabIndex(container, items[nextIndex]);
  }, [setRovingTabIndex]);

  // Enhanced focus management
  useEffect(() => {
    const handleFocusChange = (e: FocusEvent) => {
      const target = e.target as HTMLElement;
      
      if (target && target !== currentFocusRef.current) {
        // Track focus history for better navigation
        if (currentFocusRef.current) {
          focusHistoryRef.current.push(currentFocusRef.current);
          
          // Keep history manageable
          if (focusHistoryRef.current.length > 10) {
            focusHistoryRef.current.shift();
          }
        }
        
        currentFocusRef.current = target;
        
        // Announce focus changes for screen readers
        const announcement = target.getAttribute('aria-label') || 
                           target.getAttribute('title') || 
                           target.textContent?.slice(0, 50);
        
        if (announcement && target.matches('button, [role="button"], [role="tab"]')) {
          // Use aria-live region for announcements
          const announcer = document.getElementById('focus-announcer');
          if (announcer) {
            announcer.textContent = `Focused: ${announcement}`;
          }
        }
      }
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      // Global keyboard shortcuts
      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case '/':
            // Focus search (if available)
            e.preventDefault();
            const searchInput = document.querySelector('input[type="search"], input[placeholder*="search"]') as HTMLElement;
            searchInput?.focus();
            trackQuantumEvents.featureDiscovery('keyboard_shortcut_search', 'keyboard');
            break;
        }
      }
      
      // Alt + number for quick navigation
      if (e.altKey && /^[1-9]$/.test(e.key)) {
        e.preventDefault();
        const landmarks = document.querySelectorAll('nav, main, aside, section[aria-label]');
        const index = parseInt(e.key) - 1;
        if (landmarks[index]) {
          (landmarks[index] as HTMLElement).focus();
          trackQuantumEvents.featureDiscovery('landmark_navigation', 'keyboard');
        }
      }
    };

    document.addEventListener('focusin', handleFocusChange);
    document.addEventListener('keydown', handleKeyDown);
    
    // Initialize enhancements
    enhanceSkipLinks();

    return () => {
      document.removeEventListener('focusin', handleFocusChange);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [enhanceSkipLinks]);

  return {
    getFocusableElements,
    setRovingTabIndex,
    createFocusTrap,
    handleArrowNavigation,
    focusHistory: focusHistoryRef.current,
    currentFocus: currentFocusRef.current
  };
};

// Screen reader optimizations
export const useScreenReaderOptimizations = () => {
  useEffect(() => {
    // Create live region for announcements
    const announcer = document.createElement('div');
    announcer.id = 'focus-announcer';
    announcer.setAttribute('aria-live', 'polite');
    announcer.setAttribute('aria-atomic', 'true');
    announcer.className = 'sr-only';
    document.body.appendChild(announcer);

    // Create status announcer for quantum operations
    const statusAnnouncer = document.createElement('div');
    statusAnnouncer.id = 'status-announcer';
    statusAnnouncer.setAttribute('aria-live', 'assertive');
    statusAnnouncer.setAttribute('aria-atomic', 'true');
    statusAnnouncer.className = 'sr-only';
    document.body.appendChild(statusAnnouncer);

    return () => {
      announcer.remove();
      statusAnnouncer.remove();
    };
  }, []);

  const announceToScreenReader = useCallback((message: string, priority: 'polite' | 'assertive' = 'polite') => {
    const announcerId = priority === 'assertive' ? 'status-announcer' : 'focus-announcer';
    const announcer = document.getElementById(announcerId);
    
    if (announcer) {
      // Clear first to ensure the message is announced
      announcer.textContent = '';
      setTimeout(() => {
        announcer.textContent = message;
      }, 100);
    }
  }, []);

  return { announceToScreenReader };
};