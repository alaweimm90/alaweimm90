import React, { useEffect, useRef } from 'react';
import { cn } from '@/lib/utils';

// Focus trap component for modals and dialogs
interface FocusTrapProps {
  children: React.ReactNode;
  active?: boolean;
  restoreFocus?: boolean;
}

export function FocusTrap({ children, active = true, restoreFocus = true }: FocusTrapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!active) return;

    // Store the previously focused element
    if (restoreFocus) {
      previousFocusRef.current = document.activeElement as HTMLElement;
    }

    const container = containerRef.current;
    if (!container) return;

    // Get all focusable elements
    const getFocusableElements = () => {
      return container.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      ) as NodeListOf<HTMLElement>;
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;

      const focusableElements = getFocusableElements();
      if (focusableElements.length === 0) return;

      const firstElement = focusableElements[0];
      const lastElement = focusableElements[focusableElements.length - 1];

      if (e.shiftKey) {
        // Shift + Tab
        if (document.activeElement === firstElement) {
          e.preventDefault();
          lastElement.focus();
        }
      } else {
        // Tab
        if (document.activeElement === lastElement) {
          e.preventDefault();
          firstElement.focus();
        }
      }
    };

    // Focus the first focusable element
    const focusableElements = getFocusableElements();
    if (focusableElements.length > 0) {
      focusableElements[0].focus();
    }

    document.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      // Restore focus when trap is deactivated
      if (restoreFocus && previousFocusRef.current) {
        previousFocusRef.current.focus();
      }
    };
  }, [active, restoreFocus]);

  return (
    <div ref={containerRef} className="focus-trap">
      {children}
    </div>
  );
}

// Skip link component for keyboard navigation
interface SkipLinkProps {
  href: string;
  children: React.ReactNode;
  className?: string;
}

export function SkipLink({ href, children, className }: SkipLinkProps) {
  return (
    <a
      href={href}
      className={cn(
        // Hidden by default, visible on focus
        "sr-only focus:not-sr-only",
        "fixed top-4 left-4 z-50",
        "bg-primary text-primary-foreground",
        "px-4 py-2 rounded-md",
        "font-medium text-sm",
        "focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
        "transition-all duration-200",
        className
      )}
    >
      {children}
    </a>
  );
}

// Focus visible utility hook
export function useFocusVisible() {
  const [isFocusVisible, setIsFocusVisible] = React.useState(false);
  const ref = useRef<HTMLElement>(null);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    let hadKeyboardEvent = false;

    const onKeyDown = () => {
      hadKeyboardEvent = true;
    };

    const onFocus = () => {
      if (hadKeyboardEvent) {
        setIsFocusVisible(true);
      }
    };

    const onBlur = () => {
      setIsFocusVisible(false);
      hadKeyboardEvent = false;
    };

    const onMouseDown = () => {
      hadKeyboardEvent = false;
    };

    document.addEventListener('keydown', onKeyDown);
    element.addEventListener('focus', onFocus);
    element.addEventListener('blur', onBlur);
    element.addEventListener('mousedown', onMouseDown);

    return () => {
      document.removeEventListener('keydown', onKeyDown);
      element.removeEventListener('focus', onFocus);
      element.removeEventListener('blur', onBlur);
      element.removeEventListener('mousedown', onMouseDown);
    };
  }, []);

  return { isFocusVisible, ref };
}

// Roving tabindex hook for managing focus in lists
export function useRovingTabIndex<T extends HTMLElement>() {
  const itemsRef = useRef<T[]>([]);
  const [currentIndex, setCurrentIndex] = React.useState(0);

  const addItem = (item: T) => {
    if (!itemsRef.current.includes(item)) {
      itemsRef.current.push(item);
    }
  };

  const removeItem = (item: T) => {
    const index = itemsRef.current.indexOf(item);
    if (index > -1) {
      itemsRef.current.splice(index, 1);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent, index: number) => {
    const items = itemsRef.current;
    let newIndex = index;

    switch (e.key) {
      case 'ArrowDown':
      case 'ArrowRight':
        e.preventDefault();
        newIndex = (index + 1) % items.length;
        break;
      case 'ArrowUp':
      case 'ArrowLeft':
        e.preventDefault();
        newIndex = index === 0 ? items.length - 1 : index - 1;
        break;
      case 'Home':
        e.preventDefault();
        newIndex = 0;
        break;
      case 'End':
        e.preventDefault();
        newIndex = items.length - 1;
        break;
      default:
        return;
    }

    setCurrentIndex(newIndex);
    items[newIndex]?.focus();
  };

  return {
    addItem,
    removeItem,
    handleKeyDown,
    currentIndex,
    setCurrentIndex,
    getTabIndex: (index: number) => (index === currentIndex ? 0 : -1),
  };
}