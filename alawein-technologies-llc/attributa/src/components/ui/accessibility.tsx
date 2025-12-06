import React from 'react';
import { AlertTriangle, CheckCircle, XCircle, Info, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';

interface AccessibilityAnnouncerProps {
  message: string;
  type: 'polite' | 'assertive';
  children?: React.ReactNode;
}

export function AccessibilityAnnouncer({ message, type, children }: AccessibilityAnnouncerProps) {
  return (
    <>
      {children}
      <div
        aria-live={type}
        aria-atomic="true"
        className="sr-only"
      >
        {message}
      </div>
    </>
  );
}

interface FocusTrapProps {
  children: React.ReactNode;
  enabled?: boolean;
  onEscape?: () => void;
}

export function FocusTrap({ children, enabled = true, onEscape }: FocusTrapProps) {
  const containerRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (!enabled || !containerRef.current) return;

    const container = containerRef.current;
    const focusableElements = container.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    
    const firstElement = focusableElements[0] as HTMLElement;
    const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

    const handleTabKey = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;

      if (e.shiftKey) {
        if (document.activeElement === firstElement) {
          e.preventDefault();
          lastElement?.focus();
        }
      } else {
        if (document.activeElement === lastElement) {
          e.preventDefault();
          firstElement?.focus();
        }
      }
    };

    const handleEscapeKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onEscape?.();
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
  }, [enabled, onEscape]);

  return (
    <div ref={containerRef} className="focus:outline-none">
      {children}
    </div>
  );
}

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
        'sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50',
        'bg-primary text-primary-foreground px-4 py-2 rounded-md font-mono text-sm',
        'focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2',
        className
      )}
    >
      {children}
    </a>
  );
}

interface KeyboardShortcutProps {
  keys: string[];
  description: string;
  onActivate: () => void;
  className?: string;
}

export function KeyboardShortcut({ keys, description, onActivate, className }: KeyboardShortcutProps) {
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const isModifierPressed = (key: string) => {
        switch (key.toLowerCase()) {
          case 'ctrl':
          case 'cmd':
            return e.ctrlKey || e.metaKey;
          case 'shift':
            return e.shiftKey;
          case 'alt':
            return e.altKey;
          default:
            return e.key.toLowerCase() === key.toLowerCase();
        }
      };

      const allKeysPressed = keys.every(isModifierPressed);
      
      if (allKeysPressed) {
        e.preventDefault();
        onActivate();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [keys, onActivate]);

  return (
    <div className={cn('text-sm text-muted-foreground font-mono', className)}>
      <span className="sr-only">Keyboard shortcut: </span>
      {keys.map((key, index) => (
        <React.Fragment key={key}>
          <kbd className="px-1.5 py-0.5 bg-muted rounded border border-border text-xs">
            {key}
          </kbd>
          {index < keys.length - 1 && <span className="mx-1">+</span>}
        </React.Fragment>
      ))}
      <span className="ml-2">{description}</span>
    </div>
  );
}

interface AccessibleButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  loading?: boolean;
  loadingText?: string;
  variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link';
}

export function AccessibleButton({ 
  children, 
  loading = false, 
  loadingText = 'Loading...', 
  disabled,
  variant = 'default',
  className,
  ...props 
}: AccessibleButtonProps) {
  return (
    <Button
      variant={variant}
      disabled={disabled || loading}
      aria-disabled={disabled || loading}
      aria-describedby={loading ? 'loading-message' : undefined}
      className={cn(
        'focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2',
        'transition-all duration-200',
        className
      )}
      {...props}
    >
      {loading ? (
        <>
          <div className="animate-spin rounded-full h-4 w-4 border-2 border-transparent border-t-current mr-2" />
          <span id="loading-message">{loadingText}</span>
        </>
      ) : (
        children
      )}
    </Button>
  );
}