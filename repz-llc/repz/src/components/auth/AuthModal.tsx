import React, { useEffect, useRef } from 'react';
import { X } from 'lucide-react';
import { Button } from '@/ui/atoms/Button';
import { RepzLogo } from '@/ui/organisms/RepzLogo';

interface AuthModalProps {
  isOpen: boolean;
  mode: 'login' | 'signup';
  onClose: () => void;
}

/**
 * Progressive Enhancement Authentication Modal
 * 
 * Features:
 * - WCAG 2.2 AA compliant modal with focus trapping
 * - Keyboard navigation and ESC key support
 * - Accessible dialog with proper ARIA attributes
 * - Backdrop click handling
 * - Progressive enhancement: redirects to canonical pages
 */
export function AuthModal({ isOpen, mode, onClose }: AuthModalProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const previousActiveElement = useRef<HTMLElement | null>(null);

  // Focus management for accessibility
  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;

    if (isOpen) {
      // Store the previously focused element
      previousActiveElement.current = document.activeElement as HTMLElement;
      
      // Show the dialog
      dialog.showModal();
      
      // Focus the first focusable element in the modal
      const firstFocusable = dialog.querySelector(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      ) as HTMLElement;
      firstFocusable?.focus();
    } else {
      // Close the dialog
      dialog.close();
      
      // Restore focus to the previously focused element
      previousActiveElement.current?.focus();
    }
  }, [isOpen]);

  // Handle ESC key and backdrop clicks
  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;

    const handleClose = (e: Event) => {
      onClose();
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    dialog.addEventListener('close', handleClose);
    dialog.addEventListener('keydown', handleKeyDown);

    return () => {
      dialog.removeEventListener('close', handleClose);
      dialog.removeEventListener('keydown', handleKeyDown);
    };
  }, [onClose]);

  // Focus trapping
  useEffect(() => {
    if (!isOpen) return;

    const dialog = dialogRef.current;
    if (!dialog) return;

    const focusableElements = dialog.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    ) as NodeListOf<HTMLElement>;

    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    const handleTabKey = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;

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

    dialog.addEventListener('keydown', handleTabKey);

    return () => {
      dialog.removeEventListener('keydown', handleTabKey);
    };
  }, [isOpen]);

  // Handle backdrop click
  const handleBackdropClick = (e: React.MouseEvent<HTMLDialogElement>) => {
    const dialog = dialogRef.current;
    if (dialog && e.target === dialog) {
      onClose();
    }
  };

  // Progressive enhancement: redirect to canonical pages
  const handleAuthAction = () => {
    onClose();
    window.location.href = mode === 'login' ? '/login' : '/signup';
  };

  if (!isOpen) return null;

  return (
    <dialog
      ref={dialogRef}
      className="modal fixed inset-0 z-50 m-0 h-full w-full max-h-none max-w-none p-0 bg-black/30 backdrop-blur-lg"
      aria-labelledby="auth-modal-title"
      aria-describedby="auth-modal-description"
      onClick={handleBackdropClick}
    >
      <div className="flex min-h-full items-center justify-center p-4">
        <div 
          className="relative w-full max-w-lg card-glass text-foreground rounded-2xl shadow-floating"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Modal Header */}
          <div className="flex items-center justify-between p-6 border-b border-border">
            <div className="flex items-center gap-3">
              <RepzLogo size="sm" />
              <div>
                <h2 id="auth-modal-title" className="text-lg font-semibold">
                  {mode === 'login' ? 'Welcome Back' : 'Join REPZ Today'}
                </h2>
                <p id="auth-modal-description" className="text-sm text-foreground">
                  {mode === 'login' 
                    ? 'Sign in to continue your fitness journey'
                    : 'Create your account and start your personalized fitness journey'
                  }
                </p>
              </div>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              className="h-8 w-8 rounded-full"
              aria-label="Close modal"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>

          {/* Modal Content */}
          <div className="p-6 text-center">
            <div className="space-y-4">
              <p className="text-foreground">
                Continue to our secure {mode === 'login' ? 'sign in' : 'sign up'} page
              </p>
              
              <Button 
                onClick={handleAuthAction}
                className="w-full"
                size="lg"
              >
                {mode === 'login' ? 'Continue to Sign In' : 'Continue to Sign Up'}
              </Button>
              
              <div className="text-xs text-foreground">
                You'll be redirected to our secure authentication page
              </div>
            </div>
          </div>

          {/* Modal Footer */}
          <div className="p-6 border-t border-border text-center">
            {mode === 'login' ? (
              <p className="text-sm text-foreground">
                New to REPZ?{' '}
                <button
                  onClick={() => window.location.href = '/signup'}
                  className="text-primary hover:text-primary/80 underline"
                >
                  Create an account
                </button>
              </p>
            ) : (
              <p className="text-sm text-foreground">
                Already have an account?{' '}
                <button
                  onClick={() => window.location.href = '/login'}
                  className="text-primary hover:text-primary/80 underline"
                >
                  Sign in
                </button>
              </p>
            )}
          </div>
        </div>
      </div>
    </dialog>
  );
}