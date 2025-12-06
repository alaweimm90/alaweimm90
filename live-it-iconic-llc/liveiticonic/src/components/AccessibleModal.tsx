/**
 * AccessibleModal Component
 * WCAG 2.2 compliant modal/dialog wrapper with focus management
 * Implements focus trap, proper ARIA labels, and keyboard support
 */

import React, { useEffect, useRef, useCallback } from 'react';

interface AccessibleModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
  ariaDescribedBy?: string;
  returnFocusElement?: HTMLElement;
}

const AccessibleModal: React.FC<AccessibleModalProps> = ({
  isOpen,
  onClose,
  title,
  children,
  ariaDescribedBy,
  returnFocusElement,
}) => {
  const modalRef = useRef<HTMLDivElement>(null);
  const previousActiveElement = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!isOpen) return;

    // Store the element that had focus before modal opened
    previousActiveElement.current = document.activeElement as HTMLElement;

    // Find first focusable element in modal
    const focusableElements = modalRef.current?.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    if (focusableElements && focusableElements.length > 0) {
      (focusableElements[0] as HTMLElement).focus();
    }

    // Handle Escape key
    const handleEscapeKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleEscapeKey);

    // Handle focus trap (Tab/Shift+Tab)
    const handleTabKey = (e: KeyboardEvent) => {
      if (e.key !== 'Tab' || !modalRef.current) return;

      const focusableElements = modalRef.current.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      const firstElement = focusableElements[0] as HTMLElement;
      const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

      if (e.shiftKey) {
        // Shift+Tab on first element should focus last element
        if (document.activeElement === firstElement) {
          e.preventDefault();
          lastElement.focus();
        }
      } else {
        // Tab on last element should focus first element
        if (document.activeElement === lastElement) {
          e.preventDefault();
          firstElement.focus();
        }
      }
    };

    window.addEventListener('keydown', handleTabKey);

    return () => {
      window.removeEventListener('keydown', handleEscapeKey);
      window.removeEventListener('keydown', handleTabKey);

      // Return focus to the element that triggered the modal
      if (returnFocusElement) {
        returnFocusElement.focus();
      } else if (previousActiveElement.current) {
        previousActiveElement.current.focus();
      }
    };
  }, [isOpen, onClose, returnFocusElement]);

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 z-40 transition-opacity duration-300"
        onClick={onClose}
        role="presentation"
      />

      {/* Modal */}
      <div
        ref={modalRef}
        className="fixed inset-0 z-50 flex items-center justify-center p-4"
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
        aria-describedby={ariaDescribedBy}
      >
        <div className="bg-lii-ink rounded-xl max-w-2xl w-full shadow-2xl">
          <div className="p-6">
            <h2
              id="modal-title"
              className="text-2xl font-display font-semibold text-lii-cloud mb-4"
            >
              {title}
            </h2>
            <div className="text-lii-ash">{children}</div>
          </div>
        </div>
      </div>
    </>
  );
};

export default AccessibleModal;
