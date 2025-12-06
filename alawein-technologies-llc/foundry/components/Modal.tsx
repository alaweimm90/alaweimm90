/**
 * Modal Component
 * A flexible modal/dialog component with animations and accessibility
 */

import React, { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { X } from 'lucide-react';
import { cn } from '../utils/cn';
import Button from './Button';

export interface ModalProps {
  /**
   * Modal open state
   */
  isOpen: boolean;

  /**
   * Close handler
   */
  onClose: () => void;

  /**
   * Modal title
   */
  title?: string;

  /**
   * Modal description
   */
  description?: string;

  /**
   * Modal size
   */
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | 'full';

  /**
   * Center modal vertically
   */
  centered?: boolean;

  /**
   * Close on overlay click
   */
  closeOnOverlayClick?: boolean;

  /**
   * Close on Escape key
   */
  closeOnEsc?: boolean;

  /**
   * Show close button
   */
  showCloseButton?: boolean;

  /**
   * Overlay blur effect
   */
  overlayBlur?: boolean;

  /**
   * Animation type
   */
  animation?: 'fade' | 'slide' | 'scale' | 'none';

  /**
   * Footer content
   */
  footer?: React.ReactNode;

  /**
   * Additional modal class names
   */
  className?: string;

  /**
   * Additional overlay class names
   */
  overlayClassName?: string;

  /**
   * Children elements
   */
  children?: React.ReactNode;

  /**
   * Z-index
   */
  zIndex?: number;

  /**
   * Prevent body scroll when open
   */
  preventScroll?: boolean;
}

const Modal: React.FC<ModalProps> = ({
  isOpen,
  onClose,
  title,
  description,
  size = 'md',
  centered = true,
  closeOnOverlayClick = true,
  closeOnEsc = true,
  showCloseButton = true,
  overlayBlur = true,
  animation = 'scale',
  footer,
  className,
  overlayClassName,
  children,
  zIndex = 1050,
  preventScroll = true,
}) => {
  const modalRef = useRef<HTMLDivElement>(null);
  const [portalElement, setPortalElement] = useState<HTMLElement | null>(null);

  // Set up portal element
  useEffect(() => {
    const element = document.getElementById('modal-root') || document.body;
    setPortalElement(element);
  }, []);

  // Handle escape key
  useEffect(() => {
    if (!closeOnEsc || !isOpen) return;

    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    document.addEventListener('keydown', handleEsc);
    return () => document.removeEventListener('keydown', handleEsc);
  }, [closeOnEsc, isOpen, onClose]);

  // Prevent body scroll
  useEffect(() => {
    if (!preventScroll) return;

    if (isOpen) {
      const originalStyle = document.body.style.overflow;
      document.body.style.overflow = 'hidden';
      return () => {
        document.body.style.overflow = originalStyle;
      };
    }
  }, [isOpen, preventScroll]);

  // Focus trap
  useEffect(() => {
    if (!isOpen || !modalRef.current) return;

    const focusableElements = modalRef.current.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    const firstElement = focusableElements[0] as HTMLElement;
    const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

    firstElement?.focus();

    const handleTab = (e: KeyboardEvent) => {
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

    document.addEventListener('keydown', handleTab);
    return () => document.removeEventListener('keydown', handleTab);
  }, [isOpen]);

  // Size styles
  const sizes = {
    xs: 'max-w-xs',
    sm: 'max-w-sm',
    md: 'max-w-md',
    lg: 'max-w-lg',
    xl: 'max-w-xl',
    full: 'max-w-full mx-4',
  };

  // Animation variants
  const overlayVariants = {
    initial: { opacity: 0 },
    animate: { opacity: 1 },
    exit: { opacity: 0 },
  };

  const modalVariants = {
    fade: {
      initial: { opacity: 0 },
      animate: { opacity: 1 },
      exit: { opacity: 0 },
    },
    slide: {
      initial: { opacity: 0, y: 50 },
      animate: { opacity: 1, y: 0 },
      exit: { opacity: 0, y: 50 },
    },
    scale: {
      initial: { opacity: 0, scale: 0.9 },
      animate: { opacity: 1, scale: 1 },
      exit: { opacity: 0, scale: 0.9 },
    },
    none: {
      initial: {},
      animate: {},
      exit: {},
    },
  };

  // Overlay classes
  const overlayClasses = cn(
    'fixed inset-0 bg-gray-900/50 dark:bg-gray-950/70',
    overlayBlur && 'backdrop-blur-sm',
    overlayClassName
  );

  // Modal container classes
  const containerClasses = cn(
    'fixed inset-0 overflow-y-auto',
    centered ? 'flex items-center justify-center min-h-screen' : 'flex justify-center pt-20'
  );

  // Modal classes
  const modalClasses = cn(
    'relative bg-white dark:bg-gray-800 rounded-xl shadow-2xl',
    'w-full mx-4',
    sizes[size],
    className
  );

  if (!portalElement) return null;

  return createPortal(
    <AnimatePresence>
      {isOpen && (
        <div style={{ zIndex }}>
          {/* Overlay */}
          <motion.div
            className={overlayClasses}
            variants={overlayVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            transition={{ duration: 0.2 }}
            onClick={closeOnOverlayClick ? onClose : undefined}
            aria-hidden="true"
          />

          {/* Modal container */}
          <div className={containerClasses} style={{ zIndex: zIndex + 1 }}>
            <motion.div
              ref={modalRef}
              className={modalClasses}
              variants={modalVariants[animation]}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={{ duration: 0.2, type: 'spring', stiffness: 300, damping: 30 }}
              onClick={(e) => e.stopPropagation()}
              role="dialog"
              aria-modal="true"
              aria-labelledby={title ? 'modal-title' : undefined}
              aria-describedby={description ? 'modal-description' : undefined}
            >
              {/* Close button */}
              {showCloseButton && (
                <button
                  onClick={onClose}
                  className="absolute top-4 right-4 p-1 rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100 dark:text-gray-500 dark:hover:text-gray-300 dark:hover:bg-gray-700 transition-all"
                  aria-label="Close modal"
                >
                  <X className="w-5 h-5" />
                </button>
              )}

              {/* Header */}
              {(title || description) && (
                <div className="px-6 pt-6 pb-4">
                  {title && (
                    <h2
                      id="modal-title"
                      className="text-xl font-semibold text-gray-900 dark:text-gray-100"
                    >
                      {title}
                    </h2>
                  )}
                  {description && (
                    <p
                      id="modal-description"
                      className="mt-2 text-sm text-gray-500 dark:text-gray-400"
                    >
                      {description}
                    </p>
                  )}
                </div>
              )}

              {/* Body */}
              <div className={cn(
                'px-6',
                !title && !description && 'pt-6',
                !footer && 'pb-6'
              )}>
                {children}
              </div>

              {/* Footer */}
              {footer && (
                <div className="px-6 py-4 bg-gray-50 dark:bg-gray-900/50 rounded-b-xl border-t border-gray-200 dark:border-gray-700">
                  {footer}
                </div>
              )}
            </motion.div>
          </div>
        </div>
      )}
    </AnimatePresence>,
    portalElement
  );
};

Modal.displayName = 'Modal';

/**
 * Confirm Modal Component
 * A pre-configured modal for confirmation dialogs
 */
export interface ConfirmModalProps extends Omit<ModalProps, 'footer' | 'children'> {
  /**
   * Confirm button text
   */
  confirmText?: string;

  /**
   * Cancel button text
   */
  cancelText?: string;

  /**
   * Confirm button variant
   */
  confirmVariant?: 'primary' | 'danger' | 'success' | 'warning';

  /**
   * Message to display
   */
  message?: string;

  /**
   * On confirm handler
   */
  onConfirm?: () => void;

  /**
   * Loading state for confirm action
   */
  loading?: boolean;
}

export const ConfirmModal: React.FC<ConfirmModalProps> = ({
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  confirmVariant = 'primary',
  message,
  onConfirm,
  loading = false,
  ...modalProps
}) => {
  const handleConfirm = () => {
    onConfirm?.();
    if (!loading) {
      modalProps.onClose();
    }
  };

  return (
    <Modal
      {...modalProps}
      size={modalProps.size || 'sm'}
      footer={
        <div className="flex justify-end gap-3">
          <Button
            variant="ghost"
            onClick={modalProps.onClose}
            disabled={loading}
          >
            {cancelText}
          </Button>
          <Button
            variant={confirmVariant}
            onClick={handleConfirm}
            loading={loading}
            loadingText="Processing..."
          >
            {confirmText}
          </Button>
        </div>
      }
    >
      {message && (
        <p className="text-gray-600 dark:text-gray-300">{message}</p>
      )}
    </Modal>
  );
};

ConfirmModal.displayName = 'ConfirmModal';

export default Modal;