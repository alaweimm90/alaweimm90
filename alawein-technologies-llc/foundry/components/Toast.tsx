/**
 * Toast Component
 * A notification toast system with multiple variants and positions
 */

import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { X, CheckCircle, AlertCircle, Info, AlertTriangle } from 'lucide-react';
import { cn } from '../utils/cn';

export type ToastVariant = 'default' | 'success' | 'error' | 'warning' | 'info';
export type ToastPosition = 'top-left' | 'top-center' | 'top-right' | 'bottom-left' | 'bottom-center' | 'bottom-right';

export interface ToastData {
  id: string;
  title?: string;
  message: string;
  variant?: ToastVariant;
  duration?: number;
  closable?: boolean;
  action?: {
    label: string;
    onClick: () => void;
  };
  onClose?: () => void;
}

interface ToastContextType {
  toasts: ToastData[];
  showToast: (toast: Omit<ToastData, 'id'>) => string;
  removeToast: (id: string) => void;
  clearToasts: () => void;
  position: ToastPosition;
  setPosition: (position: ToastPosition) => void;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

/**
 * Toast Provider
 * Wrap your app with this provider to enable toast notifications
 */
export interface ToastProviderProps {
  children: React.ReactNode;
  position?: ToastPosition;
  maxToasts?: number;
}

export const ToastProvider: React.FC<ToastProviderProps> = ({
  children,
  position: initialPosition = 'top-right',
  maxToasts = 5,
}) => {
  const [toasts, setToasts] = useState<ToastData[]>([]);
  const [position, setPosition] = useState<ToastPosition>(initialPosition);

  const showToast = useCallback((toast: Omit<ToastData, 'id'>) => {
    const id = Math.random().toString(36).substr(2, 9);
    const newToast: ToastData = {
      id,
      duration: 5000,
      closable: true,
      variant: 'default',
      ...toast,
    };

    setToasts((prevToasts) => {
      const updated = [...prevToasts, newToast];
      // Limit number of toasts
      if (updated.length > maxToasts) {
        return updated.slice(updated.length - maxToasts);
      }
      return updated;
    });

    return id;
  }, [maxToasts]);

  const removeToast = useCallback((id: string) => {
    setToasts((prevToasts) => {
      const toast = prevToasts.find(t => t.id === id);
      toast?.onClose?.();
      return prevToasts.filter(t => t.id !== id);
    });
  }, []);

  const clearToasts = useCallback(() => {
    toasts.forEach(toast => toast.onClose?.());
    setToasts([]);
  }, [toasts]);

  return (
    <ToastContext.Provider value={{ toasts, showToast, removeToast, clearToasts, position, setPosition }}>
      {children}
      <ToastContainer />
    </ToastContext.Provider>
  );
};

/**
 * useToast Hook
 * Use this hook to show toast notifications
 */
export const useToast = () => {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider');
  }

  return {
    showToast: context.showToast,
    removeToast: context.removeToast,
    clearToasts: context.clearToasts,
    setPosition: context.setPosition,
    // Convenience methods
    success: (message: string, options?: Partial<Omit<ToastData, 'id' | 'variant' | 'message'>>) =>
      context.showToast({ ...options, message, variant: 'success' }),
    error: (message: string, options?: Partial<Omit<ToastData, 'id' | 'variant' | 'message'>>) =>
      context.showToast({ ...options, message, variant: 'error' }),
    warning: (message: string, options?: Partial<Omit<ToastData, 'id' | 'variant' | 'message'>>) =>
      context.showToast({ ...options, message, variant: 'warning' }),
    info: (message: string, options?: Partial<Omit<ToastData, 'id' | 'variant' | 'message'>>) =>
      context.showToast({ ...options, message, variant: 'info' }),
  };
};

/**
 * Toast Container
 * Renders all active toasts
 */
const ToastContainer: React.FC = () => {
  const context = useContext(ToastContext);
  const [portalElement, setPortalElement] = useState<HTMLElement | null>(null);

  useEffect(() => {
    const element = document.getElementById('toast-root') || document.body;
    setPortalElement(element);
  }, []);

  if (!context || !portalElement) return null;

  const { toasts, position } = context;

  // Position styles
  const positionStyles = {
    'top-left': 'top-4 left-4',
    'top-center': 'top-4 left-1/2 -translate-x-1/2',
    'top-right': 'top-4 right-4',
    'bottom-left': 'bottom-4 left-4',
    'bottom-center': 'bottom-4 left-1/2 -translate-x-1/2',
    'bottom-right': 'bottom-4 right-4',
  };

  const isTop = position.startsWith('top');

  return createPortal(
    <div
      className={cn(
        'fixed z-[1080] pointer-events-none',
        positionStyles[position]
      )}
    >
      <AnimatePresence mode="sync">
        {toasts.map((toast, index) => (
          <ToastItem
            key={toast.id}
            toast={toast}
            index={index}
            isTop={isTop}
          />
        ))}
      </AnimatePresence>
    </div>,
    portalElement
  );
};

/**
 * Individual Toast Item
 */
interface ToastItemProps {
  toast: ToastData;
  index: number;
  isTop: boolean;
}

const ToastItem: React.FC<ToastItemProps> = ({ toast, index, isTop }) => {
  const { removeToast } = useContext(ToastContext)!;

  useEffect(() => {
    if (toast.duration && toast.duration > 0) {
      const timer = setTimeout(() => {
        removeToast(toast.id);
      }, toast.duration);

      return () => clearTimeout(timer);
    }
  }, [toast, removeToast]);

  // Variant styles
  const variantStyles = {
    default: 'bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100',
    success: 'bg-success-50 dark:bg-success-900/30 text-success-900 dark:text-success-100 border-success-200 dark:border-success-800',
    error: 'bg-danger-50 dark:bg-danger-900/30 text-danger-900 dark:text-danger-100 border-danger-200 dark:border-danger-800',
    warning: 'bg-warning-50 dark:bg-warning-900/30 text-warning-900 dark:text-warning-100 border-warning-200 dark:border-warning-800',
    info: 'bg-primary-50 dark:bg-primary-900/30 text-primary-900 dark:text-primary-100 border-primary-200 dark:border-primary-800',
  };

  // Icon components
  const icons = {
    default: null,
    success: <CheckCircle className="w-5 h-5 text-success-600 dark:text-success-400" />,
    error: <AlertCircle className="w-5 h-5 text-danger-600 dark:text-danger-400" />,
    warning: <AlertTriangle className="w-5 h-5 text-warning-600 dark:text-warning-400" />,
    info: <Info className="w-5 h-5 text-primary-600 dark:text-primary-400" />,
  };

  const variant = toast.variant || 'default';

  return (
    <motion.div
      initial={{ opacity: 0, y: isTop ? -20 : 20, scale: 0.9 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9, transition: { duration: 0.2 } }}
      transition={{ type: 'spring', stiffness: 300, damping: 25 }}
      style={{ marginTop: isTop && index > 0 ? 12 : 0, marginBottom: !isTop && index > 0 ? 12 : 0 }}
      className="pointer-events-auto"
    >
      <div
        className={cn(
          'min-w-[320px] max-w-[420px] p-4 rounded-lg shadow-lg border',
          'backdrop-blur-sm',
          variantStyles[variant]
        )}
      >
        <div className="flex items-start gap-3">
          {/* Icon */}
          {icons[variant] && (
            <div className="flex-shrink-0 mt-0.5">
              {icons[variant]}
            </div>
          )}

          {/* Content */}
          <div className="flex-1 min-w-0">
            {toast.title && (
              <h4 className="font-semibold mb-1">
                {toast.title}
              </h4>
            )}
            <p className="text-sm opacity-90">
              {toast.message}
            </p>
            {toast.action && (
              <button
                onClick={toast.action.onClick}
                className="mt-2 text-sm font-medium hover:underline"
              >
                {toast.action.label}
              </button>
            )}
          </div>

          {/* Close button */}
          {toast.closable && (
            <button
              onClick={() => removeToast(toast.id)}
              className="flex-shrink-0 p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Progress bar for auto-dismiss */}
        {toast.duration && toast.duration > 0 && (
          <motion.div
            initial={{ scaleX: 1 }}
            animate={{ scaleX: 0 }}
            transition={{ duration: toast.duration / 1000, ease: 'linear' }}
            className="absolute bottom-0 left-0 right-0 h-1 bg-current opacity-20 origin-left rounded-b-lg"
          />
        )}
      </div>
    </motion.div>
  );
};

/**
 * Static toast methods (requires ToastProvider)
 * Usage: toast.success('Operation successful!')
 */
export const toast = {
  success: (message: string, options?: Partial<Omit<ToastData, 'id' | 'variant' | 'message'>>) => {
    const event = new CustomEvent('show-toast', {
      detail: { ...options, message, variant: 'success' },
    });
    window.dispatchEvent(event);
  },
  error: (message: string, options?: Partial<Omit<ToastData, 'id' | 'variant' | 'message'>>) => {
    const event = new CustomEvent('show-toast', {
      detail: { ...options, message, variant: 'error' },
    });
    window.dispatchEvent(event);
  },
  warning: (message: string, options?: Partial<Omit<ToastData, 'id' | 'variant' | 'message'>>) => {
    const event = new CustomEvent('show-toast', {
      detail: { ...options, message, variant: 'warning' },
    });
    window.dispatchEvent(event);
  },
  info: (message: string, options?: Partial<Omit<ToastData, 'id' | 'variant' | 'message'>>) => {
    const event = new CustomEvent('show-toast', {
      detail: { ...options, message, variant: 'info' },
    });
    window.dispatchEvent(event);
  },
  show: (options: Omit<ToastData, 'id'>) => {
    const event = new CustomEvent('show-toast', { detail: options });
    window.dispatchEvent(event);
  },
};

// Global event listener for static methods
if (typeof window !== 'undefined') {
  window.addEventListener('show-toast', ((event: CustomEvent<Omit<ToastData, 'id'>>) => {
    const toastContainer = document.querySelector('[data-toast-container]');
    if (toastContainer) {
      // This would need to be handled by the ToastProvider
      console.warn('Static toast methods require ToastProvider. Use useToast hook instead.');
    }
  }) as EventListener);
}