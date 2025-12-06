import React from 'react';
import { motion } from 'framer-motion';
import { AlertCircle, CheckCircle, Info, AlertTriangle, X } from 'lucide-react';
import { Button } from '@/ui/atoms/Button';

interface ToastProps {
  id?: string;
  title?: string;
  message: string;
  variant?: 'success' | 'error' | 'warning' | 'info';
  duration?: number;
  onClose?: () => void;
  action?: {
    label: string;
    onClick: () => void;
  };
}

const Toast: React.FC<ToastProps> = ({
  title,
  message,
  variant = 'info',
  onClose,
  action
}) => {
  const variants = {
    success: {
      icon: CheckCircle,
      className: 'bg-green-500/10 border-green-500/20 text-green-600'
    },
    error: {
      icon: AlertCircle,
      className: 'bg-red-500/10 border-red-500/20 text-red-600'
    },
    warning: {
      icon: AlertTriangle,
      className: 'bg-yellow-500/10 border-yellow-500/20 text-yellow-600'
    },
    info: {
      icon: Info,
      className: 'bg-blue-500/10 border-blue-500/20 text-blue-600'
    }
  };

  const { icon: Icon, className } = variants[variant];

  return (
    <motion.div
      initial={{ opacity: 0, y: 50, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 20, scale: 0.95 }}
      className={`
        relative max-w-sm w-full bg-card/95 backdrop-blur-sm 
        border rounded-lg shadow-elegant p-4 pointer-events-auto
        ${className}
      `}
    >
      <div className="flex items-start">
        <Icon className="w-5 h-5 mt-0.5 mr-3 flex-shrink-0" />
        
        <div className="flex-1 min-w-0">
          {title && (
            <h4 className="text-sm font-semibold text-foreground mb-1">
              {title}
            </h4>
          )}
          <p className="text-sm text-muted-foreground">
            {message}
          </p>
          
          {action && (
            <div className="mt-3">
              <Button
                size="sm"
                variant="ghost"
                onClick={action.onClick}
                className="h-8 px-3 text-xs"
              >
                {action.label}
              </Button>
            </div>
          )}
        </div>
        
        {onClose && (
          <Button
            size="sm"
            variant="ghost"
            onClick={onClose}
            className="h-6 w-6 p-0 hover:bg-muted/50 ml-2"
          >
            <X className="w-4 h-4" />
          </Button>
        )}
      </div>
    </motion.div>
  );
};

export default Toast;