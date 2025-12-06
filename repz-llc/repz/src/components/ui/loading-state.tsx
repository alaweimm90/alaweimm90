import React from 'react';
import { motion } from 'framer-motion';
import { LoadingSpinner } from './loading-spinner';

interface LoadingStateProps {
  isLoading: boolean;
  children: React.ReactNode;
  loadingMessage?: string;
  variant?: 'overlay' | 'replace' | 'inline';
  spinnerVariant?: 'default' | 'branded' | 'minimal' | 'pulse';
  minHeight?: string;
}

const LoadingState: React.FC<LoadingStateProps> = ({
  isLoading,
  children,
  loadingMessage = 'Loading...',
  variant = 'replace',
  spinnerVariant = 'default',
  minHeight = '200px'
}) => {
  if (variant === 'overlay') {
    return (
      <div className="relative">
        {children}
        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center z-50 rounded-lg"
          >
            <LoadingSpinner
              message={loadingMessage}
              size="lg"
            />
          </motion.div>
        )}
      </div>
    );
  }

  if (variant === 'inline') {
    return (
      <>
        {isLoading && (
          <div className="py-4">
            <LoadingSpinner
              message={loadingMessage}
              size="sm"
            />
          </div>
        )}
        {!isLoading && children}
      </>
    );
  }

  // Replace variant (default)
  if (isLoading) {
    return (
      <div 
        className="flex items-center justify-center w-full"
        style={{ minHeight }}
      >
        <LoadingSpinner
          message={loadingMessage}
          size="lg"
        />
      </div>
    );
  }

  return <>{children}</>;
};

export default LoadingState;