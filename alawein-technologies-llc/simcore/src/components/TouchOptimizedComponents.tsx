import React, { useEffect, useRef } from 'react';
import { useResponsive } from '@/hooks/use-responsive';

interface TouchOptimizedButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  fullWidth?: boolean;
}

export function TouchOptimizedButton({
  children,
  variant = 'primary',
  size = 'md',
  fullWidth = false,
  className = '',
  ...props
}: TouchOptimizedButtonProps) {
  const { isMobile } = useResponsive();
  
  const baseClasses = 'touch-manipulation select-none transition-all duration-200 font-medium rounded-lg focus:outline-none focus:ring-2 focus:ring-offset-2';
  
  const variantClasses = {
    primary: 'bg-primary text-primary-foreground hover:bg-primary/90 focus:ring-primary',
    secondary: 'bg-secondary text-secondary-foreground hover:bg-secondary/80 focus:ring-secondary',
    ghost: 'hover:bg-accent hover:text-accent-foreground focus:ring-accent'
  };
  
  const sizeClasses = {
    sm: isMobile ? 'px-4 py-3 text-sm min-h-[44px]' : 'px-3 py-2 text-sm',
    md: isMobile ? 'px-6 py-3 text-base min-h-[44px]' : 'px-4 py-2 text-sm',
    lg: isMobile ? 'px-8 py-4 text-lg min-h-[48px]' : 'px-6 py-3 text-base'
  };
  
  const widthClass = fullWidth ? 'w-full' : '';
  
  return (
    <button
      className={`${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]} ${widthClass} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}

interface ResponsiveModalProps {
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
  title?: string;
  size?: 'sm' | 'md' | 'lg' | 'full';
}

export function ResponsiveModal({
  isOpen,
  onClose,
  children,
  title,
  size = 'md'
}: ResponsiveModalProps) {
  const { isMobile } = useResponsive();
  const modalRef = useRef<HTMLDivElement>(null);
  
  const sizeClasses = {
    sm: isMobile ? 'max-w-full h-full' : 'max-w-md',
    md: isMobile ? 'max-w-full h-full' : 'max-w-lg',
    lg: isMobile ? 'max-w-full h-full' : 'max-w-2xl',
    full: 'max-w-full h-full'
  };
  
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }
    
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isOpen]);
  
  if (!isOpen) return null;
  
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-background/50" onClick={onClose} />
      <div
        ref={modalRef}
        className={`
          relative bg-background rounded-lg shadow-2xl
          ${sizeClasses[size]}
          ${isMobile ? 'rounded-none' : 'mx-4'}
          max-h-[90vh] overflow-auto
        `}
      >
        {title && (
          <div className="flex items-center justify-between p-4 border-b">
            <h2 className="text-lg font-semibold">{title}</h2>
            <TouchOptimizedButton variant="ghost" size="sm" onClick={onClose}>
              ×
            </TouchOptimizedButton>
          </div>
        )}
        <div className={isMobile ? 'p-4' : 'p-6'}>
          {children}
        </div>
      </div>
    </div>
  );
}

interface SwipeableDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
  direction?: 'left' | 'right' | 'bottom';
}

export function SwipeableDrawer({
  isOpen,
  onClose,
  children,
  direction = 'bottom'
}: SwipeableDrawerProps) {
  const { isMobile } = useResponsive();
  const [dragOffset, setDragOffset] = React.useState(0);
  const startPosRef = useRef(0);
  
  const directionClasses = {
    left: 'left-0 top-0 h-full w-80',
    right: 'right-0 top-0 h-full w-80', 
    bottom: 'bottom-0 left-0 right-0 max-h-[90vh]'
  };
  
  const transformClasses = {
    left: isOpen ? 'translate-x-0' : '-translate-x-full',
    right: isOpen ? 'translate-x-0' : 'translate-x-full',
    bottom: isOpen ? 'translate-y-0' : 'translate-y-full'
  };
  
  if (!isMobile) {
    return (
      <ResponsiveModal isOpen={isOpen} onClose={onClose} size="lg">
        {children}
      </ResponsiveModal>
    );
  }
  
  return (
    <div className={`fixed inset-0 z-50 ${isOpen ? 'pointer-events-auto' : 'pointer-events-none'}`}>
      <div 
        className={`absolute inset-0 bg-background transition-opacity duration-300 ${isOpen ? 'opacity-50' : 'opacity-0'}`}
        onClick={onClose}
      />
      <div
        className={`
          absolute ${directionClasses[direction]} 
          bg-background shadow-2xl transition-transform duration-300
          ${transformClasses[direction]}
        `}
        style={{
          transform: direction === 'bottom' 
            ? `translateY(${isOpen ? dragOffset : 100}%)` 
            : undefined
        }}
      >
        {direction === 'bottom' && (
          <div className="h-1 w-12 bg-muted rounded-full mx-auto my-3" />
        )}
        <div className="p-4 overflow-auto max-h-full">
          {children}
        </div>
      </div>
    </div>
  );
}