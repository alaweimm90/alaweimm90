import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Capacitor } from '@capacitor/core';
import { StatusBar } from '@capacitor/status-bar';
import { Haptics, ImpactStyle } from '@capacitor/haptics';
import { cn } from '@/lib/utils';
import { useMobileDetection } from '@/hooks/useMobileDetection';
import { useAdvancedMobileCapabilities, NativeInteractions } from '@/hooks/useAdvancedMobileCapabilities';
import { EnhancedCard, EnhancedText, EnhancedBadge } from './enhanced-design-system';

// ============================================================================
// PHASE 3: ADVANCED MOBILE COMPONENTS
// Native-optimized UI components with gesture support
// ============================================================================

// Advanced Touch/Gesture Button
interface TouchOptimizedButtonProps {
  variant?: 'primary' | 'secondary' | 'ghost' | 'destructive';
  size?: 'sm' | 'md' | 'lg';
  hapticFeedback?: boolean;
  rippleEffect?: boolean;
  children: React.ReactNode;
  className?: string;
  onClick?: (e: React.MouseEvent<HTMLButtonElement>) => void;
  disabled?: boolean;
}

export const TouchOptimizedButton: React.FC<TouchOptimizedButtonProps> = ({
  variant = 'primary',
  size = 'md',
  hapticFeedback = true,
  rippleEffect = true,
  className,
  children,
  onClick,
  disabled,
  ...props
}) => {
  const [ripples, setRipples] = useState<Array<{ id: number; x: number; y: number }>>([]);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const { isMobile, isTouchDevice } = useMobileDetection();

  const handleClick = useCallback(async (e: React.MouseEvent<HTMLButtonElement>) => {
    if (disabled) return;

    // Haptic feedback for native apps
    if (hapticFeedback && isTouchDevice) {
      await NativeInteractions.hapticFeedback(ImpactStyle.Light);
    }

    // Create ripple effect
    if (rippleEffect && buttonRef.current) {
      const rect = buttonRef.current.getBoundingClientRect();
      const rippleX = e.clientX - rect.left;
      const rippleY = e.clientY - rect.top;
      
      const newRipple = { id: Date.now(), x: rippleX, y: rippleY };
      setRipples(prev => [...prev, newRipple]);
      
      // Remove ripple after animation
      setTimeout(() => {
        setRipples(prev => prev.filter(r => r.id !== newRipple.id));
      }, 600);
    }

    onClick?.(e);
  }, [disabled, hapticFeedback, isTouchDevice, rippleEffect, onClick]);

  const variants = {
    primary: 'bg-primary hover:bg-primary/90 text-primary-foreground',
    secondary: 'bg-secondary hover:bg-secondary/80 text-secondary-foreground',
    ghost: 'hover:bg-accent hover:text-accent-foreground',
    destructive: 'bg-destructive hover:bg-destructive/90 text-destructive-foreground'
  };

  const sizes = {
    sm: 'h-10 px-4 text-sm',
    md: 'h-12 px-6 text-base',
    lg: 'h-14 px-8 text-lg'
  };

  return (
    <motion.button
      ref={buttonRef}
      className={cn(
        'relative overflow-hidden rounded-xl font-medium transition-all duration-200',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
        'disabled:pointer-events-none disabled:opacity-50',
        // Larger touch targets for mobile
        isMobile && 'min-h-[48px] min-w-[48px]',
        variants[variant],
        sizes[size],
        className
      )}
      onClick={handleClick}
      disabled={disabled}
    >
      {children}
      
      {/* Ripple Effects */}
      <AnimatePresence>
        {ripples.map(ripple => (
          <motion.div
            key={ripple.id}
            className="absolute rounded-full bg-white/30"
            style={{
              left: ripple.x - 10,
              top: ripple.y - 10,
              width: 20,
              height: 20,
            }}
            initial={{ scale: 0, opacity: 1 }}
            animate={{ scale: 4, opacity: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.6, ease: "easeOut" }}
          />
        ))}
      </AnimatePresence>
    </motion.button>
  );
};

// Swipeable Card Component
interface SwipeableCardProps {
  children: React.ReactNode;
  onSwipeLeft?: () => void;
  onSwipeRight?: () => void;
  onSwipeUp?: () => void;
  onSwipeDown?: () => void;
  swipeThreshold?: number;
  className?: string;
}

export const SwipeableCard: React.FC<SwipeableCardProps> = ({
  children,
  onSwipeLeft,
  onSwipeRight,
  onSwipeUp,
  onSwipeDown,
  swipeThreshold = 100,
  className
}) => {
  const [isDragging, setIsDragging] = useState(false);
  
  return (
    <motion.div
      className={cn("relative", className)}
      drag
      dragConstraints={{ left: 0, right: 0, top: 0, bottom: 0 }}
      dragElastic={{ left: 0.2, right: 0.2, top: 0.2, bottom: 0.2 }}
      onDragStart={() => setIsDragging(true)}
      onDragEnd={(event, info) => {
        setIsDragging(false);
        
        const { offset, velocity } = info;
        const swipe = Math.abs(offset.x) > Math.abs(offset.y) ? 
          { direction: offset.x > 0 ? 'right' : 'left', value: Math.abs(offset.x) } :
          { direction: offset.y > 0 ? 'down' : 'up', value: Math.abs(offset.y) };
          
        if (swipe.value > swipeThreshold || Math.abs(velocity.x) > 500 || Math.abs(velocity.y) > 500) {
          switch (swipe.direction) {
            case 'left': onSwipeLeft?.(); break;
            case 'right': onSwipeRight?.(); break;
            case 'up': onSwipeUp?.(); break;
            case 'down': onSwipeDown?.(); break;
          }
        }
      }}
      whileDrag={{ scale: 0.95, rotateZ: isDragging ? 2 : 0 }}
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
    >
      {children}
    </motion.div>
  );
};

// Pull-to-Refresh Component
interface PullToRefreshProps {
  onRefresh: () => Promise<void>;
  children: React.ReactNode;
  disabled?: boolean;
  className?: string;
}

export const PullToRefresh: React.FC<PullToRefreshProps> = ({
  onRefresh,
  children,
  disabled = false,
  className
}) => {
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [pullDistance, setPullDistance] = useState(0);
  const [startY, setStartY] = useState(0);
  
  const handleTouchStart = (e: React.TouchEvent) => {
    if (disabled || window.scrollY > 0) return;
    setStartY(e.touches[0].clientY);
  };
  
  const handleTouchMove = (e: React.TouchEvent) => {
    if (disabled || window.scrollY > 0 || startY === 0) return;
    
    const currentY = e.touches[0].clientY;
    const distance = Math.max(0, currentY - startY);
    setPullDistance(Math.min(distance, 150));
  };
  
  const handleTouchEnd = async () => {
    if (disabled || pullDistance < 80) {
      setPullDistance(0);
      setStartY(0);
      return;
    }
    
    setIsRefreshing(true);
    await NativeInteractions.hapticFeedback(ImpactStyle.Medium);
    
    try {
      await onRefresh();
    } finally {
      setIsRefreshing(false);
      setPullDistance(0);
      setStartY(0);
    }
  };
  
  return (
    <div 
      className={cn("relative overflow-hidden", className)}
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
    >
      {/* Pull Indicator */}
      <motion.div
        className="absolute top-0 left-0 right-0 flex items-center justify-center z-10"
        style={{ height: pullDistance }}
        initial={{ opacity: 0 }}
        animate={{ opacity: pullDistance > 20 ? 1 : 0 }}
      >
        <div className="bg-background/90 backdrop-blur-sm rounded-full p-3">
          {isRefreshing ? (
            <motion.div
              className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full"
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            />
          ) : (
            <motion.div
              className="w-6 h-6 text-primary"
              animate={{ rotate: pullDistance > 80 ? 180 : 0 }}
            >
              ↓
            </motion.div>
          )}
        </div>
      </motion.div>
      
      {/* Content */}
      <motion.div
        style={{ 
          transform: isRefreshing ? 'translateY(60px)' : `translateY(${pullDistance * 0.4}px)`
        }}
        transition={{ type: "spring", stiffness: 300, damping: 30 }}
      >
        {children}
      </motion.div>
    </div>
  );
};

// Mobile-Optimized Modal/Sheet
interface MobileSheetProps {
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
  title?: string;
  snapPoints?: number[];
  className?: string;
}

export const MobileSheet: React.FC<MobileSheetProps> = ({
  isOpen,
  onClose,
  children,
  title,
  snapPoints = [0.3, 0.6, 0.9],
  className
}) => {
  const [currentSnap, setCurrentSnap] = useState(snapPoints[1]);
  const { isMobile } = useMobileDetection();
  
  useEffect(() => {
    if (isOpen && Capacitor.isNativePlatform()) {
      NativeInteractions.setStatusBarStyle('light');
    }
    
    return () => {
      if (Capacitor.isNativePlatform()) {
        NativeInteractions.setStatusBarStyle('dark');
      }
    };
  }, [isOpen]);
  
  if (!isMobile) {
    // Desktop fallback - regular modal
    return (
      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          >
            <motion.div
              className={cn("bg-background rounded-lg p-6 max-w-md w-full", className)}
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.8, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
            >
              {title && (
                <h2 className="text-lg font-semibold mb-4">{title}</h2>
              )}
              {children}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    );
  }
  
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            className="fixed inset-0 z-40 bg-black/50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />
          
          {/* Sheet */}
          <motion.div
            className={cn(
              "fixed bottom-0 left-0 right-0 z-50 bg-background rounded-t-2xl",
              "shadow-2xl border-t border-border",
              className
            )}
            initial={{ y: "100%" }}
            animate={{ y: `${(1 - currentSnap) * 100}%` }}
            exit={{ y: "100%" }}
            transition={{ type: "spring", damping: 30, stiffness: 300 }}
            drag="y"
            dragConstraints={{ top: 0, bottom: 0 }}
            dragElastic={{ top: 0, bottom: 0.2 }}
            onDragEnd={(event, info) => {
              const { offset, velocity } = info;
              
              if (velocity.y > 500 || offset.y > 200) {
                onClose();
                return;
              }
              
              // Snap to closest point
              const targetHeight = window.innerHeight * currentSnap;
              const currentHeight = targetHeight - offset.y;
              const currentRatio = currentHeight / window.innerHeight;
              
              const closestSnap = snapPoints.reduce((prev, current) => 
                Math.abs(current - currentRatio) < Math.abs(prev - currentRatio) ? current : prev
              );
              
              setCurrentSnap(closestSnap);
            }}
          >
            {/* Drag Handle */}
            <div className="flex justify-center py-3">
              <div className="w-10 h-1 bg-muted rounded-full" />
            </div>
            
            {/* Header */}
            {title && (
              <div className="px-6 pb-4 border-b border-border">
                <h2 className="text-lg font-semibold">{title}</h2>
              </div>
            )}
            
            {/* Content */}
            <div className="p-6 pb-safe">
              {children}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

// Device Info Display (for debugging/admin)
export const DeviceInfoCard: React.FC<{ className?: string }> = ({ className }) => {
  const { capabilities, isLoading } = useAdvancedMobileCapabilities();
  const mobileDetection = useMobileDetection();
  
  if (isLoading || !capabilities) {
    return (
      <EnhancedCard className={cn("p-4", className)}>
        <div className="animate-pulse">
          <div className="h-4 bg-muted rounded w-3/4 mb-2" />
          <div className="h-4 bg-muted rounded w-1/2" />
        </div>
      </EnhancedCard>
    );
  }
  
  return (
    <EnhancedCard className={cn("p-4 space-y-3", className)}>
      <div className="flex items-center justify-between">
        <EnhancedText variant="premium" weight="semibold">
          Device Information
        </EnhancedText>
        <EnhancedBadge variant="status" status={capabilities.networkStatus.connected ? 'active' : 'inactive'}>
          {capabilities.networkStatus.connected ? 'Online' : 'Offline'}
        </EnhancedBadge>
      </div>
      
      <div className="grid grid-cols-2 gap-2 text-sm">
        <div>
          <EnhancedText variant="muted" size="sm">Platform</EnhancedText>
          <EnhancedText variant="elegant" size="sm">{capabilities.deviceInfo.platform}</EnhancedText>
        </div>
        
        <div>
          <EnhancedText variant="muted" size="sm">Screen</EnhancedText>
          <EnhancedText variant="elegant" size="sm">
            {capabilities.screen.width}x{capabilities.screen.height}
          </EnhancedText>
        </div>
        
        <div>
          <EnhancedText variant="muted" size="sm">Orientation</EnhancedText>
          <EnhancedText variant="elegant" size="sm">{capabilities.screen.orientation}</EnhancedText>
        </div>
        
        <div>
          <EnhancedText variant="muted" size="sm">Connection</EnhancedText>
          <EnhancedText variant="elegant" size="sm">{capabilities.networkStatus.connectionType}</EnhancedText>
        </div>
      </div>
      
      <div className="flex flex-wrap gap-1">
        {Object.entries(capabilities.capabilities).map(([key, value]) => (
          <EnhancedBadge 
            key={key}
            variant="feature"
            className={value ? 'bg-green-500/20 text-green-400' : 'bg-gray-500/20 text-gray-400'}
          >
            {key.replace('has', '').toLowerCase()}
          </EnhancedBadge>
        ))}
      </div>
    </EnhancedCard>
  );
};