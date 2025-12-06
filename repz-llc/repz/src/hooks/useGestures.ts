import { useEffect, useRef, useState } from 'react';
import { useMobileCapacitor } from './useMobileCapacitor';

export interface GestureHandler {
  onSwipeLeft?: () => void;
  onSwipeRight?: () => void;
  onSwipeUp?: () => void;
  onSwipeDown?: () => void;
  onPinch?: (scale: number) => void;
  onLongPress?: () => void;
  onDoubleTap?: () => void;
}

export const useGestures = (handlers: GestureHandler) => {
  const { hapticFeedback, isNative } = useMobileCapacitor();
  const elementRef = useRef<HTMLDivElement>(null);
  const [isPressed, setIsPressed] = useState(false);
  
  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    let startX = 0;
    let startY = 0;
    let startTime = 0;
    let longPressTimeout: NodeJS.Timeout;
    let lastTapTime = 0;
    let initialDistance = 0;

    const handleTouchStart = (e: TouchEvent) => {
      setIsPressed(true);
      startTime = Date.now();
      
      if (e.touches.length === 1) {
        startX = e.touches[0].clientX;
        startY = e.touches[0].clientY;
        
        // Long press detection
        longPressTimeout = setTimeout(() => {
          if (handlers.onLongPress) {
            handlers.onLongPress();
            if (isNative) hapticFeedback.heavy();
          }
        }, 500);
      } else if (e.touches.length === 2) {
        // Pinch gesture setup
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        initialDistance = Math.sqrt(dx * dx + dy * dy);
        clearTimeout(longPressTimeout);
      }
    };

    const handleTouchMove = (e: TouchEvent) => {
      if (e.touches.length === 2 && initialDistance > 0 && handlers.onPinch) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        const currentDistance = Math.sqrt(dx * dx + dy * dy);
        const scale = currentDistance / initialDistance;
        handlers.onPinch(scale);
      }
      
      clearTimeout(longPressTimeout);
    };

    const handleTouchEnd = (e: TouchEvent) => {
      setIsPressed(false);
      clearTimeout(longPressTimeout);
      
      if (e.changedTouches.length !== 1) return;
      
      const endX = e.changedTouches[0].clientX;
      const endY = e.changedTouches[0].clientY;
      const deltaX = endX - startX;
      const deltaY = endY - startY;
      const deltaTime = Date.now() - startTime;
      
      // Double tap detection
      if (deltaTime < 300 && Math.abs(deltaX) < 10 && Math.abs(deltaY) < 10) {
        const timeSinceLastTap = Date.now() - lastTapTime;
        if (timeSinceLastTap < 300 && handlers.onDoubleTap) {
          handlers.onDoubleTap();
          if (isNative) hapticFeedback.medium();
          return;
        }
        lastTapTime = Date.now();
      }
      
      // Swipe detection
      if (deltaTime < 500) {
        const minSwipeDistance = 50;
        
        if (Math.abs(deltaX) > Math.abs(deltaY) && Math.abs(deltaX) > minSwipeDistance) {
          if (deltaX > 0 && handlers.onSwipeRight) {
            handlers.onSwipeRight();
            if (isNative) hapticFeedback.light();
          } else if (deltaX < 0 && handlers.onSwipeLeft) {
            handlers.onSwipeLeft();
            if (isNative) hapticFeedback.light();
          }
        } else if (Math.abs(deltaY) > minSwipeDistance) {
          if (deltaY > 0 && handlers.onSwipeDown) {
            handlers.onSwipeDown();
            if (isNative) hapticFeedback.light();
          } else if (deltaY < 0 && handlers.onSwipeUp) {
            handlers.onSwipeUp();
            if (isNative) hapticFeedback.light();
          }
        }
      }
    };

    element.addEventListener('touchstart', handleTouchStart, { passive: true });
    element.addEventListener('touchmove', handleTouchMove, { passive: true });
    element.addEventListener('touchend', handleTouchEnd, { passive: true });

    return () => {
      element.removeEventListener('touchstart', handleTouchStart);
      element.removeEventListener('touchmove', handleTouchMove);
      element.removeEventListener('touchend', handleTouchEnd);
      clearTimeout(longPressTimeout);
    };
  }, [handlers, hapticFeedback, isNative]);

  return { elementRef, isPressed };
};