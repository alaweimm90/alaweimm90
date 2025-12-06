import React, { useState, useRef, useEffect } from 'react';
import { useResponsiveEnhanced } from '@/hooks/use-responsive-enhanced';
import { cn } from '@/lib/utils';

interface GestureState {
  scale: number;
  translateX: number;
  translateY: number;
  rotation: number;
}

interface TouchGestureHandlerProps {
  children: React.ReactNode;
  enablePinch?: boolean;
  enablePan?: boolean;
  enableRotation?: boolean;
  minScale?: number;
  maxScale?: number;
  onGestureStart?: () => void;
  onGestureEnd?: (state: GestureState) => void;
  className?: string;
}

export function TouchGestureHandler({
  children,
  enablePinch = true,
  enablePan = true,
  enableRotation = false,
  minScale = 0.5,
  maxScale = 3,
  onGestureStart,
  onGestureEnd,
  className = ''
}: TouchGestureHandlerProps) {
  const { isTouch, isMobile } = useResponsiveEnhanced();
  const containerRef = useRef<HTMLDivElement>(null);
  
  const [gestureState, setGestureState] = useState<GestureState>({
    scale: 1,
    translateX: 0,
    translateY: 0,
    rotation: 0
  });
  
  const [isGesturing, setIsGesturing] = useState(false);
  const [lastTouchDistance, setLastTouchDistance] = useState(0);
  const [lastTouchAngle, setLastTouchAngle] = useState(0);
  const [lastTouchCenter, setLastTouchCenter] = useState({ x: 0, y: 0 });
  const [initialTouchCenter, setInitialTouchCenter] = useState({ x: 0, y: 0 });
  const [lastTapTime, setLastTapTime] = useState(0);

  // Calculate distance between two touches
  const getTouchDistance = (touches: React.TouchList) => {
    if (touches.length < 2) return 0;
    const dx = touches[0].clientX - touches[1].clientX;
    const dy = touches[0].clientY - touches[1].clientY;
    return Math.sqrt(dx * dx + dy * dy);
  };

  // Calculate angle between two touches
  const getTouchAngle = (touches: React.TouchList) => {
    if (touches.length < 2) return 0;
    const dx = touches[1].clientX - touches[0].clientX;
    const dy = touches[1].clientY - touches[0].clientY;
    return Math.atan2(dy, dx) * (180 / Math.PI);
  };

  // Calculate center point of touches
  const getTouchCenter = (touches: React.TouchList) => {
    if (touches.length === 0) return { x: 0, y: 0 };
    
    let sumX = 0, sumY = 0;
    for (let i = 0; i < touches.length; i++) {
      sumX += touches[i].clientX;
      sumY += touches[i].clientY;
    }
    
    return {
      x: sumX / touches.length,
      y: sumY / touches.length
    };
  };

  const handleTouchStart = (e: React.TouchEvent) => {
    if (!isTouch) return;
    
    e.preventDefault();
    const touches = e.touches;
    
    // Double tap to reset
    if (touches.length === 1) {
      const now = Date.now();
      if (now - lastTapTime < 300) {
        // Double tap detected - reset to default state
        setGestureState({
          scale: 1,
          translateX: 0,
          translateY: 0,
          rotation: 0
        });
        return;
      }
      setLastTapTime(now);
    }
    
    if (touches.length >= 1) {
      setIsGesturing(true);
      onGestureStart?.();
      
      const center = getTouchCenter(touches);
      setLastTouchCenter(center);
      setInitialTouchCenter(center);
      
      if (touches.length >= 2) {
        setLastTouchDistance(getTouchDistance(touches));
        if (enableRotation) {
          setLastTouchAngle(getTouchAngle(touches));
        }
      }
    }
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (!isTouch || !isGesturing) return;
    
    e.preventDefault();
    const touches = e.touches;
    
    if (touches.length >= 1) {
      const center = getTouchCenter(touches);
      
      // Pan gesture
      if (enablePan) {
        const deltaX = center.x - lastTouchCenter.x;
        const deltaY = center.y - lastTouchCenter.y;
        
        setGestureState(prev => ({
          ...prev,
          translateX: prev.translateX + deltaX,
          translateY: prev.translateY + deltaY
        }));
      }
      
      // Pinch gesture
      if (touches.length >= 2 && enablePinch) {
        const currentDistance = getTouchDistance(touches);
        if (lastTouchDistance > 0) {
          const scaleChange = currentDistance / lastTouchDistance;
          setGestureState(prev => ({
            ...prev,
            scale: Math.min(maxScale, Math.max(minScale, prev.scale * scaleChange))
          }));
        }
        setLastTouchDistance(currentDistance);
        
        // Rotation gesture
        if (enableRotation) {
          const currentAngle = getTouchAngle(touches);
          const angleDelta = currentAngle - lastTouchAngle;
          setGestureState(prev => ({
            ...prev,
            rotation: prev.rotation + angleDelta
          }));
          setLastTouchAngle(currentAngle);
        }
      }
      
      setLastTouchCenter(center);
    }
  };

  const handleTouchEnd = (e: React.TouchEvent) => {
    if (!isTouch) return;
    
    const touches = e.touches;
    
    if (touches.length === 0) {
      setIsGesturing(false);
      onGestureEnd?.(gestureState);
      setLastTouchDistance(0);
      setLastTouchAngle(0);
    } else if (touches.length === 1) {
      // Reset pinch/rotation tracking when going from 2+ touches to 1
      setLastTouchDistance(0);
      setLastTouchAngle(0);
      setLastTouchCenter(getTouchCenter(touches));
    }
  };

  // Transform style
  const transformStyle = {
    transform: `
      translate(${gestureState.translateX}px, ${gestureState.translateY}px)
      scale(${gestureState.scale})
      rotate(${gestureState.rotation}deg)
    `,
    transformOrigin: 'center center',
    transition: isGesturing ? 'none' : 'transform 0.2s ease-out'
  };

  return (
    <div
      ref={containerRef}
      className={cn(
        'relative overflow-hidden touch-none select-none',
        className
      )}
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
      style={{
        touchAction: 'none',
        userSelect: 'none',
        WebkitUserSelect: 'none'
      }}
    >
      <div style={transformStyle}>
        {children}
      </div>
      
      {/* Gesture indicators */}
      {isGesturing && isMobile && (
        <div className="absolute top-2 left-2 bg-background/80 text-foreground text-xs px-2 py-1 rounded backdrop-blur-sm border">
          Scale: {gestureState.scale.toFixed(1)}x
          {enableRotation && ` | Rotation: ${gestureState.rotation.toFixed(0)}°`}
        </div>
      )}
    </div>
  );
}

// Hook for easier gesture integration
export function useGestureEnhanced(options: Partial<TouchGestureHandlerProps> = {}) {
  const { isTouch, isMobile } = useResponsiveEnhanced();
  const [isActive, setIsActive] = useState(false);
  
  const gestureProps = {
    enablePinch: true,
    enablePan: true,
    enableRotation: false,
    minScale: 0.5,
    maxScale: 3,
    onGestureStart: () => setIsActive(true),
    onGestureEnd: () => setIsActive(false),
    ...options
  };
  
  return {
    gestureProps,
    isActive,
    isSupported: isTouch && isMobile
  };
}