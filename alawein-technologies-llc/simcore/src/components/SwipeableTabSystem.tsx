import React, { useState, useRef, useEffect } from 'react';
import { useResponsiveEnhanced } from '@/hooks/use-responsive-enhanced';
import { cn } from '@/lib/utils';

interface SwipeableTabsProps {
  tabs: Array<{
    value: string;
    label: string;
    icon?: React.ComponentType<any>;
    content: React.ReactNode;
  }>;
  value?: string;
  onValueChange?: (value: string) => void;
  className?: string;
}

export function SwipeableTabSystem({
  tabs,
  value: controlledValue,
  onValueChange,
  className = ''
}: SwipeableTabsProps) {
  const { isMobile, isTouch } = useResponsiveEnhanced();
  const [activeIndex, setActiveIndex] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [startX, setStartX] = useState(0);
  const [currentX, setCurrentX] = useState(0);
  const [translateX, setTranslateX] = useState(0);
  const contentRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Sync with controlled value
  useEffect(() => {
    if (controlledValue) {
      const index = tabs.findIndex(tab => tab.value === controlledValue);
      if (index !== -1) setActiveIndex(index);
    }
  }, [controlledValue, tabs]);

  const handleTouchStart = (e: React.TouchEvent) => {
    if (!isTouch || !isMobile) return;
    setIsDragging(true);
    setStartX(e.touches[0].clientX);
    setCurrentX(e.touches[0].clientX);
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (!isDragging || !isTouch || !isMobile) return;
    e.preventDefault();
    const x = e.touches[0].clientX;
    setCurrentX(x);
    const diff = x - startX;
    setTranslateX(diff);
  };

  const handleTouchEnd = () => {
    if (!isDragging || !isTouch || !isMobile) return;
    setIsDragging(false);
    
    const diff = currentX - startX;
    const threshold = 100; // Minimum swipe distance
    
    if (Math.abs(diff) > threshold) {
      if (diff > 0 && activeIndex > 0) {
        // Swipe right - go to previous tab
        handleTabClick(activeIndex - 1);
      } else if (diff < 0 && activeIndex < tabs.length - 1) {
        // Swipe left - go to next tab
        handleTabClick(activeIndex + 1);
      }
    }
    
    setTranslateX(0);
    setStartX(0);
    setCurrentX(0);
  };

  const handleTabClick = (index: number) => {
    setActiveIndex(index);
    onValueChange?.(tabs[index].value);
  };

  return (
    <div className={cn('w-full', className)} ref={containerRef}>
      {/* Tab Navigation */}
      <div className="flex overflow-x-auto scrollbar-hide border-b border-border">
        <div className="flex min-w-full">
          {tabs.map((tab, index) => {
            const Icon = tab.icon;
            const isActive = index === activeIndex;
            
            return (
              <button
                key={tab.value}
                onClick={() => handleTabClick(index)}
                className={cn(
                  'flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium transition-all duration-200',
                  'border-b-2 whitespace-nowrap flex-1 min-w-fit',
                  // Touch optimization
                  isTouch && 'min-h-[48px] min-w-[48px]',
                  // Mobile specific styles
                  isMobile ? 'flex-col text-xs px-3 py-2' : 'flex-row',
                  isActive
                    ? 'border-primary text-primary bg-primary/5'
                    : 'border-transparent text-muted-foreground hover:text-foreground hover:border-muted'
                )}
                aria-selected={isActive}
                role="tab"
              >
                {Icon && <Icon className={cn('flex-shrink-0', isMobile ? 'w-5 h-5' : 'w-4 h-4')} />}
                <span className={cn(
                  'transition-all duration-200',
                  isMobile ? 'text-[11px] leading-tight mt-1' : 'text-sm'
                )}>
                  {isMobile && tab.label.length > 8 ? tab.label.substring(0, 6) + '...' : tab.label}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Tab Content with Swipe Support */}
      <div 
        className="relative overflow-hidden"
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
      >
        <div
          ref={contentRef}
          className="flex transition-transform duration-300 ease-out"
          style={{
            transform: `translateX(calc(-${activeIndex * 100}% + ${isDragging ? translateX : 0}px))`,
            width: `${tabs.length * 100}%`
          }}
        >
          {tabs.map((tab, index) => (
            <div
              key={tab.value}
              className="w-full flex-shrink-0 p-4"
              style={{ width: `${100 / tabs.length}%` }}
            >
              {tab.content}
            </div>
          ))}
        </div>
      </div>

      {/* Mobile Swipe Indicator */}
      {isMobile && tabs.length > 1 && (
        <div className="flex justify-center gap-2 mt-4">
          {tabs.map((_, index) => (
            <button
              key={index}
              onClick={() => handleTabClick(index)}
              className={cn(
                'w-2 h-2 rounded-full transition-all duration-200',
                index === activeIndex ? 'bg-primary' : 'bg-muted'
              )}
              aria-label={`Go to ${tabs[index].label}`}
            />
          ))}
        </div>
      )}

      <style dangerouslySetInnerHTML={{
        __html: `
          .scrollbar-hide::-webkit-scrollbar {
            display: none;
          }
          .scrollbar-hide {
            -ms-overflow-style: none;
            scrollbar-width: none;
          }
        `
      }} />
    </div>
  );
}

// Hook for gesture-enhanced interactions
export function useSwipeGesture(onSwipeLeft?: () => void, onSwipeRight?: () => void) {
  const { isMobile, isTouch } = useResponsiveEnhanced();
  const [touchStart, setTouchStart] = useState<number | null>(null);
  const [touchEnd, setTouchEnd] = useState<number | null>(null);

  const minSwipeDistance = 50;

  const onTouchStart = (e: TouchEvent) => {
    setTouchEnd(null);
    setTouchStart(e.targetTouches[0].clientX);
  };

  const onTouchMove = (e: TouchEvent) => {
    setTouchEnd(e.targetTouches[0].clientX);
  };

  const onTouchEnd = () => {
    if (!touchStart || !touchEnd) return;
    
    const distance = touchStart - touchEnd;
    const isLeftSwipe = distance > minSwipeDistance;
    const isRightSwipe = distance < -minSwipeDistance;

    if (isLeftSwipe) onSwipeLeft?.();
    if (isRightSwipe) onSwipeRight?.();
  };

  return {
    onTouchStart: isTouch && isMobile ? onTouchStart : undefined,
    onTouchMove: isTouch && isMobile ? onTouchMove : undefined,
    onTouchEnd: isTouch && isMobile ? onTouchEnd : undefined,
  };
}