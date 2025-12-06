import React, { useRef, useEffect, useState } from 'react';
import { useResponsiveEnhanced } from '@/hooks/use-responsive-enhanced';
import { cn } from '@/lib/utils';
import { ChevronLeft, ChevronRight } from 'lucide-react';

interface ResponsiveTabsProps {
  tabs: Array<{
    value: string;
    label: string;
    icon?: React.ComponentType<any>;
    content: React.ReactNode;
  }>;
  activeTab: string;
  onTabChange: (value: string) => void;
  className?: string;
}

export function ResponsiveScrollableTabs({
  tabs,
  activeTab,
  onTabChange,
  className = ''
}: ResponsiveTabsProps) {
  const { isMobile, isTouch } = useResponsiveEnhanced();
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [showLeftArrow, setShowLeftArrow] = useState(false);
  const [showRightArrow, setShowRightArrow] = useState(false);

  // Check scroll position
  const checkScroll = () => {
    const container = scrollContainerRef.current;
    if (!container) return;

    setShowLeftArrow(container.scrollLeft > 0);
    setShowRightArrow(
      container.scrollLeft < container.scrollWidth - container.clientWidth
    );
  };

  useEffect(() => {
    checkScroll();
    const container = scrollContainerRef.current;
    if (container) {
      container.addEventListener('scroll', checkScroll, { passive: true });
      return () => container.removeEventListener('scroll', checkScroll);
    }
  }, [tabs]);

  // Scroll functions
  const scrollLeft = () => {
    scrollContainerRef.current?.scrollBy({ left: -200, behavior: 'smooth' });
  };

  const scrollRight = () => {
    scrollContainerRef.current?.scrollBy({ left: 200, behavior: 'smooth' });
  };

  // Auto-scroll active tab into view
  useEffect(() => {
    const container = scrollContainerRef.current;
    const activeButton = container?.querySelector(`[data-value="${activeTab}"]`) as HTMLElement;
    
    if (container && activeButton) {
      const containerRect = container.getBoundingClientRect();
      const buttonRect = activeButton.getBoundingClientRect();
      
      if (buttonRect.left < containerRect.left || buttonRect.right > containerRect.right) {
        activeButton.scrollIntoView({ behavior: 'smooth', inline: 'center' });
      }
    }
  }, [activeTab]);

  return (
    <div className={cn('relative', className)}>
      {/* Tab List */}
      <div className="relative flex items-center">
        {/* Left scroll arrow */}
        {(isMobile || isTouch) && showLeftArrow && (
          <button
            onClick={scrollLeft}
            className="absolute left-0 z-10 w-8 h-8 bg-background/80 backdrop-blur-sm border rounded-full flex items-center justify-center shadow-lg"
            aria-label="Scroll tabs left"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
        )}

        {/* Scrollable tab container */}
        <div
          ref={scrollContainerRef}
          className="flex overflow-x-auto scrollbar-hide gap-1 px-1 py-2"
          style={{
            scrollbarWidth: 'none',
            msOverflowStyle: 'none',
            WebkitOverflowScrolling: 'touch'
          }}
        >
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.value;
            
            return (
              <button
                key={tab.value}
                data-value={tab.value}
                onClick={() => onTabChange(tab.value)}
                className={cn(
                  'flex-shrink-0 flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200',
                  'whitespace-nowrap min-w-fit',
                  // Touch optimization
                  isTouch && 'min-h-[44px] min-w-[44px]',
                  isActive
                    ? 'bg-primary text-primary-foreground shadow-md'
                    : 'hover:bg-muted text-muted-foreground hover:text-foreground'
                )}
                aria-selected={isActive}
                role="tab"
              >
                {Icon && <Icon className="w-4 h-4 flex-shrink-0" />}
                <span className={cn(
                  isMobile && tabs.length > 4 ? 'hidden' : 'block'
                )}>
                  {tab.label}
                </span>
                {/* Show only icon on mobile for many tabs */}
                {isMobile && tabs.length > 4 && (
                  <span className="sr-only">{tab.label}</span>
                )}
              </button>
            );
          })}
        </div>

        {/* Right scroll arrow */}
        {(isMobile || isTouch) && showRightArrow && (
          <button
            onClick={scrollRight}
            className="absolute right-0 z-10 w-8 h-8 bg-background/80 backdrop-blur-sm border rounded-full flex items-center justify-center shadow-lg"
            aria-label="Scroll tabs right"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Tab Content */}
      <div className="mt-4">
        {tabs.find(tab => tab.value === activeTab)?.content}
      </div>

      <style dangerouslySetInnerHTML={{
        __html: `
          .scrollbar-hide::-webkit-scrollbar {
            display: none;
          }
        `
      }} />
    </div>
  );
}

// Responsive tab trigger for existing tab systems
interface ResponsiveTabTriggerProps {
  value: string;
  icon?: React.ComponentType<any>;
  label: string;
  isActive?: boolean;
  onClick?: () => void;
  className?: string;
}

export function ResponsiveTabTrigger({
  value,
  icon: Icon,
  label,
  isActive = false,
  onClick,
  className = ''
}: ResponsiveTabTriggerProps) {
  const { isMobile, isTouch } = useResponsiveEnhanced();

  return (
    <button
      data-value={value}
      onClick={onClick}
      className={cn(
        'flex items-center justify-center gap-2 px-3 py-2 text-sm font-medium transition-all duration-200',
        'rounded-lg whitespace-nowrap',
        // Touch optimization
        isTouch && 'min-h-[44px] min-w-[44px]',
        // Mobile specific styles
        isMobile ? 'flex-col text-xs px-2 py-3' : 'flex-row',
        isActive
          ? 'bg-primary text-primary-foreground shadow-md'
          : 'hover:bg-muted text-muted-foreground hover:text-foreground',
        className
      )}
      aria-selected={isActive}
      role="tab"
    >
      {Icon && <Icon className={cn('flex-shrink-0', isMobile ? 'w-5 h-5' : 'w-4 h-4')} />}
      <span className={cn(
        'transition-all duration-200',
        isMobile ? 'text-[10px] leading-none mt-1' : 'text-sm'
      )}>
        {isMobile && label.length > 8 ? label.substring(0, 6) + '...' : label}
      </span>
    </button>
  );
}