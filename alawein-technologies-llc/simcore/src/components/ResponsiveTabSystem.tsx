import React from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useResponsive } from '@/hooks/use-responsive';

interface TabConfig {
  value: string;
  label: string;
  shortLabel?: string;
  icon: React.ComponentType<{ className?: string }>;
  content: React.ReactNode;
}

interface ResponsiveTabSystemProps {
  tabs: TabConfig[];
  defaultValue?: string;
  value?: string;
  onValueChange?: (value: string) => void;
  className?: string;
}

export const ResponsiveTabSystem: React.FC<ResponsiveTabSystemProps> = ({
  tabs,
  defaultValue,
  value,
  onValueChange,
  className = ""
}) => {
  const { isMobile, isTablet } = useResponsive();

  // Calculate responsive grid columns based on number of tabs
  const getGridCols = () => {
    const tabCount = tabs.length;
    if (isMobile) {
      return tabCount <= 2 ? `grid-cols-${tabCount}` : 'grid-cols-2';
    }
    if (isTablet) {
      return tabCount <= 3 ? `grid-cols-${tabCount}` : 'grid-cols-3';
    }
    return `grid-cols-${Math.min(tabCount, 5)}`;
  };

  return (
    <Tabs 
      value={value} 
      defaultValue={defaultValue}
      onValueChange={onValueChange} 
      className={`w-full ${className}`}
    >
      <TabsList className={`
        grid w-full ${getGridCols()} 
        mb-4 sm:mb-6 lg:mb-8 
        gap-1 sm:gap-0 
        min-h-[48px] sm:min-h-[52px]
        p-1
      `}>
        {tabs.map((tab) => (
          <TabsTrigger
            key={tab.value}
            value={tab.value}
            className="
              min-h-[44px] min-w-[44px] 
              flex flex-col sm:flex-row 
              items-center justify-center 
              gap-1 sm:gap-2 
              text-xs sm:text-sm 
              p-2 sm:p-3
              touch-target
              font-medium
              transition-all duration-200 ease-out
              hover:scale-[1.02] active:scale-[0.98]
              data-[state=active]:bg-background
              data-[state=active]:text-foreground
              data-[state=active]:shadow-sm
            "
          >
            <tab.icon className="w-4 h-4 flex-shrink-0" />
            <span className="hidden sm:inline font-medium">
              {tab.label}
            </span>
            <span className="inline sm:hidden text-[11px] font-medium leading-tight text-center">
              {tab.shortLabel || tab.label.slice(0, 4)}
            </span>
          </TabsTrigger>
        ))}
      </TabsList>

      {tabs.map((tab) => (
        <TabsContent 
          key={tab.value} 
          value={tab.value}
          className="space-y-4 sm:space-y-6"
        >
          {tab.content}
        </TabsContent>
      ))}
    </Tabs>
  );
};

// Utility function for creating tab configurations
export const createTabConfig = (
  value: string,
  label: string,
  icon: React.ComponentType<{ className?: string }>,
  content: React.ReactNode,
  shortLabel?: string
): TabConfig => ({
  value,
  label,
  shortLabel,
  icon,
  content
});

// Hook for responsive tab behavior
export const useResponsiveTabs = () => {
  const { isMobile, isTablet } = useResponsive();

  const getTabProps = (isActive: boolean) => ({
    className: `
      min-h-[44px] min-w-[44px]
      flex items-center justify-center
      ${isMobile ? 'flex-col gap-1 p-2' : 'flex-row gap-2 p-3'}
      text-xs sm:text-sm font-medium
      touch-target
      transition-all duration-200 ease-out
      ${isActive ? 'scale-105' : 'hover:scale-[1.02] active:scale-[0.98]'}
    `
  });

  const shouldShowFullLabel = !isMobile;
  const shouldUseIconOnly = isMobile && isTablet;

  return {
    getTabProps,
    shouldShowFullLabel,
    shouldUseIconOnly,
    isMobile,
    isTablet
  };
};