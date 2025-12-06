import React from 'react';
import { useResponsive } from '@/hooks/use-responsive';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ArrowLeft } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface ResponsiveSimulationLayoutProps {
  title: string;
  description: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced' | 'Research';
  onExport?: () => void;
  children: React.ReactNode;
}

const difficultyColors = {
  'Beginner': 'bg-green-500/10 text-green-400 border-green-500/20',
  'Intermediate': 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  'Advanced': 'bg-red-500/10 text-red-400 border-red-500/20',
  'Research': 'bg-purple-500/10 text-purple-400 border-purple-500/20',
};

export function ResponsiveSimulationLayout({
  title,
  description,
  difficulty,
  onExport,
  children
}: ResponsiveSimulationLayoutProps) {
  const navigate = useNavigate();
  const { isMobile, isTablet } = useResponsive();

  return (
    <div className="min-h-screen">
      <div className="absolute inset-0 bg-gradient-cosmic" />
      <div className="w-full mx-auto p-4 sm:p-6 container-responsive">
        {/* Responsive Header */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-6 sm:mb-8 gap-4">
          <div className="flex items-center gap-4 w-full sm:w-auto">
            <Button variant="ghost" size="sm" onClick={() => navigate('/')} className="touch-target">
              <ArrowLeft className="w-4 h-4 mr-2" />
              <span className="hidden sm:inline">Back to Modules</span>
              <span className="inline sm:hidden">Back</span>
            </Button>
            <div className="min-w-0 flex-1">
              <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold font-serif text-foreground leading-tight">
                {title}
              </h1>
              {description && (
                <p className="text-sm sm:text-base lg:text-lg text-muted-foreground mt-1 sm:mt-2 leading-relaxed">
                  {description}
                </p>
              )}
            </div>
          </div>
          
          <div className="flex items-center gap-3 w-full sm:w-auto justify-end">
            <Badge variant="outline" className={`${difficultyColors[difficulty]} touch-target`}>
              {difficulty}
            </Badge>
            {onExport && (
              <Button variant="outline" size="sm" onClick={onExport} className="touch-target">
                <span className="hidden sm:inline">Export</span>
                <span className="inline sm:hidden">Save</span>
              </Button>
            )}
          </div>
        </div>

        {/* Responsive Content Container */}
        <div className={`
          ${isMobile ? 'space-y-4' : isTablet ? 'space-y-6' : 'space-y-8'}
        `}>
          {children}
        </div>
      </div>
    </div>
  );
}

// Responsive Tab Configuration Helper
export function getResponsiveTabConfig(tabs: Array<{ value: string; label: string; icon: React.ComponentType<any> }>) {
  return {
    listClassName: `grid w-full grid-cols-${Math.min(tabs.length, 2)} sm:grid-cols-${Math.min(tabs.length, 3)} lg:grid-cols-${tabs.length} mb-4 sm:mb-8 gap-1 sm:gap-0 min-h-12`,
    getTriggerProps: (tab: { value: string; label: string; icon: React.ComponentType<any> }) => ({
      value: tab.value,
      className: "flex flex-col sm:flex-row items-center gap-1 sm:gap-2 text-xs sm:text-sm p-2",
      children: (
        <>
          <tab.icon className="w-4 h-4 flex-shrink-0" />
          <span className="hidden sm:inline">{tab.label}</span>
          <span className="inline sm:hidden text-[10px]">
            {tab.label.length > 6 ? tab.label.substring(0, 5) : tab.label}
          </span>
        </>
      )
    })
  };
}

// Responsive Grid Layout for Simulation Content
interface ResponsiveGridProps {
  children: React.ReactNode;
  columns?: { mobile: number; tablet: number; desktop: number };
  gap?: string;
  className?: string;
}

export function ResponsiveGrid({ 
  children, 
  columns = { mobile: 1, tablet: 2, desktop: 2 },
  gap = "gap-4 sm:gap-6",
  className = ""
}: ResponsiveGridProps) {
  const gridCols = `grid-cols-${columns.mobile} sm:grid-cols-${columns.tablet} lg:grid-cols-${columns.desktop}`;
  
  return (
    <div className={`grid ${gridCols} ${gap} ${className}`}>
      {children}
    </div>
  );
}

// Responsive Card Container
interface ResponsiveCardProps {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  className?: string;
  minHeight?: string;
}

export function ResponsiveCard({ 
  title, 
  subtitle, 
  children, 
  className = "",
  minHeight = "min-h-[400px] sm:min-h-[500px]"
}: ResponsiveCardProps) {
  return (
    <div className={`p-4 sm:p-6 border rounded-lg bg-card ${minHeight} ${className}`}>
      <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-3 sm:mb-4 gap-2">
        <h3 className="text-lg sm:text-xl font-semibold leading-tight">{title}</h3>
        {subtitle && (
          <div className="text-xs sm:text-sm text-muted-foreground">{subtitle}</div>
        )}
      </div>
      <div className="h-[300px] sm:h-[350px] lg:h-[420px]">
        {children}
      </div>
    </div>
  );
}