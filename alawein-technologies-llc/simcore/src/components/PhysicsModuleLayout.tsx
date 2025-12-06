import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { AutoBreadcrumbs } from './AutoBreadcrumbs';

interface PhysicsModuleLayoutProps {
  children: React.ReactNode;
  className?: string;
  background?: 'quantum' | 'statistical' | 'energy' | 'fields' | 'cosmic';
}

interface PhysicsContentCardProps {
  title: string;
  description?: string;
  children: React.ReactNode;
  className?: string;
  variant?: 'default' | 'glass' | 'elegant' | 'minimal';
  glow?: boolean;
}

interface PhysicsTagProps {
  children: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'success' | 'warning' | 'danger' | 'info';
  size?: 'sm' | 'md' | 'lg';
  glow?: boolean;
  animated?: boolean;
}

const backgroundPatterns = {
  quantum: 'bg-gradient-cosmic',
  statistical: 'bg-gradient-to-br from-background via-emerald-950/5 to-background',
  energy: 'bg-gradient-to-br from-background via-red-950/5 to-background',
  fields: 'bg-gradient-to-br from-background via-amber-950/5 to-background',
  cosmic: 'bg-gradient-cosmic'
};

const tagVariants = {
  primary: 'bg-primary/15 text-primary border-primary/30 shadow-primary/20',
  secondary: 'bg-muted/50 text-muted-foreground border-border/50',
  success: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30 shadow-emerald-500/20',
  warning: 'bg-amber-500/15 text-amber-400 border-amber-500/30 shadow-amber-500/20',
  danger: 'bg-red-500/15 text-red-400 border-red-500/30 shadow-red-500/20',
  info: 'bg-blue-500/15 text-blue-400 border-blue-500/30 shadow-blue-500/20'
};

const tagSizes = {
  sm: 'px-2 py-1 text-xs',
  md: 'px-3 py-1.5 text-sm',
  lg: 'px-4 py-2 text-base'
};

export const PhysicsModuleLayout: React.FC<PhysicsModuleLayoutProps> = ({
  children,
  className = '',
  background = 'cosmic'
}) => {
  return (
    <div className={cn(
      'min-h-screen relative overflow-hidden',
      backgroundPatterns[background],
      className
    )}>
      {/* Animated Particle Background */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {/* Floating particles */}
        <div className="absolute top-1/4 left-1/4 w-1 h-1 bg-primary/20 rounded-full animate-ping animation-delay-0" />
        <div className="absolute top-1/3 right-1/3 w-0.5 h-0.5 bg-primary/30 rounded-full animate-ping animation-delay-1000" />
        <div className="absolute bottom-1/4 left-1/3 w-1.5 h-1.5 bg-primary/10 rounded-full animate-pulse animation-delay-2000" />
        <div className="absolute top-2/3 right-1/4 w-0.5 h-0.5 bg-primary/25 rounded-full animate-ping animation-delay-3000" />
        
        {/* Quantum field lines */}
        <div className="absolute inset-0 opacity-[0.02]">
          <svg width="100%" height="100%" className="absolute inset-0">
            <defs>
              <pattern id="quantum-grid" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse">
                <path d="M 100 0 L 0 0 0 100" fill="none" stroke="currentColor" strokeWidth="1"/>
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#quantum-grid)" />
          </svg>
        </div>
      </div>
      
      <div className="relative z-10 container mx-auto px-[--spacing-content] py-[--spacing-lg]">
        <AutoBreadcrumbs className="mb-[--spacing-md]" />
        {children}
      </div>
      
      {/* Elegant Bottom Gradient */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-background/50 to-transparent pointer-events-none" />
    </div>
  );
};

export const PhysicsContentCard: React.FC<PhysicsContentCardProps> = ({
  title,
  description,
  children,
  className = '',
  variant = 'default',
  glow = false
}) => {
  const cardVariants = {
    default: 'border-border/50 bg-card/80 backdrop-blur-sm',
    glass: 'border-border/30 bg-background/10 backdrop-blur-xl',
    elegant: 'border-border/50 bg-gradient-to-br from-card/90 to-card/50 backdrop-blur-lg shadow-2xl',
    minimal: 'border-none bg-transparent'
  };

  return (
    <Card className={cn(
      'rounded-2xl transition-all duration-300 hover:shadow-lg',
      cardVariants[variant],
      glow && 'shadow-lg shadow-primary/10 hover:shadow-primary/20',
      className
    )}>
      {title && (
        <CardHeader className="pb-[--spacing-md]">
          <CardTitle className="text-xl font-semibold tracking-tight bg-gradient-to-r from-foreground to-foreground/80 bg-clip-text">
            {title}
          </CardTitle>
          {description && (
            <CardDescription className="text-muted-foreground/80 leading-relaxed">
              {description}
            </CardDescription>
          )}
        </CardHeader>
      )}
      <CardContent className={title ? 'pt-0' : ''}>
        {children}
      </CardContent>
    </Card>
  );
};

export const PhysicsTag: React.FC<PhysicsTagProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  glow = false,
  animated = false
}) => {
  return (
    <Badge 
      variant="outline" 
      className={cn(
        'border-2 font-medium rounded-full transition-all duration-300',
        tagVariants[variant],
        tagSizes[size],
        glow && 'shadow-lg',
        animated && 'hover:scale-105 hover:shadow-lg',
        'backdrop-blur-sm'
      )}
    >
      {children}
    </Badge>
  );
};

// Enhanced Tab System for Physics Modules
interface PhysicsTabSystemProps {
  tabs: Array<{
    id: string;
    label: string;
    icon?: React.ReactNode;
    content: React.ReactNode;
  }>;
  defaultTab?: string;
  className?: string;
}

export const PhysicsTabSystem: React.FC<PhysicsTabSystemProps> = ({
  tabs,
  defaultTab,
  className = ''
}) => {
  const [activeTab, setActiveTab] = React.useState(defaultTab || tabs[0]?.id);

  return (
    <div className={cn('space-y-[--spacing-lg]', className)}>
      {/* Elegant Tab Navigation */}
      <div className="flex items-center gap-[--spacing-xs] p-1 bg-background/50 backdrop-blur-sm border border-border/50 rounded-2xl overflow-x-auto">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={cn(
              'flex items-center gap-[--spacing-xs] px-[--spacing-md] py-[--spacing-sm] rounded-xl font-medium transition-all duration-300 whitespace-nowrap min-w-fit min-h-[--touch-target-min]',
              activeTab === tab.id
                ? 'bg-primary text-primary-foreground shadow-lg shadow-primary/25 scale-105'
                : 'text-muted-foreground hover:text-foreground hover:bg-background/50'
            )}
          >
            {tab.icon}
            <span>{tab.label}</span>
          </button>
        ))}
      </div>
      
      {/* Tab Content */}
      <div className="animate-fade-in">
        {tabs.find(tab => tab.id === activeTab)?.content}
      </div>
    </div>
  );
};