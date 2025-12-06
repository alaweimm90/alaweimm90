import React from 'react';
import { cn } from '@/lib/utils';
import { useResponsiveEnhanced } from '@/hooks/use-responsive-enhanced';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Sparkles, Zap, Atom } from 'lucide-react';

// Gradient Background Component
export const GradientBackground = ({ 
  variant = 'subtle',
  className,
  children 
}: {
  variant?: 'subtle' | 'vibrant' | 'physics';
  className?: string;
  children?: React.ReactNode;
}) => {
  const gradients = {
    subtle: 'bg-gradient-to-br from-background via-background to-muted/20',
    vibrant: 'bg-gradient-to-br from-primary/5 via-background to-secondary/5',
    physics: 'bg-gradient-to-br from-blue-500/5 via-purple-500/5 to-cyan-500/5'
  };

  return (
    <div className={cn(gradients[variant], className)}>
      {children}
    </div>
  );
};

// Enhanced Button with Physics Theme
export const PhysicsButton = React.forwardRef<
  HTMLButtonElement,
  React.ButtonHTMLAttributes<HTMLButtonElement> & {
    variant?: 'default' | 'physics' | 'quantum' | 'simulation';
    size?: 'sm' | 'md' | 'lg';
    isLoading?: boolean;
  }
>(({ className, variant = 'default', size = 'md', isLoading, children, ...props }, ref) => {
  const { isMobile } = useResponsiveEnhanced();

  const variants = {
    default: 'bg-primary hover:bg-primary/90 text-primary-foreground',
    physics: 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-[hsl(var(--semantic-text-primary))]',
    quantum: 'bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-600 hover:to-teal-600 text-[hsl(var(--semantic-text-primary))]',
    simulation: 'bg-gradient-to-r from-emerald-500 to-blue-500 hover:from-emerald-600 hover:to-blue-600 text-[hsl(var(--semantic-text-primary))]'
  };

  const sizes = {
    sm: 'h-8 px-3 text-xs',
    md: 'h-10 px-4 py-2',
    lg: 'h-11 px-8'
  };

  return (
    <Button
      ref={ref}
      className={cn(
        'transition-all duration-200 font-medium',
        'hover:scale-105 active:scale-95',
        'shadow-lg hover:shadow-xl',
        variants[variant],
        sizes[size],
        isMobile && 'touch-manipulation',
        className
      )}
      disabled={isLoading}
      {...props}
    >
      {isLoading && (
        <div className="w-4 h-4 mr-2 border-2 border-current border-t-transparent rounded-full animate-spin" />
      )}
      {children}
    </Button>
  );
});
PhysicsButton.displayName = 'PhysicsButton';

// Enhanced Module Card with Physics Styling
export const EnhancedModuleCard = ({ 
  title, 
  description, 
  icon: Icon, 
  difficulty = 'intermediate',
  tags = [],
  isNew = false,
  onClick,
  className 
}: {
  title: string;
  description: string;
  icon?: React.ComponentType<any>;
  difficulty?: 'beginner' | 'intermediate' | 'advanced';
  tags?: string[];
  isNew?: boolean;
  onClick?: () => void;
  className?: string;
}) => {
  const { isMobile } = useResponsiveEnhanced();

  const difficultyColors = {
    beginner: 'bg-green-500/10 text-green-700 border-green-500/20',
    intermediate: 'bg-blue-500/10 text-blue-700 border-blue-500/20',
    advanced: 'bg-red-500/10 text-red-700 border-red-500/20'
  };

  return (
    <Card 
      className={cn(
        'group cursor-pointer transition-all duration-300',
        'hover:shadow-xl hover:scale-[1.02] hover:-translate-y-1',
        'border-border/50 hover:border-primary/30',
        'bg-gradient-to-br from-card to-card/80',
        isMobile && 'touch-manipulation active:scale-95',
        className
      )}
      onClick={onClick}
    >
      <CardContent className="p-6">
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            {Icon && (
              <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors">
                <Icon className="w-6 h-6 text-primary" />
              </div>
            )}
            <div>
              <h3 className="font-semibold text-foreground group-hover:text-primary transition-colors">
                {title}
              </h3>
              {isNew && (
                <Badge variant="secondary" className="mt-1 text-xs">
                  <Sparkles className="w-3 h-3 mr-1" />
                  New
                </Badge>
              )}
            </div>
          </div>
          <Badge 
            variant="outline" 
            className={difficultyColors[difficulty]}
          >
            {difficulty}
          </Badge>
        </div>

        <p className="text-muted-foreground text-sm mb-4 leading-relaxed">
          {description}
        </p>

        {tags.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {tags.map((tag, index) => (
              <Badge 
                key={index} 
                variant="secondary" 
                className="text-xs bg-muted/50 hover:bg-muted transition-colors"
              >
                {tag}
              </Badge>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

// Physics-themed Loading Spinner
export const PhysicsSpinner = ({ 
  size = 'md',
  variant = 'orbit' 
}: {
  size?: 'sm' | 'md' | 'lg';
  variant?: 'orbit' | 'atom' | 'wave';
}) => {
  const sizes = {
    sm: 'w-6 h-6',
    md: 'w-8 h-8',
    lg: 'w-12 h-12'
  };

  if (variant === 'atom') {
    return (
      <div className={cn('relative', sizes[size])}>
        <div className="absolute inset-0 rounded-full border-2 border-primary/20" />
        <div className="absolute inset-1 rounded-full border-2 border-primary/40 animate-spin" />
        <div className="absolute inset-2 rounded-full border-2 border-primary animate-ping" />
        <div className="absolute inset-0 flex items-center justify-center">
          <Atom className="w-3 h-3 text-primary" />
        </div>
      </div>
    );
  }

  if (variant === 'wave') {
    return (
      <div className={cn('flex gap-1', sizes[size])}>
        {[0, 1, 2].map(i => (
          <div
            key={i}
            className="w-1 bg-primary rounded-full animate-pulse"
            style={{
              height: '100%',
              animationDelay: `${i * 0.2}s`,
              animationDuration: '1s'
            }}
          />
        ))}
      </div>
    );
  }

  // Default orbit variant
  return (
    <div className={cn('relative', sizes[size])}>
      <div className="absolute inset-0 rounded-full border-2 border-muted" />
      <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-primary animate-spin" />
      <div className="absolute inset-0 flex items-center justify-center">
        <Zap className="w-3 h-3 text-primary" />
      </div>
    </div>
  );
};

// Smooth Page Transition Wrapper
export const PageTransition = ({ 
  children,
  className 
}: {
  children: React.ReactNode;
  className?: string;
}) => {
  return (
    <div className={cn(
      'animate-fade-in',
      'motion-reduce:animate-none',
      className
    )}>
      {children}
    </div>
  );
};