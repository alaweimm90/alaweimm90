import React from 'react';
import { cn } from '@/lib/utils';

// Design System Utility Components for consistent styling

interface SemanticTextProps {
  variant?: 'primary' | 'secondary' | 'muted' | 'inverse' | 'accent';
  className?: string;
  children: React.ReactNode;
}

export const SemanticText: React.FC<SemanticTextProps & React.HTMLAttributes<HTMLSpanElement>> = ({
  variant = 'primary',
  className,
  children,
  ...props
}) => {
  const variantClasses = {
    primary: 'text-foreground',
    secondary: 'text-muted-foreground',
    muted: 'text-muted-foreground',
    inverse: 'text-primary-foreground',
    accent: 'text-primary'
  };

  return (
    <span 
      className={cn(variantClasses[variant], className)} 
      {...props}
    >
      {children}
    </span>
  );
};

interface SemanticContainerProps {
  variant?: 'primary' | 'secondary' | 'muted' | 'inverse' | 'accent';
  className?: string;
  children: React.ReactNode;
}

export const SemanticContainer: React.FC<SemanticContainerProps & React.HTMLAttributes<HTMLDivElement>> = ({
  variant = 'primary',
  className,
  children,
  ...props
}) => {
  const variantClasses = {
    primary: 'bg-background text-foreground',
    secondary: 'bg-muted text-muted-foreground',
    muted: 'bg-muted/50 text-muted-foreground',
    inverse: 'bg-primary text-primary-foreground',
    accent: 'bg-accent text-accent-foreground'
  };

  return (
    <div 
      className={cn(variantClasses[variant], className)} 
      {...props}
    >
      {children}
    </div>
  );
};

// Animation Wrapper Components
interface AnimatedElementProps {
  animation?: 'fade-in' | 'scale-in' | 'slide-up' | 'fade-in-up' | 'bounce-subtle' | 'float';
  delay?: number;
  className?: string;
  children: React.ReactNode;
}

export const AnimatedElement: React.FC<AnimatedElementProps & React.HTMLAttributes<HTMLDivElement>> = ({
  animation = 'fade-in',
  delay = 0,
  className,
  children,
  ...props
}) => {
  const animationClasses = {
    'fade-in': 'animate-fade-in',
    'scale-in': 'animate-scale-in',
    'slide-up': 'animate-fade-in-up',
    'fade-in-up': 'animate-fade-in-up',
    'bounce-subtle': 'animate-bounce-subtle',
    'float': 'animate-float'
  };

  return (
    <div 
      className={cn(animationClasses[animation], className)}
      style={{ animationDelay: `${delay}ms` }}
      {...props}
    >
      {children}
    </div>
  );
};

// Interactive Element with Hover Effects
interface InteractiveElementProps {
  effect?: 'lift' | 'glow' | 'scale' | 'button' | 'none';
  className?: string;
  children: React.ReactNode;
}

export const InteractiveElement: React.FC<InteractiveElementProps & React.HTMLAttributes<HTMLDivElement>> = ({
  effect = 'lift',
  className,
  children,
  ...props
}) => {
  const effectClasses = {
    lift: 'hover-lift',
    glow: 'hover-glow',
    scale: 'hover-scale',
    button: 'hover-button',
    none: ''
  };

  return (
    <div 
      className={cn(effectClasses[effect], className)} 
      {...props}
    >
      {children}
    </div>
  );
};

// Gradient Text Component
interface GradientTextProps {
  className?: string;
  children: React.ReactNode;
}

export const GradientText: React.FC<GradientTextProps> = ({ className, children }) => {
  return (
    <span className={cn('gradient-text', className)}>
      {children}
    </span>
  );
};

// Link with underline animation
interface AnimatedLinkProps {
  href?: string;
  className?: string;
  children: React.ReactNode;
}

export const AnimatedLink: React.FC<AnimatedLinkProps & React.AnchorHTMLAttributes<HTMLAnchorElement>> = ({
  className,
  children,
  ...props
}) => {
  return (
    <a 
      className={cn('link-underline', className)} 
      {...props}
    >
      {children}
    </a>
  );
};

// Professional Card with semantic styling
interface ProfessionalCardProps {
  variant?: 'default' | 'premium' | 'coaching';
  className?: string;
  children: React.ReactNode;
}

export const ProfessionalCard: React.FC<ProfessionalCardProps & React.HTMLAttributes<HTMLDivElement>> = ({
  variant = 'default',
  className,
  children,
  ...props
}) => {
  const variantClasses = {
    default: 'repz-card',
    premium: 'repz-card bg-gradient-elegant',
    coaching: 'repz-card coaching-gradient text-primary-foreground'
  };

  return (
    <div 
      className={cn(variantClasses[variant], className)} 
      {...props}
    >
      {children}
    </div>
  );
};

// Form Field with enhanced focus states
interface FormFieldProps {
  error?: boolean;
  success?: boolean;
  className?: string;
}

export const FormField: React.FC<FormFieldProps & React.InputHTMLAttributes<HTMLInputElement>> = ({
  error,
  success,
  className,
  ...props
}) => {
  const stateClasses = {
    default: 'form-input',
    error: 'form-input form-error',
    success: 'form-input form-success'
  };

  const currentState = error ? 'error' : success ? 'success' : 'default';

  return (
    <input 
      className={cn(stateClasses[currentState], className)} 
      {...props}
    />
  );
};

// Responsive Container
interface ResponsiveContainerProps {
  className?: string;
  children: React.ReactNode;
}

export const ResponsiveContainer: React.FC<ResponsiveContainerProps> = ({ className, children }) => {
  return (
    <div className={cn('container-repz', className)}>
      {children}
    </div>
  );
};

// Section with professional padding
interface ProfessionalSectionProps {
  className?: string;
  children: React.ReactNode;
}

export const ProfessionalSection: React.FC<ProfessionalSectionProps & React.HTMLAttributes<HTMLElement>> = ({ 
  className, 
  children, 
  ...props 
}) => {
  return (
    <section className={cn('section-padding', className)} {...props}>
      {children}
    </section>
  );
};

// Development mode indicators
export const BreakpointIndicator: React.FC = () => {
  if (process.env.NODE_ENV !== 'development') return null;
  
  return <div className="breakpoint-indicator" />;
};

// Touch-optimized button wrapper
interface TouchButtonProps {
  className?: string;
  children: React.ReactNode;
}

export const TouchButton: React.FC<TouchButtonProps & React.ButtonHTMLAttributes<HTMLButtonElement>> = ({
  className,
  children,
  ...props
}) => {
  return (
    <button 
      className={cn('touch-target', className)} 
      {...props}
     onClick={() => console.log("design-system button clicked")}>
      {children}
    </button>
  );
};

// Export all utilities
export {
  // Design System Colors (prevent direct color usage)
  type SemanticTextProps,
  type SemanticContainerProps,
  type AnimatedElementProps,
  type InteractiveElementProps,
  type GradientTextProps,
  type AnimatedLinkProps,
  type ProfessionalCardProps,
  type FormFieldProps,
  type ResponsiveContainerProps,
  type ProfessionalSectionProps,
  type TouchButtonProps
};