import React from 'react';

// Comprehensive responsive testing component for cross-device verification
export const ResponsiveTestIndicator: React.FC = () => {
  return (
    <div className="fixed top-0 right-0 z-50 bg-black text-white text-xs p-2 rounded-bl-lg opacity-75">
      <div className="block sm:hidden">📱 Mobile (320px-767px)</div>
      <div className="hidden sm:block lg:hidden">📟 Tablet (768px-1199px)</div>
      <div className="hidden lg:block">🖥️ Desktop (1200px+)</div>
    </div>
  );
};

// Touch target size validator for mobile compatibility
export const TouchTargetValidator: React.FC<{ children: React.ReactNode; className?: string }> = ({ 
  children, 
  className = "" 
}) => {
  return (
    <div className={`min-h-[44px] min-w-[44px] flex items-center justify-center ${className}`}>
      {children}
    </div>
  );
};

// Responsive container with proper breakpoint handling
export const ResponsiveContainer: React.FC<{ 
  children: React.ReactNode;
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl' | '2xl';
  padding?: 'none' | 'sm' | 'md' | 'lg';
}> = ({ 
  children, 
  maxWidth = 'xl',
  padding = 'md' 
}) => {
  const maxWidthClasses = {
    sm: 'max-w-sm',
    md: 'max-w-md', 
    lg: 'max-w-lg',
    xl: 'max-w-xl',
    '2xl': 'max-w-2xl'
  };

  const paddingClasses = {
    none: '',
    sm: 'px-4 sm:px-6',
    md: 'px-4 sm:px-6 lg:px-8',
    lg: 'px-6 sm:px-8 lg:px-12'
  };

  return (
    <div className={`mx-auto ${maxWidthClasses[maxWidth]} ${paddingClasses[padding]}`}>
      {children}
    </div>
  );
};

// Grid system with proper responsive behavior
export const ResponsiveGrid: React.FC<{
  children: React.ReactNode;
  cols?: {
    mobile?: number;
    tablet?: number;
    desktop?: number;
  };
  gap?: 'sm' | 'md' | 'lg';
}> = ({ 
  children,
  cols = { mobile: 1, tablet: 2, desktop: 4 },
  gap = 'md'
}) => {
  const gapClasses = {
    sm: 'gap-4',
    md: 'gap-6', 
    lg: 'gap-8'
  };

  const gridCols = `grid-cols-${cols.mobile} sm:grid-cols-${cols.tablet} lg:grid-cols-${cols.desktop}`;

  return (
    <div className={`grid ${gridCols} ${gapClasses[gap]} w-full`}>
      {children}
    </div>
  );
};

// Typography with responsive scaling
export const ResponsiveText: React.FC<{
  children: React.ReactNode;
  size?: 'xs' | 'sm' | 'base' | 'lg' | 'xl' | '2xl' | '3xl';
  weight?: 'normal' | 'medium' | 'semibold' | 'bold';
  className?: string;
}> = ({ 
  children,
  size = 'base',
  weight = 'normal',
  className = ''
}) => {
  const sizeClasses = {
    xs: 'text-xs sm:text-sm',
    sm: 'text-sm sm:text-base',
    base: 'text-base sm:text-lg',
    lg: 'text-lg sm:text-xl',
    xl: 'text-xl sm:text-2xl',
    '2xl': 'text-2xl sm:text-3xl',
    '3xl': 'text-3xl sm:text-4xl lg:text-5xl'
  };

  const weightClasses = {
    normal: 'font-normal',
    medium: 'font-medium',
    semibold: 'font-semibold', 
    bold: 'font-bold'
  };

  return (
    <div className={`${sizeClasses[size]} ${weightClasses[weight]} ${className}`}>
      {children}
    </div>
  );
};