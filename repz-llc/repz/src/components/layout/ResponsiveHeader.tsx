import React, { ReactNode } from 'react';
import { BrandLogo } from '@/components/shared/BrandLogo';

interface ResponsiveHeaderProps {
  children?: ReactNode;
  className?: string;
  variant?: 'default' | 'dashboard' | 'auth';
  colorVariant?: 'default' | 'white' | 'dark';
  sticky?: boolean;
}

export const ResponsiveHeader: React.FC<ResponsiveHeaderProps> = ({
  children,
  className = '',
  variant = 'default',
  colorVariant = 'default',
  sticky = true
}) => {
  const getHeaderClasses = () => {
    const baseClasses = 'w-full border-b bg-background/95 backdrop-blur-sm';
    const stickyClasses = sticky ? 'sticky top-0 z-50' : '';
    
    switch (variant) {
      case 'dashboard':
        return `${baseClasses} ${stickyClasses} border-border`;
      case 'auth':
        return `${baseClasses} ${stickyClasses} bg-transparent border-transparent`;
      default:
        return `${baseClasses} ${stickyClasses} border-border`;
    }
  };

  const getLogoVariant = (): 'full' | 'icon-only' | 'text-only' => {
    // Always use full variant to show logo + text
    return 'full';
  };

  return (
    <header className={`${getHeaderClasses()} ${className}`}>
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand Section - Far Left */}
          <div className="flex items-center flex-shrink-0">
            <BrandLogo 
              variant={getLogoVariant()}
              size="md"
              colorVariant={colorVariant}
              className="min-w-fit"
            />
          </div>
          
          {/* Content Section - Right Side */}
          {children && (
            <div className="flex items-center space-x-4 ml-4">
              {children}
            </div>
          )}
        </div>
      </div>
    </header>
  );
};