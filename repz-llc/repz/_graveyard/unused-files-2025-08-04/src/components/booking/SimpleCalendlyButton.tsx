import React from 'react';
import { Button } from '@/components/ui/button';
import { Calendar, ExternalLink } from 'lucide-react';

interface SimpleCalendlyButtonProps {
  buttonText?: string;
  variant?: 'default' | 'primary' | 'secondary' | 'outline' | 'ghost';
  size?: 'sm' | 'default' | 'lg';
  className?: string;
  icon?: boolean;
}

export const SimpleCalendlyButton: React.FC<SimpleCalendlyButtonProps> = ({
  buttonText = 'Book a Session',
  variant = 'primary',
  size = 'default',
  className = '',
  icon = true,
}) => {
  const calendlyUrl = import.meta.env.VITE_CALENDLY_BASE_URL || 'https://calendly.com/repz';

  const handleClick = () => {
    window.open(calendlyUrl, '_blank');
  };

  return (
    <Button
      variant={variant}
      size={size}
      onClick={handleClick}
      className={`flex items-center gap-2 ${className}`}
    >
      {icon && <Calendar className="h-4 w-4" />}
      {buttonText}
      <ExternalLink className="h-3 w-3" />
    </Button>
  );
};