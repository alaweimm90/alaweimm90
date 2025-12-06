import React, { useState } from 'react';
import { Button, ButtonProps } from '@/ui/atoms/Button';
import { cn } from '@/lib/utils';

interface RippleButtonProps extends ButtonProps {
  rippleColor?: string;
}

export const RippleButton: React.FC<RippleButtonProps> = ({
  children,
  className,
  rippleColor = 'rgba(255, 255, 255, 0.3)',
  onClick,
  ...props
}) => {
  const [ripples, setRipples] = useState<Array<{ id: number; x: number; y: number }>>([]);

  const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    const button = event.currentTarget;
    const rect = button.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const newRipple = {
      id: Date.now(),
      x,
      y,
    };

    setRipples(prev => [...prev, newRipple]);

    // Remove ripple after animation
    setTimeout(() => {
      setRipples(prev => prev.filter(ripple => ripple.id !== newRipple.id));
    }, 600);

    if (onClick) {
      onClick(event);
    }
  };

  return (
    <Button
      className={cn('relative overflow-hidden', className)}
      onClick={handleClick}
      {...props}
    >
      {children}
      
      {/* Ripple effects */}
      {ripples.map((ripple) => (
        <span
          key={ripple.id}
          className="absolute rounded-full animate-ping pointer-events-none"
          style={{
            left: ripple.x - 10,
            top: ripple.y - 10,
            width: 20,
            height: 20,
            backgroundColor: rippleColor,
            animation: 'button-ripple 0.6s ease-out',
          }}
        />
      ))}
    </Button>
  );
};