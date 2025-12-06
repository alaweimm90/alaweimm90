import React, { useState } from 'react';
import { Button, ButtonProps } from '@/ui/atoms/Button';
import { Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface LoadingButtonProps extends ButtonProps {
  loading?: boolean;
  loadingText?: string;
  onAsyncClick?: () => Promise<void>;
}

export const LoadingButton = React.forwardRef<HTMLButtonElement, LoadingButtonProps>(
  ({ loading = false, loadingText = "Loading...", onAsyncClick, onClick, children, disabled, className, ...props }, ref) => {
    const [isLoading, setIsLoading] = useState(false);
    
    const handleClick = async (event: React.MouseEvent<HTMLButtonElement>) => {
      if (onAsyncClick) {
        setIsLoading(true);
        try {
          await onAsyncClick();
        } finally {
          setIsLoading(false);
        }
      } else if (onClick) {
        onClick(event);
      }
    };

    const isDisabled = loading || isLoading || disabled;
    const showLoading = loading || isLoading;

    return (
      <Button
        ref={ref}
        onClick={handleClick}
        disabled={isDisabled}
        className={cn(
          "transition-all duration-300",
          showLoading && "cursor-not-allowed",
          className
        )}
        {...props}
      >
        {showLoading && (
          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
        )}
        {showLoading ? loadingText : children}
      </Button>
    );
  }
);

LoadingButton.displayName = "LoadingButton";