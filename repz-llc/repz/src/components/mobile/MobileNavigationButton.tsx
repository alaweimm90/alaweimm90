import React from 'react';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { useMobileCapacitor } from '@/hooks/useMobileCapacitor';
import { ImpactStyle } from '@capacitor/haptics';
import { cn } from '@/lib/utils';

interface MobileNavigationButtonProps {
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
  badge?: string | number;
  variant?: 'default' | 'ghost' | 'outline';
  className?: string;
  hapticStyle?: ImpactStyle;
}

const MobileNavigationButton: React.FC<MobileNavigationButtonProps> = ({
  onClick,
  icon,
  label,
  badge,
  variant = 'ghost',
  className,
  hapticStyle = ImpactStyle.Light
}) => {
  const { hapticFeedback, isNative } = useMobileCapacitor();

  const handleClick = () => {
    if (isNative) {
      hapticFeedback.light();
    }
    onClick();
  };

  return (
    <Button
      variant={variant}
      onClick={handleClick}
      className={cn(
        'flex flex-col items-center justify-center h-16 w-full relative',
        'text-xs font-medium gap-1',
        className
      )}
    >
      <div className="relative">
        {icon}
        {badge && (
          <Badge 
            variant="destructive" 
            className="absolute -top-2 -right-2 h-5 w-5 rounded-full p-0 flex items-center justify-center text-xs"
          >
            {badge}
          </Badge>
        )}
      </div>
      <span className="truncate max-w-full">{label}</span>
    </Button>
  );
};

export default MobileNavigationButton;