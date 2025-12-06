import { cn } from "@/lib/utils";
import { ReactNode } from "react";

interface RepzCardProps {
  variant?: 'default' | 'elevated' | 'tier';
  glow?: boolean;
  children: ReactNode;
  className?: string;
}

export const RepzCard = ({ 
  variant = 'default', 
  glow = false,
  children,
  className 
}: RepzCardProps) => {
  const variants = {
    default: 'bg-repz-surface border-white/10',
    elevated: 'bg-repz-surface-light border-white/15',
    tier: 'bg-repz-surface border-white/10 hover:border-white/20'
  };

  return (
    <div className={cn(
      'relative rounded-xl border transition-all duration-300',
      variants[variant],
      glow && 'shadow-lg shadow-current/20',
      className
    )}>
      {children}
    </div>
  );
};