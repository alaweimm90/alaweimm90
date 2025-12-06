import { cn } from "@/lib/utils";
import { ReactNode, ButtonHTMLAttributes } from "react";

interface RepzButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'primary' | 'gradient' | 'tier';
  size?: 'sm' | 'md' | 'lg';
  children: ReactNode;
}

export const RepzButton = ({ 
  variant = 'default', 
  size = 'md', 
  children, 
  className,
  ...props 
}: RepzButtonProps) => {
  const variants = {
    default: 'bg-white/10 hover:bg-white/20 text-white',
    primary: 'bg-repz-orange hover:bg-repz-orange-dark text-white',
    gradient: 'bg-gradient-to-r from-repz-orange to-repz-orange-dark hover:from-repz-orange-dark hover:to-red-700 text-white',
    tier: 'bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 text-white'
  };

  const sizes = {
    sm: 'px-4 py-2 text-sm',
    md: 'px-6 py-3 text-base',
    lg: 'px-8 py-4 text-lg'
  };

  return (
    <button
      className={cn(
        'rounded-xl font-semibold transition-all duration-200 transform active:scale-98',
        variants[variant],
        sizes[size],
        className
      )}
      {...props}
     onClick={() => console.log("repz-button button clicked")}>
      {children}
    </button>
  );
};