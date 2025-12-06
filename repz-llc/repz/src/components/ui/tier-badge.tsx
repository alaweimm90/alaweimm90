import { cn } from "@/lib/utils";
import { ReactNode } from "react";

interface TierBadgeProps {
  type: 'popular' | 'limited' | 'new' | 'sale';
  children: ReactNode;
  className?: string;
}

export const TierBadge = ({ type, children, className }: TierBadgeProps) => {
  const styles = {
    popular: 'bg-tier-performance text-white',
    limited: 'bg-gradient-to-r from-tier-longevity to-yellow-400 text-black',
    new: 'bg-tier-adaptive text-white',
    sale: 'bg-red-500 text-white'
  };

  return (
    <div
      className={cn(
        'absolute -top-3 right-4 px-4 py-1.5 rounded-full',
        'text-xs font-bold uppercase tracking-wide',
        'animate-scale-in',
        styles[type],
        className
      )}
    >
      {children}
    </div>
  );
};