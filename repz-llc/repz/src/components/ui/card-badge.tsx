import { cn } from "@/lib/utils";

export type BadgeType = 'MOST_POPULAR' | 'COACH_PICK' | 'PREMIUM' | 'LIMITED';

interface CardBadgeProps {
  type: BadgeType;
  position?: 'top-left' | 'top-right';
  className?: string;
}

const badgeStyles = {
  MOST_POPULAR: {
    bg: 'bg-gradient-to-r from-purple-500 to-purple-600',
    border: 'border-purple-400',
    text: 'text-white',
    glow: 'bg-purple-500/40'
  },
  COACH_PICK: {
    bg: 'bg-transparent',
    border: 'border-green-400',
    text: 'text-green-400',
    glow: 'bg-green-500/40'
  },
  PREMIUM: {
    bg: 'bg-gradient-to-r from-yellow-400 to-yellow-600',
    border: 'border-yellow-500',
    text: 'text-black',
    glow: 'bg-yellow-500/40'
  },
  LIMITED: {
    bg: 'bg-transparent border-2',
    border: 'border-blue-400',
    text: 'text-blue-300',
    glow: 'bg-blue-500/40'
  }
};

const badgeText = {
  MOST_POPULAR: 'Most Popular',
  COACH_PICK: "Coach's Pick",
  PREMIUM: 'Premium',
  LIMITED: 'Limited Spots'
};

export const CardBadge = ({ type, position = 'top-left', className }: CardBadgeProps) => {
  const styles = badgeStyles[type];
  const positionClasses = position === 'top-left' ? 'top-3 left-3' : 'top-3 right-3';
  
  return (
    <div className={cn(`absolute ${positionClasses} z-20`, className)}>
      <div className="relative">
        {/* Glow effect */}
        <div className={`absolute inset-0 ${styles.glow} rounded-full blur-sm opacity-50`}></div>
        {/* Badge */}
        <div className={cn(
          "relative px-3 py-1 text-xs font-bold rounded-full shadow-lg uppercase",
          styles.bg,
          styles.border,
          styles.text
        )}>
          {badgeText[type]}
        </div>
      </div>
    </div>
  );
};