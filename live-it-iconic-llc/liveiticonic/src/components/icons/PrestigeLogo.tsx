import { cn } from '@/lib/utils';

interface PrestigeLogoProps {
  className?: string;
}

const PrestigeLogo = ({ className }: PrestigeLogoProps) => {
  return (
    <div className={cn('relative group', className)}>
      <svg
        width="36"
        height="36"
        viewBox="0 0 36 36"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="transition-all duration-600 group-hover:scale-105"
      >
        {/* Crown/Crest Shape - Royalty & Prestige */}
        <path
          d="M6 24 L10 12 L14 20 L18 8 L22 20 L26 12 L30 24 L6 24 Z"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.8"
          fill="url(#prestigeGradient)"
          strokeLinejoin="round"
          className="group-hover:stroke-[2.2] transition-all duration-400"
        />

        {/* Central Jewel - Excellence */}
        <circle
          cx="18"
          cy="16"
          r="3"
          fill="hsl(var(--lii-gold))"
          className="group-hover:scale-110 transition-transform duration-300"
        />
        <circle
          cx="18"
          cy="16"
          r="1.5"
          fill="hsl(var(--lii-champagne))"
          className="group-hover:scale-125 transition-transform duration-300"
        />

        {/* Side Gems - Supporting Excellence */}
        <circle cx="12" cy="18" r="1.5" fill="hsl(var(--lii-gold))" opacity="0.8" />
        <circle cx="24" cy="18" r="1.5" fill="hsl(var(--lii-gold))" opacity="0.8" />

        {/* Base Line - Foundation */}
        <rect
          x="8"
          y="24"
          width="20"
          height="2"
          fill="hsl(var(--lii-gold))"
          className="group-hover:scale-x-110 transition-transform duration-300"
        />

        <defs>
          <linearGradient id="prestigeGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="hsl(var(--lii-gold))" stopOpacity="0.3" />
            <stop offset="50%" stopColor="hsl(var(--lii-champagne))" stopOpacity="0.2" />
            <stop offset="100%" stopColor="hsl(var(--lii-gold))" stopOpacity="0.1" />
          </linearGradient>
        </defs>
      </svg>
    </div>
  );
};

export default PrestigeLogo;
