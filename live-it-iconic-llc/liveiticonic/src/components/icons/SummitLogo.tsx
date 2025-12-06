import { cn } from '@/lib/utils';

interface SummitLogoProps {
  className?: string;
}

const SummitLogo = ({ className }: SummitLogoProps) => {
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
        {/* Mountain Layers */}
        <path
          d="M18 8 L28 28 L8 28 Z"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2"
          fill="none"
          className="group-hover:stroke-[2.5] transition-all duration-400"
        />
        <path
          d="M18 8 L24 20 L12 20 Z"
          fill="hsl(var(--lii-gold))"
          opacity="0.2"
          className="group-hover:opacity-30 transition-opacity duration-400"
        />

        {/* Summit Flag */}
        <path
          d="M18 8 L18 4 M18 4 L22 6 L18 8"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
        />

        {/* Peak Marker */}
        <circle
          cx="18"
          cy="8"
          r="2"
          fill="hsl(var(--lii-gold))"
          className="group-hover:scale-125 transition-transform duration-300"
        />

        {/* Base Line */}
        <line x1="6" y1="28" x2="30" y2="28" stroke="hsl(var(--lii-champagne))" strokeWidth="2" />
      </svg>
    </div>
  );
};

export default SummitLogo;
