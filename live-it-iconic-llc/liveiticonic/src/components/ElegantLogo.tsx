import { cn } from '@/lib/utils';

interface ElegantLogoProps {
  className?: string;
}

const ElegantLogo = ({ className }: ElegantLogoProps) => {
  return (
    <div className={cn('relative group', className)}>
      {/* Sophisticated Brand Mark */}
      <svg
        width="36"
        height="36"
        viewBox="0 0 36 36"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="transition-all duration-700 group-hover:scale-110 group-hover:rotate-1"
      >
        {/* Outer Diamond Frame */}
        <path
          d="M18 2 L30 18 L18 34 L6 18 Z"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.5"
          fill="none"
          className="group-hover:stroke-[2] transition-all duration-500"
        />

        {/* Inner Geometric L.I.I */}
        <path
          d="M12 12 L12 24 M12 22 L15 22 M20 12 L20 18 M20 20 L20 20 M24 12 L24 24 M24 22 L27 22"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.2"
          strokeLinecap="round"
          className="group-hover:stroke-[1.8] transition-all duration-400"
        />

        {/* Center Accent */}
        <circle
          cx="18"
          cy="18"
          r="1.5"
          fill="hsl(var(--lii-gold))"
          className="transition-all duration-500 group-hover:scale-125"
        />

        {/* Luxury Corner Accents */}
        <circle
          cx="18"
          cy="6"
          r="0.8"
          fill="hsl(var(--lii-champagne))"
          opacity="0.8"
          className="group-hover:opacity-100 transition-opacity duration-400"
        />
        <circle
          cx="18"
          cy="30"
          r="0.8"
          fill="hsl(var(--lii-champagne))"
          opacity="0.8"
          className="group-hover:opacity-100 transition-opacity duration-400"
        />
      </svg>
    </div>
  );
};

export default ElegantLogo;
