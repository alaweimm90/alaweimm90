import { cn } from '@/lib/utils';

interface ElevationLogoProps {
  className?: string;
}

const ElevationLogo = ({ className }: ElevationLogoProps) => {
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
        {/* Mountain Peaks - Aspiration & Achievement */}
        <path
          d="M6 28 L12 16 L18 20 L24 12 L30 28"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
          className="group-hover:stroke-[2.5] transition-all duration-400"
        />

        {/* Gradient Fill - Luxury Ascent */}
        <path
          d="M6 28 L12 16 L18 20 L24 12 L30 28 L6 28 Z"
          fill="url(#elevationGradient)"
          opacity="0.3"
          className="group-hover:opacity-40 transition-opacity duration-400"
        />

        {/* Peak Accent - Excellence */}
        <circle
          cx="24"
          cy="12"
          r="2"
          fill="hsl(var(--lii-gold))"
          className="group-hover:scale-125 transition-transform duration-300"
        />

        {/* Horizon Line - Foundation */}
        <line
          x1="4"
          y1="28"
          x2="32"
          y2="28"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1"
          opacity="0.6"
        />

        <defs>
          <linearGradient id="elevationGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="hsl(var(--lii-gold))" stopOpacity="0.4" />
            <stop offset="100%" stopColor="hsl(var(--lii-gold))" stopOpacity="0.1" />
          </linearGradient>
        </defs>
      </svg>
    </div>
  );
};

export default ElevationLogo;
