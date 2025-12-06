import { cn } from '@/lib/utils';

interface AuroraLogoProps {
  className?: string;
}

const AuroraLogo = ({ className }: AuroraLogoProps) => {
  return (
    <div className={cn('relative group', className)}>
      <svg
        width="36"
        height="36"
        viewBox="0 0 36 36"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="transition-all duration-700 group-hover:scale-110"
      >
        {/* Flowing Aurora Waves */}
        <path
          d="M4 18 Q10 10, 18 14 T32 18"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2"
          fill="none"
          className="group-hover:stroke-[2.5] transition-all duration-500"
        />
        <path
          d="M4 22 Q10 16, 18 20 T32 22"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.5"
          fill="none"
          opacity="0.6"
          className="group-hover:opacity-80 transition-opacity duration-500"
        />
        <path
          d="M4 14 Q10 8, 18 12 T32 14"
          stroke="hsl(var(--lii-champagne))"
          strokeWidth="1"
          fill="none"
          opacity="0.4"
          className="group-hover:opacity-60 transition-opacity duration-500"
        />

        {/* Radiant Center */}
        <circle
          cx="18"
          cy="18"
          r="3"
          fill="hsl(var(--lii-gold))"
          className="group-hover:scale-125 transition-transform duration-300"
        />
        <circle
          cx="18"
          cy="18"
          r="5"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="0.5"
          fill="none"
          opacity="0.3"
        />
      </svg>
    </div>
  );
};

export default AuroraLogo;
