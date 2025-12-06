import { cn } from '@/lib/utils';

interface VortexLogoProps {
  className?: string;
}

const VortexLogo = ({ className }: VortexLogoProps) => {
  return (
    <div className={cn('relative group', className)}>
      <svg
        width="36"
        height="36"
        viewBox="0 0 36 36"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="transition-all duration-700 group-hover:rotate-12 group-hover:scale-110"
      >
        {/* Spiral Vortex */}
        <path
          d="M18 18 L26 18 A8 8 0 0 1 18 26 L18 18 L18 10 A8 8 0 0 1 26 18"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2"
          fill="none"
          className="group-hover:stroke-[2.5] transition-all duration-500"
        />
        <path
          d="M18 18 L10 18 A8 8 0 0 0 18 10"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2"
          fill="none"
          opacity="0.7"
          className="group-hover:opacity-90 transition-opacity duration-400"
        />

        {/* Outer Ring */}
        <circle
          cx="18"
          cy="18"
          r="13"
          stroke="hsl(var(--lii-champagne))"
          strokeWidth="1"
          fill="none"
          opacity="0.3"
        />

        {/* Center Point */}
        <circle
          cx="18"
          cy="18"
          r="2"
          fill="hsl(var(--lii-gold))"
          className="group-hover:scale-150 transition-transform duration-300"
        />
      </svg>
    </div>
  );
};

export default VortexLogo;
