import { cn } from '@/lib/utils';

interface InfinityLogoProps {
  className?: string;
}

const InfinityLogo = ({ className }: InfinityLogoProps) => {
  return (
    <div className={cn('relative group', className)}>
      <svg
        width="36"
        height="36"
        viewBox="0 0 36 36"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="transition-all duration-700 group-hover:scale-110 group-hover:rotate-2"
      >
        {/* Infinity Symbol - Endless Potential */}
        <path
          d="M8 18 C8 14, 12 10, 16 14 C20 18, 20 18, 24 14 C28 10, 32 14, 32 18 C32 22, 28 26, 24 22 C20 18, 20 18, 16 22 C12 26, 8 22, 8 18 Z"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2"
          fill="none"
          strokeLinecap="round"
          className="group-hover:stroke-[2.5] transition-all duration-500"
        />

        {/* Central Intersection - Unity */}
        <circle
          cx="18"
          cy="18"
          r="1.5"
          fill="hsl(var(--lii-gold))"
          className="group-hover:scale-150 transition-transform duration-400"
        />

        {/* Flow Indicators - Dynamic Energy */}
        <circle
          cx="12"
          cy="15"
          r="1"
          fill="hsl(var(--lii-gold))"
          opacity="0.6"
          className="group-hover:opacity-100 group-hover:scale-125 transition-all duration-300"
        />
        <circle
          cx="24"
          cy="21"
          r="1"
          fill="hsl(var(--lii-gold))"
          opacity="0.6"
          className="group-hover:opacity-100 group-hover:scale-125 transition-all duration-300 delay-100"
        />

        {/* Outer Glow Effect */}
        <path
          d="M8 18 C8 14, 12 10, 16 14 C20 18, 20 18, 24 14 C28 10, 32 14, 32 18 C32 22, 28 26, 24 22 C20 18, 20 18, 16 22 C12 26, 8 22, 8 18 Z"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="4"
          fill="none"
          opacity="0.2"
          className="group-hover:opacity-30 transition-opacity duration-500"
        />
      </svg>
    </div>
  );
};

export default InfinityLogo;
