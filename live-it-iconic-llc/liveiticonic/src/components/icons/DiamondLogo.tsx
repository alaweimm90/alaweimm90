import { cn } from '@/lib/utils';

interface DiamondLogoProps {
  className?: string;
}

const DiamondLogo = ({ className }: DiamondLogoProps) => {
  return (
    <div className={cn('relative group', className)}>
      <svg
        viewBox="0 0 36 36"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        preserveAspectRatio="xMidYMid meet"
        className="w-full h-full align-middle block transition-all duration-700 group-hover:scale-110"
      >
        {/* Outer Diamond - Represents Excellence & Precision */}
        <path
          d="M18 3 L30 18 L18 33 L6 18 Z"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.5"
          fill="none"
          className="group-hover:stroke-[2] transition-all duration-500"
        />

        {/* Inner Facets - Luxury & Refinement */}
        <path
          d="M18 3 L24 12 L18 18 L12 12 Z"
          fill="hsl(var(--lii-gold))"
          opacity="0.2"
          className="group-hover:opacity-30 transition-opacity duration-400"
        />
        <path d="M12 12 L18 18 L6 18 Z" fill="hsl(var(--lii-gold))" opacity="0.15" />
        <path d="M24 12 L30 18 L18 18 Z" fill="hsl(var(--lii-gold))" opacity="0.15" />

        {/* Center Point - Focus & Clarity */}
        <circle
          cx="18"
          cy="18"
          r="2"
          fill="hsl(var(--lii-gold))"
          className="group-hover:scale-125 transition-transform duration-300"
        />
      </svg>
    </div>
  );
};

export default DiamondLogo;
