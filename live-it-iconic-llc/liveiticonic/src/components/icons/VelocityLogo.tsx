import { cn } from '@/lib/utils';

interface VelocityLogoProps {
  className?: string;
}

const VelocityLogo = ({ className }: VelocityLogoProps) => {
  return (
    <div className={cn('relative group', className)}>
      <svg
        width="36"
        height="36"
        viewBox="0 0 36 36"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="transition-all duration-500 group-hover:scale-105"
      >
        {/* Dynamic Movement Lines - Performance & Speed */}
        <path
          d="M4 18 L12 14 L20 18 L28 12 L32 18"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2"
          strokeLinecap="round"
          className="group-hover:stroke-[3] transition-all duration-300"
        />
        <path
          d="M6 22 L14 18 L22 22 L30 16"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.5"
          strokeLinecap="round"
          opacity="0.7"
          className="group-hover:opacity-90 transition-opacity duration-300"
        />
        <path
          d="M8 26 L16 22 L24 26 L32 20"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1"
          strokeLinecap="round"
          opacity="0.5"
          className="group-hover:opacity-70 transition-opacity duration-300"
        />

        {/* Central Focus Point - "I" for Iconic */}
        <rect
          x="17"
          y="10"
          width="2"
          height="16"
          fill="hsl(var(--lii-gold))"
          className="group-hover:scale-110 transition-transform duration-200"
        />
        <circle cx="18" cy="8" r="1.5" fill="hsl(var(--lii-gold))" />
        <circle cx="18" cy="28" r="1.5" fill="hsl(var(--lii-gold))" />
      </svg>
    </div>
  );
};

export default VelocityLogo;
