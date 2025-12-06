import { cn } from '@/lib/utils';

interface ZenithLogoProps {
  className?: string;
}

const ZenithLogo = ({ className }: ZenithLogoProps) => {
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
        {/* Concentric Circles - Peak Achievement */}
        <circle
          cx="18"
          cy="18"
          r="14"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1"
          fill="none"
          opacity="0.3"
        />
        <circle
          cx="18"
          cy="18"
          r="10"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.5"
          fill="none"
          opacity="0.5"
          className="group-hover:opacity-70 transition-opacity duration-400"
        />
        <circle
          cx="18"
          cy="18"
          r="6"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2"
          fill="none"
          className="group-hover:stroke-[2.5] transition-all duration-300"
        />

        {/* Zenith Point */}
        <path
          d="M18 8 L18 4"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2"
          strokeLinecap="round"
          className="group-hover:stroke-[3] transition-all duration-300"
        />
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

export default ZenithLogo;
