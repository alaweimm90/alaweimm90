import { cn } from '@/lib/utils';

interface NovaLogoProps {
  className?: string;
}

const NovaLogo = ({ className }: NovaLogoProps) => {
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
        {/* Starburst */}
        <path
          d="M18 4 L20 16 L32 18 L20 20 L18 32 L16 20 L4 18 L16 16 Z"
          fill="hsl(var(--lii-gold))"
          opacity="0.3"
          className="group-hover:opacity-40 transition-opacity duration-500"
        />
        <path
          d="M18 4 L20 16 L32 18 L20 20 L18 32 L16 20 L4 18 L16 16 Z"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.5"
          fill="none"
          className="group-hover:stroke-[2] transition-all duration-400"
        />

        {/* Inner Star */}
        <path
          d="M18 12 L19 17 L24 18 L19 19 L18 24 L17 19 L12 18 L17 17 Z"
          fill="hsl(var(--lii-gold))"
          className="group-hover:scale-110 transition-transform duration-300"
        />

        {/* Core */}
        <circle
          cx="18"
          cy="18"
          r="2"
          fill="hsl(var(--lii-champagne))"
          className="group-hover:scale-125 transition-transform duration-300"
        />
      </svg>
    </div>
  );
};

export default NovaLogo;
