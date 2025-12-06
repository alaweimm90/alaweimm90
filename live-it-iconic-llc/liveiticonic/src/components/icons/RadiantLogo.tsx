import { cn } from '@/lib/utils';

interface RadiantLogoProps {
  className?: string;
}

const RadiantLogo = ({ className }: RadiantLogoProps) => {
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
        {/* Radiant Rays */}
        <path
          d="M18 6 L18 10"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2"
          strokeLinecap="round"
        />
        <path
          d="M18 26 L18 30"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2"
          strokeLinecap="round"
        />
        <path
          d="M6 18 L10 18"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2"
          strokeLinecap="round"
        />
        <path
          d="M26 18 L30 18"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2"
          strokeLinecap="round"
        />
        <path
          d="M10 10 L12.5 12.5"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.5"
          strokeLinecap="round"
          opacity="0.7"
        />
        <path
          d="M23.5 23.5 L26 26"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.5"
          strokeLinecap="round"
          opacity="0.7"
        />
        <path
          d="M10 26 L12.5 23.5"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.5"
          strokeLinecap="round"
          opacity="0.7"
        />
        <path
          d="M23.5 12.5 L26 10"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.5"
          strokeLinecap="round"
          opacity="0.7"
        />

        {/* Central Sun */}
        <circle
          cx="18"
          cy="18"
          r="6"
          fill="hsl(var(--lii-gold))"
          className="group-hover:scale-110 transition-transform duration-300"
        />
        <circle
          cx="18"
          cy="18"
          r="8"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="0.5"
          fill="none"
          opacity="0.4"
        />
      </svg>
    </div>
  );
};

export default RadiantLogo;
