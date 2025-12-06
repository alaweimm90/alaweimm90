import { cn } from '@/lib/utils';

interface PhoenixLogoProps {
  className?: string;
}

const PhoenixLogo = ({ className }: PhoenixLogoProps) => {
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
        {/* Phoenix Wings */}
        <path
          d="M18 28 L12 18 L8 20 L6 16"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="group-hover:stroke-[2.2] transition-all duration-400"
        />
        <path
          d="M18 28 L24 18 L28 20 L30 16"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="group-hover:stroke-[2.2] transition-all duration-400"
        />

        {/* Head and Crown */}
        <circle
          cx="18"
          cy="12"
          r="3"
          fill="hsl(var(--lii-gold))"
          className="group-hover:scale-110 transition-transform duration-300"
        />
        <path
          d="M15 10 L18 6 L21 10"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
        />

        {/* Tail Feathers */}
        <path
          d="M18 28 L18 32"
          stroke="hsl(var(--lii-champagne))"
          strokeWidth="2"
          strokeLinecap="round"
        />
      </svg>
    </div>
  );
};

export default PhoenixLogo;
