import { cn } from '@/lib/utils';

interface ApexLogoProps {
  className?: string;
}

const ApexLogo = ({ className }: ApexLogoProps) => {
  return (
    <div className={cn('relative group', className)}>
      <svg
        width="36"
        height="36"
        viewBox="0 0 36 36"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="transition-all duration-600 group-hover:scale-110"
      >
        {/* Apex Triangle */}
        <path
          d="M18 6 L30 30 L6 30 Z"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2"
          fill="none"
          className="group-hover:stroke-[2.5] transition-all duration-500"
        />

        {/* Inner Structure */}
        <path
          d="M18 6 L18 18 M12 22 L24 22"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.5"
          strokeLinecap="round"
          opacity="0.7"
          className="group-hover:opacity-90 transition-opacity duration-400"
        />

        {/* Peak Accent */}
        <circle
          cx="18"
          cy="6"
          r="2"
          fill="hsl(var(--lii-gold))"
          className="group-hover:scale-125 transition-transform duration-300"
        />

        {/* Base Points */}
        <circle cx="6" cy="30" r="1.5" fill="hsl(var(--lii-champagne))" opacity="0.8" />
        <circle cx="30" cy="30" r="1.5" fill="hsl(var(--lii-champagne))" opacity="0.8" />
      </svg>
    </div>
  );
};

export default ApexLogo;
