import { cn } from '@/lib/utils';

interface NexusLogoProps {
  className?: string;
}

const NexusLogo = ({ className }: NexusLogoProps) => {
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
        {/* Nexus Connection Points */}
        <circle cx="18" cy="8" r="2" fill="hsl(var(--lii-gold))" />
        <circle cx="28" cy="18" r="2" fill="hsl(var(--lii-gold))" />
        <circle cx="18" cy="28" r="2" fill="hsl(var(--lii-gold))" />
        <circle cx="8" cy="18" r="2" fill="hsl(var(--lii-gold))" />

        {/* Connection Lines */}
        <path
          d="M18 8 L28 18 L18 28 L8 18 Z"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.5"
          fill="none"
          className="group-hover:stroke-[2] transition-all duration-400"
        />

        {/* Inner Cross */}
        <path
          d="M18 12 L18 24 M12 18 L24 18"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.5"
          strokeLinecap="round"
          opacity="0.6"
          className="group-hover:opacity-80 transition-opacity duration-300"
        />

        {/* Central Core */}
        <circle
          cx="18"
          cy="18"
          r="3"
          fill="hsl(var(--lii-gold))"
          className="group-hover:scale-125 transition-transform duration-300"
        />
      </svg>
    </div>
  );
};

export default NexusLogo;
