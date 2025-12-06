import { cn } from '@/lib/utils';

interface PulseLogoProps {
  className?: string;
}

const PulseLogo = ({ className }: PulseLogoProps) => {
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
        {/* Heartbeat/Pulse Line */}
        <path
          d="M4 18 L10 18 L13 12 L16 24 L19 14 L22 20 L26 18 L32 18"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
          className="group-hover:stroke-[3] transition-all duration-300"
        />

        {/* Pulse Points */}
        <circle
          cx="16"
          cy="24"
          r="1.5"
          fill="hsl(var(--lii-gold))"
          className="group-hover:scale-150 transition-transform duration-300"
        />
        <circle
          cx="13"
          cy="12"
          r="1.5"
          fill="hsl(var(--lii-gold))"
          className="group-hover:scale-150 transition-transform duration-300"
        />

        {/* Energy Rings */}
        <circle
          cx="18"
          cy="18"
          r="12"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="0.5"
          fill="none"
          opacity="0.2"
          className="group-hover:opacity-40 transition-opacity duration-400"
        />
      </svg>
    </div>
  );
};

export default PulseLogo;
