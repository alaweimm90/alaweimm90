import { cn } from '@/lib/utils';

interface TechMonospaceProps {
  className?: string;
}

const TechMonospace = ({ className }: TechMonospaceProps) => {
  return (
    <div className={cn('relative group', className)}>
      <svg
        width="200"
        height="40"
        viewBox="0 0 200 40"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="transition-all duration-500"
      >
        {/* Tech grid background */}
        <rect
          x="8"
          y="8"
          width="184"
          height="24"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="0.5"
          fill="none"
          opacity="0.2"
        />

        {/* LIVE_IT_ICONIC in monospace */}
        <text
          x="12"
          y="25"
          fill="hsl(var(--lii-gold))"
          fontSize="16"
          fontWeight="500"
          fontFamily="monospace"
          letterSpacing="2"
        >
          LIVE_IT_ICONIC
        </text>

        {/* Tech corner brackets */}
        <path
          d="M5 5 L5 10 M5 5 L10 5"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
        <path
          d="M195 5 L195 10 M195 5 L190 5"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
        <path
          d="M5 35 L5 30 M5 35 L10 35"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
        <path
          d="M195 35 L195 30 M195 35 L190 35"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
      </svg>
    </div>
  );
};

export default TechMonospace;
