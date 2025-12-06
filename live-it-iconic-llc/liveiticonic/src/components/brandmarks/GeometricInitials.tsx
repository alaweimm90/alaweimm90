import { cn } from '@/lib/utils';

interface GeometricInitialsProps {
  className?: string;
}

const GeometricInitials = ({ className }: GeometricInitialsProps) => {
  return (
    <div className={cn('relative group', className)}>
      <svg
        width="100"
        height="40"
        viewBox="0 0 100 40"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="transition-all duration-500 group-hover:scale-105"
      >
        {/* L in geometric frame */}
        <rect
          x="5"
          y="5"
          width="20"
          height="30"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.5"
          fill="none"
        />
        <path
          d="M10 10 L10 30 L20 30"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2"
          strokeLinecap="round"
        />

        {/* . */}
        <circle cx="35" cy="30" r="2" fill="hsl(var(--lii-gold))" />

        {/* I in geometric frame */}
        <rect
          x="45"
          y="5"
          width="10"
          height="30"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.5"
          fill="none"
        />
        <rect x="48" y="10" width="4" height="20" fill="hsl(var(--lii-gold))" />

        {/* . */}
        <circle cx="65" cy="30" r="2" fill="hsl(var(--lii-gold))" />

        {/* I in geometric frame */}
        <rect
          x="75"
          y="5"
          width="10"
          height="30"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.5"
          fill="none"
        />
        <rect x="78" y="10" width="4" height="20" fill="hsl(var(--lii-gold))" />
      </svg>
    </div>
  );
};

export default GeometricInitials;
