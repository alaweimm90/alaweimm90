import { cn } from '@/lib/utils';

interface ArchitecturalBlockProps {
  className?: string;
}

const ArchitecturalBlock = ({ className }: ArchitecturalBlockProps) => {
  return (
    <div className={cn('relative group', className)}>
      <svg
        width="160"
        height="80"
        viewBox="0 0 160 80"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="transition-all duration-500 group-hover:scale-105"
      >
        {/* Architectural blocks - LIVE */}
        <rect
          x="10"
          y="20"
          width="35"
          height="40"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2"
          fill="hsl(var(--lii-gold))"
          fillOpacity="0.1"
        />
        <text
          x="27.5"
          y="47"
          fill="hsl(var(--lii-gold))"
          fontSize="18"
          fontWeight="700"
          fontFamily="sans-serif"
          textAnchor="middle"
        >
          LIVE
        </text>

        {/* IT block - smaller, offset */}
        <rect
          x="50"
          y="30"
          width="25"
          height="30"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.5"
          fill="hsl(var(--lii-gold))"
          fillOpacity="0.15"
        />
        <text
          x="62.5"
          y="50"
          fill="hsl(var(--lii-gold))"
          fontSize="14"
          fontWeight="500"
          fontFamily="sans-serif"
          textAnchor="middle"
        >
          IT
        </text>

        {/* ICONIC block - largest */}
        <rect
          x="80"
          y="15"
          width="70"
          height="50"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2"
          fill="hsl(var(--lii-gold))"
          fillOpacity="0.1"
        />
        <text
          x="115"
          y="47"
          fill="hsl(var(--lii-gold))"
          fontSize="20"
          fontWeight="700"
          fontFamily="sans-serif"
          textAnchor="middle"
          letterSpacing="1"
        >
          ICONIC
        </text>

        {/* Foundation line */}
        <rect x="5" y="65" width="150" height="2" fill="hsl(var(--lii-gold))" opacity="0.6" />
      </svg>
    </div>
  );
};

export default ArchitecturalBlock;
