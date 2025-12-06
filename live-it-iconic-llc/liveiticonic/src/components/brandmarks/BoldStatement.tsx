import { cn } from '@/lib/utils';

interface BoldStatementProps {
  className?: string;
}

const BoldStatement = ({ className }: BoldStatementProps) => {
  return (
    <div className={cn('relative group', className)}>
      <svg
        width="200"
        height="45"
        viewBox="0 0 200 45"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="transition-all duration-500"
      >
        {/* LIVE - Bold */}
        <text
          x="10"
          y="30"
          fill="hsl(var(--lii-gold))"
          fontSize="28"
          fontWeight="800"
          fontFamily="sans-serif"
          letterSpacing="1"
        >
          LIVE
        </text>

        {/* Divider */}
        <rect x="80" y="12" width="2" height="20" fill="hsl(var(--lii-gold))" opacity="0.4" />

        {/* IT - Medium */}
        <text
          x="90"
          y="30"
          fill="hsl(var(--lii-gold))"
          fontSize="22"
          fontWeight="400"
          fontFamily="sans-serif"
        >
          IT
        </text>

        {/* Divider */}
        <rect x="120" y="12" width="2" height="20" fill="hsl(var(--lii-gold))" opacity="0.4" />

        {/* ICONIC - Bold */}
        <text
          x="130"
          y="30"
          fill="hsl(var(--lii-gold))"
          fontSize="28"
          fontWeight="800"
          fontFamily="sans-serif"
          letterSpacing="1"
        >
          ICONIC
        </text>
      </svg>
    </div>
  );
};

export default BoldStatement;
