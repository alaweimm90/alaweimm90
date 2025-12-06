import React from 'react';

interface RepzLogoProps {
  size?: 'sm' | 'md' | 'lg' | 'xl';
  variant?: string; // Allow any variant for backwards compatibility
  className?: string;
  showText?: boolean;
}

export const RepzLogo: React.FC<RepzLogoProps> = ({ 
  size = 'md', 
  variant, // Accept but ignore variant for now
  className = '',
  showText = true
}) => {
  // Size mappings for the new bar chart logo
  const sizeConfig = {
    sm: { width: 32, height: 32, textSize: 'text-lg' },
    md: { width: 48, height: 48, textSize: 'text-xl' },
    lg: { width: 64, height: 64, textSize: 'text-2xl' },
    xl: { width: 80, height: 80, textSize: 'text-3xl' }
  };

  const config = sizeConfig[size];

  return (
    <div className={`flex items-center gap-4 ${className}`}>
      {/* New bar chart logo SVG */}
      <svg 
        width={config.width} 
        height={config.height} 
        viewBox="0 0 80 80"
        className="flex-shrink-0"
      >
        {/* Orange rounded square background */}
        <rect 
          x="0" 
          y="0" 
          width="80" 
          height="80" 
          rx="12" 
          fill="#F15B23" 
        />
        {/* Black bar chart bars */}
        <rect 
          x="16" 
          y="19" 
          width="10" 
          height="42" 
          rx="3" 
          fill="black" 
        />
        <rect 
          x="35" 
          y="11" 
          width="12" 
          height="58" 
          rx="3" 
          fill="black" 
        />
        <rect 
          x="56" 
          y="19" 
          width="10" 
          height="42" 
          rx="3" 
          fill="black" 
        />
      </svg>
      
      {/* REPZ text */}
      {showText && (
        <span 
          className={`font-bold ${config.textSize}`}
          style={{ 
            letterSpacing: '0.1em',
            color: '#F15B23',
            fontFamily: 'Montserrat, sans-serif',
            fontWeight: '700'
          }}
        >
          REPZ
        </span>
      )}
    </div>
  );
};