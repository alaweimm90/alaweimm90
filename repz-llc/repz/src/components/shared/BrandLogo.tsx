import React from 'react';
import { useNavigate } from 'react-router-dom';

interface BrandLogoProps {
  variant?: 'full' | 'icon-only' | 'text-only';
  size?: 'sm' | 'md' | 'lg' | 'xl';
  className?: string;
  onClick?: () => void;
  clickable?: boolean;
  colorVariant?: 'default' | 'white' | 'dark';
}

export const BrandLogo: React.FC<BrandLogoProps> = ({
  variant = 'full',
  size = 'md',
  className = '',
  onClick,
  clickable = true,
  colorVariant = 'default'
}) => {
  const navigate = useNavigate();

  const handleClick = () => {
    if (onClick) {
      onClick();
    } else if (clickable) {
      navigate('/');
    }
  };

  // Size mappings for the new bar chart logo
  const sizeConfig = {
    sm: { width: 32, height: 32, textSize: 'text-lg' },
    md: { width: 48, height: 48, textSize: 'text-xl' },
    lg: { width: 64, height: 64, textSize: 'text-2xl' },
    xl: { width: 80, height: 80, textSize: 'text-3xl' }
  };

  // Logo Icon Component using the new SVG bar chart design
  const LogoIcon = () => {
    const config = sizeConfig[size];
    return (
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
    );
  };

  // Brand Text Component
  const BrandText = () => {
    const config = sizeConfig[size];
    return (
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
    );
  };

  // Render based on variant
  const renderContent = () => {
    switch (variant) {
      case 'icon-only':
        return <LogoIcon />;
      
      case 'text-only':
        return <BrandText />;
      
      case 'full':
      default:
        return (
          <div className="flex items-center gap-4">
            <LogoIcon />
            <BrandText />
          </div>
        );
    }
  };

  return (
    <div
      className={`
        inline-flex items-center
        ${clickable ? 'cursor-pointer hover:opacity-90 transition-opacity duration-200' : ''}
        ${className}
      `}
      onClick={handleClick}
      role={clickable ? 'button' : undefined}
      tabIndex={clickable ? 0 : undefined}
      onKeyDown={clickable ? (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          handleClick();
        }
      } : undefined}
      aria-label={clickable ? 'REPZ - Go to homepage' : 'REPZ Logo'}
    >
      {renderContent()}
    </div>
  );
};