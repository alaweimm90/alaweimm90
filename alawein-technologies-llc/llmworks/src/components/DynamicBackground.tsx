import { memo, useEffect, useState, useRef } from 'react';

interface FloatingElement {
  id: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  opacity: number;
  type: 'circuit' | 'data' | 'signal' | 'node';
  color: string;
  rotation: number;
  rotationSpeed: number;
}

interface DynamicBackgroundProps {
  intensity?: 'low' | 'medium' | 'high';
  theme?: 'tactical' | 'neural' | 'cyber';
}

const DynamicBackgroundComponent = ({ 
  intensity = 'medium',
  theme = 'tactical' 
}: DynamicBackgroundProps) => {
  const [elements, setElements] = useState<FloatingElement[]>([]);
  const [mousePos, setMousePos] = useState({ x: 50, y: 50 });
  const containerRef = useRef<HTMLDivElement>(null);

  const getElementCount = () => {
    switch (intensity) {
      case 'low': return 8;
      case 'medium': return 15;
      case 'high': return 25;
      default: return 15;
    }
  };

  const getThemeColors = () => {
    switch (theme) {
      case 'tactical':
        return ['#2563EB', '#0891B2', '#B45309', '#059669'];
      case 'neural':
        return ['#8B5CF6', '#EC4899', '#10B981', '#F59E0B'];
      case 'cyber':
        return ['#06B6D4', '#84CC16', '#F97316', '#EF4444'];
      default:
        return ['#2563EB', '#0891B2', '#B45309', '#059669'];
    }
  };

  const generateElements = (): FloatingElement[] => {
    const colors = getThemeColors();
    const types: Array<'circuit' | 'data' | 'signal' | 'node'> = ['circuit', 'data', 'signal', 'node'];
    
    return Array.from({ length: getElementCount() }, (_, i) => ({
      id: `element-${i}`,
      x: Math.random() * 100,
      y: Math.random() * 100,
      vx: (Math.random() - 0.5) * 0.2,
      vy: (Math.random() - 0.5) * 0.2,
      size: 2 + Math.random() * 4,
      opacity: 0.1 + Math.random() * 0.3,
      type: types[Math.floor(Math.random() * types.length)],
      color: colors[Math.floor(Math.random() * colors.length)],
      rotation: Math.random() * 360,
      rotationSpeed: (Math.random() - 0.5) * 2
    }));
  };

  useEffect(() => {
    setElements(generateElements());
  }, [intensity, theme]);

  // Mouse tracking for interactive effects
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!containerRef.current) return;
      
      const rect = containerRef.current.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * 100;
      const y = ((e.clientY - rect.top) / rect.height) * 100;
      
      setMousePos({ x, y });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  // Animation loop
  useEffect(() => {
    const animationLoop = setInterval(() => {
      setElements(prevElements => 
        prevElements.map(element => ({
          ...element,
          x: (element.x + element.vx + 100) % 100,
          y: (element.y + element.vy + 100) % 100,
          rotation: (element.rotation + element.rotationSpeed) % 360,
          // Subtle mouse interaction
          vx: element.vx + (Math.random() - 0.5) * 0.01,
          vy: element.vy + (Math.random() - 0.5) * 0.01
        }))
      );
    }, 50);

    return () => clearInterval(animationLoop);
  }, []);

  const renderElement = (element: FloatingElement) => {
    const distance = Math.sqrt(
      Math.pow(element.x - mousePos.x, 2) + Math.pow(element.y - mousePos.y, 2)
    );
    const mouseInfluence = Math.max(0, (20 - distance) / 20);
    const adjustedOpacity = element.opacity + (mouseInfluence * 0.3);

    const commonProps = {
      style: {
        left: `${element.x}%`,
        top: `${element.y}%`,
        transform: `translate(-50%, -50%) rotate(${element.rotation}deg) scale(${1 + mouseInfluence * 0.5})`,
        opacity: adjustedOpacity,
        color: element.color,
        fontSize: `${element.size}px`,
        transition: 'all 0.3s ease-out'
      },
      className: "absolute pointer-events-none"
    };

    switch (element.type) {
      case 'circuit':
        return (
          <div key={element.id} {...commonProps}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
              <path d="M2 2h12v12H2V2zm2 2v8h8V4H4zm2 2h4v4H6V6z" opacity="0.6"/>
              <circle cx="4" cy="4" r="1" fill="currentColor"/>
              <circle cx="12" cy="12" r="1" fill="currentColor"/>
            </svg>
          </div>
        );
      
      case 'data':
        return (
          <div key={element.id} {...commonProps}>
            <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
              <rect x="1" y="1" width="10" height="2" opacity="0.8"/>
              <rect x="1" y="4" width="6" height="2" opacity="0.6"/>
              <rect x="1" y="7" width="8" height="2" opacity="0.4"/>
            </svg>
          </div>
        );
      
      case 'signal':
        return (
          <div key={element.id} {...commonProps}>
            <div 
              style={{
                width: `${element.size * 2}px`,
                height: '2px',
                background: `linear-gradient(90deg, transparent, ${element.color}, transparent)`,
                borderRadius: '1px'
              }}
            />
          </div>
        );
      
      case 'node':
        return (
          <div 
            key={element.id} 
            {...commonProps}
          >
            <div
              style={{
                width: `${element.size}px`,
                height: `${element.size}px`,
                borderRadius: '50%',
                background: `radial-gradient(circle, ${element.color}40, transparent)`,
                border: `1px solid ${element.color}60`,
                boxShadow: `0 0 ${element.size * 2}px ${element.color}20`
              }}
            />
          </div>
        );
      
      default:
        return null;
    }
  };

  return (
    <div 
      ref={containerRef}
      className="fixed inset-0 pointer-events-none overflow-hidden"
      style={{ zIndex: 1 }}
    >
      {/* Gradient Overlays for Depth */}
      <div className="absolute inset-0 bg-gradient-to-br from-primary/[0.02] via-transparent to-secondary/[0.02]" />
      <div className="absolute inset-0 bg-gradient-to-tl from-accent/[0.01] via-transparent to-primary/[0.01]" />
      
      {/* Animated Grid */}
      <div className="absolute inset-0 opacity-10">
        <svg width="100%" height="100%" className="absolute inset-0">
          <defs>
            <pattern 
              id="tactical-grid" 
              width="80" 
              height="80" 
              patternUnits="userSpaceOnUse"
              className="text-primary/20"
            >
              <path 
                d="M 80 0 L 0 0 0 80" 
                fill="none" 
                stroke="currentColor" 
                strokeWidth="0.5"
                opacity="0.3"
              />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#tactical-grid)" />
        </svg>
      </div>

      {/* Floating Elements */}
      {elements.map(renderElement)}

      {/* Ambient Light Effects */}
      <div className="absolute inset-0">
        <div 
          className="absolute w-96 h-96 bg-primary/5 rounded-full blur-3xl"
          style={{
            left: `${mousePos.x - 10}%`,
            top: `${mousePos.y - 10}%`,
            transform: 'translate(-50%, -50%)',
            transition: 'all 0.3s ease-out'
          }}
        />
        <div 
          className="absolute w-64 h-64 bg-secondary/3 rounded-full blur-2xl"
          style={{
            left: `${100 - mousePos.x + 5}%`,
            top: `${100 - mousePos.y + 5}%`,
            transform: 'translate(-50%, -50%)',
            transition: 'all 0.4s ease-out'
          }}
        />
      </div>

      {/* Scanning Lines */}
      <div className="absolute inset-0">
        <div 
          className="absolute w-full h-0.5 bg-gradient-to-r from-transparent via-primary/20 to-transparent"
          style={{
            top: '20%',
            animation: 'scan-vertical 8s linear infinite'
          }}
        />
        <div 
          className="absolute h-full w-0.5 bg-gradient-to-b from-transparent via-secondary/15 to-transparent"
          style={{
            left: '80%',
            animation: 'scan-horizontal 12s linear infinite'
          }}
        />
      </div>

      <style>{`
        @keyframes scan-vertical {
          0% { top: -2px; opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { top: 100%; opacity: 0; }
        }
        
        @keyframes scan-horizontal {
          0% { left: -2px; opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { left: 100%; opacity: 0; }
        }
      `}</style>
    </div>
  );
};

export const DynamicBackground = memo(DynamicBackgroundComponent);