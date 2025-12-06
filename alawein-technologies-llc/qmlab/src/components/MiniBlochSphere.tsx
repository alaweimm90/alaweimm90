import React, { useState, useEffect } from 'react';
import { useAccessibility } from '@/hooks/useAccessibility';

interface MiniBlochSphereProps {
  className?: string;
  animated?: boolean;
  preset?: 'identity' | 'hadamard' | 'superposition' | 'entangled';
}

export const MiniBlochSphere: React.FC<MiniBlochSphereProps> = ({ 
  className = "", 
  animated = true,
  preset = 'hadamard' 
}) => {
  const [phase, setPhase] = useState(0);
  const { reduceMotion } = useAccessibility();
  
  useEffect(() => {
    if (!animated || reduceMotion) return;
    
    const interval = setInterval(() => {
      setPhase(prev => (prev + 0.02) % (2 * Math.PI));
    }, 32); // ~30fps
    
    return () => clearInterval(interval);
  }, [animated, reduceMotion]);
  
  // Calculate state vector position based on preset and phase
  const getStateVector = () => {
    switch (preset) {
      case 'identity':
        return { x: 0, y: 0, z: 1 }; // |0⟩ state
      case 'hadamard':
        return { 
          x: Math.cos(phase) * 0.8, 
          y: 0, 
          z: Math.sin(phase) * 0.6 
        };
      case 'superposition':
        return { 
          x: Math.cos(phase) * 0.9, 
          y: Math.sin(phase) * 0.9, 
          z: 0 
        };
      case 'entangled':
        return { 
          x: Math.cos(phase * 1.5) * 0.7, 
          y: Math.sin(phase * 2) * 0.7, 
          z: Math.cos(phase * 0.8) * 0.5 
        };
      default:
        return { x: 0, y: 0, z: 1 };
    }
  };
  
  const stateVector = getStateVector();
  
  // Convert 3D coordinates to 2D for display
  const x2d = stateVector.x * 40 + 50;
  const y2d = -stateVector.z * 40 + 50;
  const depth = stateVector.y; // For visual depth indication
  
  return (
    <div className={`relative ${className}`}>
      <svg 
        viewBox="0 0 100 100" 
        className="w-full h-full"
        role="img"
        aria-label={`Quantum state visualization: ${preset} preset`}
      >
        {/* Background circle (Bloch sphere) */}
        <circle 
          cx="50" 
          cy="50" 
          r="45" 
          fill="none" 
          stroke="hsl(var(--border))" 
          strokeWidth="1" 
          opacity="0.6"
        />
        
        {/* Equator */}
        <ellipse 
          cx="50" 
          cy="50" 
          rx="45" 
          ry="12" 
          fill="none" 
          stroke="hsl(var(--border))" 
          strokeWidth="0.5" 
          opacity="0.4"
        />
        
        {/* Vertical axis */}
        <line 
          x1="50" 
          y1="5" 
          x2="50" 
          y2="95" 
          stroke="hsl(var(--muted))" 
          strokeWidth="0.5" 
          opacity="0.3"
        />
        
        {/* State vector */}
        <line 
          x1="50" 
          y1="50" 
          x2={x2d} 
          y2={y2d} 
          stroke="hsl(var(--primary))" 
          strokeWidth="2" 
          className="drop-shadow-sm"
        />
        
        {/* State point */}
        <circle 
          cx={x2d} 
          cy={y2d} 
          r={3 + Math.abs(depth) * 1.5} 
          fill="hsl(var(--primary))" 
          className="drop-shadow-md"
          style={{
            filter: `brightness(${1 + depth * 0.3})`
          }}
        />
        
        {/* Subtle glow effect */}
        <circle 
          cx={x2d} 
          cy={y2d} 
          r={6 + Math.abs(depth) * 2} 
          fill="hsl(var(--primary))" 
          opacity={0.2 + Math.abs(depth) * 0.1}
          className="animate-pulse"
        />
        
        {/* Axis labels */}
        <text 
          x="50" 
          y="8" 
          textAnchor="middle" 
          className="text-xs fill-muted font-mono" 
          fontSize="6"
        >
          |0⟩
        </text>
        <text 
          x="50" 
          y="97" 
          textAnchor="middle" 
          className="text-xs fill-muted font-mono" 
          fontSize="6"
        >
          |1⟩
        </text>
      </svg>
      
      {/* State info overlay */}
      <div className="absolute bottom-0 right-0 text-xs font-mono text-muted bg-surface-1/80 rounded px-1 py-0.5">
        {preset}
      </div>
    </div>
  );
};