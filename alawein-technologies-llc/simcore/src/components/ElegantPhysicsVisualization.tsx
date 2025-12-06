import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  Settings,
  Zap,
  Atom,
  Waves,
  ChevronRight
} from 'lucide-react';
import { usePrefersReducedMotion } from '@/lib/accessibility-utils';
import { useDomainTheme } from '@/components/DomainThemeProvider';

interface ElegantPhysicsVisualizationProps {
  type: 'pendulum' | 'wave' | 'orbit' | 'field';
  className?: string;
  autoPlay?: boolean;
}

export const ElegantPhysicsVisualization: React.FC<ElegantPhysicsVisualizationProps> = ({
  type,
  className = '',
  autoPlay = true
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();
  const [isPlaying, setIsPlaying] = useState(autoPlay);
  const [time, setTime] = useState(0);
  const [isHovered, setIsHovered] = useState(false);
  const prefersReducedMotion = usePrefersReducedMotion();
  const { currentDomain } = useDomainTheme();

  // Enhanced animation loop with smooth performance
  useEffect(() => {
    if (prefersReducedMotion || !isPlaying) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set high-DPI canvas dimensions
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';

    let lastTime = 0;
    const animate = (currentTime: number) => {
      const deltaTime = (currentTime - lastTime) / 1000;
      lastTime = currentTime;
      
      setTime(prevTime => prevTime + deltaTime);

      ctx.clearRect(0, 0, rect.width, rect.height);
      ctx.save();

      // Enhanced rendering with smooth animations
      switch (type) {
        case 'pendulum':
          drawEnhancedPendulum(ctx, rect, currentTime * 0.001);
          break;
        case 'wave':
          drawEnhancedWave(ctx, rect, currentTime * 0.001);
          break;
        case 'orbit':
          drawEnhancedOrbit(ctx, rect, currentTime * 0.001);
          break;
        case 'field':
          drawEnhancedField(ctx, rect, currentTime * 0.001);
          break;
      }

      ctx.restore();
      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [type, isPlaying, prefersReducedMotion]);

  // Enhanced drawing functions with uniform styling
  const drawEnhancedPendulum = (ctx: CanvasRenderingContext2D, rect: DOMRect, time: number) => {
    const centerX = rect.width / 2;
    const centerY = 60;
    const length = Math.min(rect.width, rect.height) * 0.3;
    const angle = Math.sin(time * 1.5) * 0.6;
    
    const bobX = centerX + Math.sin(angle) * length;
    const bobY = centerY + Math.cos(angle) * length;

    // Uniform blue string
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.8)';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(bobX, bobY);
    ctx.stroke();

    // Uniform blue pivot
    ctx.shadowColor = 'rgba(59, 130, 246, 0.5)';
    ctx.shadowBlur = 8;
    ctx.fillStyle = '#3b82f6';
    ctx.beginPath();
    ctx.arc(centerX, centerY, 4, 0, Math.PI * 2);
    ctx.fill();

    // Uniform blue bob
    ctx.shadowBlur = 12;
    ctx.fillStyle = '#3b82f6';
    ctx.beginPath();
    ctx.arc(bobX, bobY, 12, 0, Math.PI * 2);
    ctx.fill();

    ctx.shadowBlur = 0;

    // Subtle trail effect
    ctx.globalAlpha = 0.1;
    for (let i = 0; i < 5; i++) {
      const trailTime = time - i * 0.1;
      const trailAngle = Math.sin(trailTime * 1.5) * 0.6;
      const trailX = centerX + Math.sin(trailAngle) * length;
      const trailY = centerY + Math.cos(trailAngle) * length;
      
      ctx.fillStyle = '#60a5fa';
      ctx.beginPath();
      ctx.arc(trailX, trailY, 3 - i * 0.4, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  };

  const drawEnhancedWave = (ctx: CanvasRenderingContext2D, rect: DOMRect, time: number) => {
    const amplitude = rect.height * 0.12;
    const frequency = 0.015;
    const speed = 2;
    const centerY = rect.height / 2;

    // Primary wave with uniform blue styling
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.9)';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.shadowColor = 'rgba(59, 130, 246, 0.4)';
    ctx.shadowBlur = 8;
    
    ctx.beginPath();
    for (let x = 0; x < rect.width; x += 2) {
      const y = centerY + amplitude * Math.sin(frequency * x + time * speed);
      if (x === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Secondary wave with subtle variation
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.6)';
    ctx.lineWidth = 2;
    ctx.shadowBlur = 4;
    
    ctx.beginPath();
    for (let x = 0; x < rect.width; x += 2) {
      const y1 = amplitude * Math.sin(frequency * x + time * speed);
      const y2 = amplitude * 0.5 * Math.sin(frequency * x * 1.3 - time * speed * 0.7);
      const y = centerY + (y1 + y2);
      
      if (x === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    ctx.shadowBlur = 0;
  };

  const drawEnhancedOrbit = (ctx: CanvasRenderingContext2D, rect: DOMRect, time: number) => {
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    const orbitRadius = Math.min(rect.width, rect.height) * 0.22;
    
    // Central mass with uniform styling
    ctx.shadowColor = 'rgba(59, 130, 246, 0.6)';
    ctx.shadowBlur = 12;
    ctx.fillStyle = '#3b82f6';
    ctx.beginPath();
    ctx.arc(centerX, centerY, 16, 0, Math.PI * 2);
    ctx.fill();

    // Orbit path
    ctx.strokeStyle = 'rgba(100, 116, 139, 0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([6, 4]);
    ctx.shadowBlur = 3;
    ctx.beginPath();
    ctx.arc(centerX, centerY, orbitRadius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Orbiting object
    const angle = time * 1.2;
    const objectX = centerX + Math.cos(angle) * orbitRadius;
    const objectY = centerY + Math.sin(angle) * orbitRadius;

    ctx.shadowColor = 'rgba(59, 130, 246, 0.6)';
    ctx.shadowBlur = 10;
    ctx.fillStyle = '#60a5fa';
    ctx.beginPath();
    ctx.arc(objectX, objectY, 8, 0, Math.PI * 2);
    ctx.fill();

    // Velocity vector
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.shadowBlur = 4;
    ctx.beginPath();
    ctx.moveTo(objectX, objectY);
    ctx.lineTo(objectX - Math.sin(angle) * 20, objectY + Math.cos(angle) * 20);
    ctx.stroke();

    ctx.shadowBlur = 0;
  };

  const drawEnhancedField = (ctx: CanvasRenderingContext2D, rect: DOMRect, time: number) => {
    const gridSize = 20;
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;

    // Field vectors with uniform blue styling
    for (let x = gridSize; x < rect.width; x += gridSize) {
      for (let y = gridSize; y < rect.height; y += gridSize) {
        const dx = x - centerX;
        const dy = y - centerY;
        const distance = Math.sqrt(dx * dx + dy * dy) + 1;
        const strength = Math.min(200 / distance, 12);
        
        const fieldX = (dx / distance) * strength * 0.6;
        const fieldY = (dy / distance) * strength * 0.6;
        
        const oscillation = Math.sin(time * 2 + distance * 0.03) * 0.3;
        const alpha = Math.min(strength / 15, 0.8);
        
        ctx.strokeStyle = `rgba(59, 130, 246, ${alpha})`;
        ctx.lineWidth = 1.5;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + fieldX + oscillation, y + fieldY + oscillation);
        ctx.stroke();

        // Arrow heads
        const arrowLength = 3;
        const arrowAngle = Math.atan2(fieldY, fieldX);
        
        ctx.beginPath();
        ctx.moveTo(x + fieldX, y + fieldY);
        ctx.lineTo(
          x + fieldX - arrowLength * Math.cos(arrowAngle - Math.PI / 6),
          y + fieldY - arrowLength * Math.sin(arrowAngle - Math.PI / 6)
        );
        ctx.moveTo(x + fieldX, y + fieldY);
        ctx.lineTo(
          x + fieldX - arrowLength * Math.cos(arrowAngle + Math.PI / 6),
          y + fieldY - arrowLength * Math.sin(arrowAngle + Math.PI / 6)
        );
        ctx.stroke();
      }
    }

    // Central charge with subtle pulsing
    const pulseScale = 1 + Math.sin(time * 3) * 0.15;
    ctx.shadowColor = 'rgba(59, 130, 246, 0.6)';
    ctx.shadowBlur = 15 * pulseScale;
    ctx.fillStyle = '#3b82f6';
    ctx.beginPath();
    ctx.arc(centerX, centerY, 10 * pulseScale, 0, Math.PI * 2);
    ctx.fill();

    ctx.shadowBlur = 0;
  };

  const getTypeIcon = () => {
    switch (type) {
      case 'pendulum': return Settings;
      case 'wave': return Waves;
      case 'orbit': return Atom;
      case 'field': return Zap;
      default: return Settings;
    }
  };

  const getTypeColor = () => {
    // Uniform color scheme with subtle variations
    const baseColor = '#3b82f6'; // Blue base
    const glowColor = 'rgba(59, 130, 246, 0.25)';
    
    return { main: baseColor, glow: glowColor };
  };

  const handleReset = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setTime(0);
  }, []);

  const handlePlayPause = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    console.log('Play/Pause button clicked, current state:', isPlaying, 'new state will be:', !isPlaying);
    setIsPlaying(!isPlaying);
  }, [isPlaying]);

  const TypeIcon = getTypeIcon();
  const colors = getTypeColor();

  if (prefersReducedMotion) {
    return (
      <Card className={`w-full rounded-2xl overflow-hidden bg-gradient-to-br from-slate-900/50 via-slate-800/40 to-slate-900/50 backdrop-blur-xl border border-slate-600/40 shadow-xl ${className}`}>
        <div className="p-6 h-full flex flex-col justify-center items-center min-h-[280px] lg:min-h-[320px]">
          <div className="text-center space-y-4">
            <div
              className="w-16 h-16 mx-auto rounded-2xl flex items-center justify-center border-2 shadow-lg"
              style={{
                backgroundColor: `${colors.main}20`,
                borderColor: colors.main,
                boxShadow: `0 4px 20px ${colors.glow}`
              }}
            >
              <TypeIcon className="w-8 h-8" style={{ color: colors.main }} />
            </div>
            <h3 className="text-lg font-bold capitalize text-slate-200">
              {type} Simulation
            </h3>
            <p className="text-sm text-slate-400 font-medium">
              Animation disabled (reduced motion preference)
            </p>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card 
      className={`w-full rounded-2xl overflow-hidden group bg-gradient-to-br from-slate-900/70 via-slate-800/50 to-slate-900/70 backdrop-blur-xl border border-slate-600/40 shadow-xl transition-all duration-700 ${className} ${
        !prefersReducedMotion ? 'hover:shadow-2xl hover:border-slate-500/60 hover:scale-[1.02] hover:-translate-y-1' : ''
      }`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Enhanced glow overlay */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-0 group-hover:opacity-40 transition-opacity duration-700"
        style={{
          background: `radial-gradient(500px 200px at 50% 0%, ${colors.glow}, transparent 70%)`
        }}
      />

      <div className="relative p-6 h-full flex flex-col">
        {/* Enhanced header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div
              className={`w-12 h-12 rounded-xl flex items-center justify-center border-2 shadow-lg transition-all duration-500 ${
                isHovered ? 'scale-110 rotate-12' : ''
              }`}
              style={{
                backgroundColor: `${colors.main}20`,
                borderColor: colors.main,
                boxShadow: `0 4px 20px ${colors.glow}`
              }}
            >
              <TypeIcon className="w-6 h-6 transition-all duration-300" style={{ color: colors.main }} />
            </div>
            <div>
              <h3 className="text-lg font-bold capitalize text-slate-200 mb-1">
                {type} Simulation
              </h3>
              <div className="flex items-center gap-2">
                {isPlaying && (
                  <div className="flex items-center gap-1">
                    <div 
                      className="w-2 h-2 rounded-full animate-pulse" 
                      style={{ backgroundColor: '#10b981' }}
                    />
                    <span className="text-xs text-emerald-400 font-medium">Running</span>
                  </div>
                )}
              </div>
            </div>
          </div>
          
          {/* Enhanced control buttons */}
          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handleReset}
              className={`h-10 w-10 p-0 border-2 transition-all duration-300 ${
                !prefersReducedMotion ? 'hover:scale-110 hover:rotate-180' : ''
              }`}
              style={{
                borderColor: colors.main,
                color: colors.main,
                backgroundColor: `${colors.main}10`
              }}
              aria-label="Reset simulation"
            >
              <RotateCcw className="w-5 h-5" />
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handlePlayPause}
              className={`h-10 w-10 p-0 border-2 transition-all duration-300 ${
                !prefersReducedMotion ? 'hover:scale-110' : ''
              }`}
              style={{
                borderColor: '#10b981',
                color: '#10b981',
                backgroundColor: '#10b98110'
              }}
              aria-label={isPlaying ? 'Pause simulation' : 'Play simulation'}
            >
              {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
            </Button>
          </div>
        </div>

        {/* Enhanced canvas container with proper centering */}
        <div className="relative flex-1 rounded-xl overflow-hidden border border-slate-600/30 min-h-[200px] lg:min-h-[240px] flex items-center justify-center">
          <canvas
            ref={canvasRef}
            className="w-full h-full block bg-gradient-to-br from-slate-950/90 to-slate-900/70"
            style={{
              maxWidth: '100%',
              maxHeight: '100%',
              objectFit: 'contain'
            }}
          />
          
          {/* Subtle overlay gradient */}
          <div
            aria-hidden
            className="pointer-events-none absolute inset-0 opacity-10"
            style={{
              background: `radial-gradient(400px 200px at 50% 50%, ${colors.glow}, transparent 60%)`
            }}
          />
        </div>

        {/* Enhanced status bar */}
        <div 
          className="mt-4 text-center py-2 px-4 rounded-lg border transition-all duration-300"
          style={{
            backgroundColor: `${colors.main}08`,
            borderColor: `${colors.main}30`,
            color: colors.main
          }}
        >
          <div className="flex items-center justify-center gap-3 text-sm font-medium">
            <span>⏱️ Time: {time.toFixed(1)}s</span>
            <div className="w-1 h-1 rounded-full bg-current opacity-50" />
            <span className="flex items-center gap-1">
              {isPlaying ? (
                <>
                  <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  Running
                </>
              ) : (
                <>
                  <div className="w-2 h-2 rounded-full bg-red-400" />
                  Paused
                </>
              )}
            </span>
          </div>
        </div>
      </div>
    </Card>
  );
};