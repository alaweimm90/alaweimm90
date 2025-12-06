import React, { useEffect, useRef, useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  Settings,
  Zap,
  Atom,
  Waves
} from 'lucide-react';
import { usePrefersReducedMotion } from '@/lib/accessibility-utils';
import { useDomainTheme } from '@/components/DomainThemeProvider';
interface PhysicsVisualizationProps {
  type: 'pendulum' | 'wave' | 'orbit' | 'field';
  className?: string;
  autoPlay?: boolean;
}

export const PhysicsVisualization: React.FC<PhysicsVisualizationProps> = ({
  type,
  className = '',
  autoPlay = true
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();
  const [isPlaying, setIsPlaying] = useState(autoPlay);
  const [time, setTime] = useState(0);
  const prefersReducedMotion = usePrefersReducedMotion();
  const { currentDomain } = useDomainTheme();
  useEffect(() => {
    if (prefersReducedMotion || !isPlaying) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = 300;
    canvas.height = 200;

    const animate = (currentTime: number) => {
      const deltaTime = currentTime * 0.001; // Convert to seconds
      setTime(deltaTime);

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();

      switch (type) {
        case 'pendulum':
          drawPendulum(ctx, canvas, deltaTime);
          break;
        case 'wave':
          drawWave(ctx, canvas, deltaTime);
          break;
        case 'orbit':
          drawOrbit(ctx, canvas, deltaTime);
          break;
        case 'field':
          drawField(ctx, canvas, deltaTime);
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

  const drawPendulum = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, time: number) => {
    const centerX = canvas.width / 2;
    const centerY = 30;
    const length = 120;
    const angle = Math.sin(time * 2) * 0.5; // Simple harmonic motion
    
    const bobX = centerX + Math.sin(angle) * length;
    const bobY = centerY + Math.cos(angle) * length;

    // Draw string
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(bobX, bobY);
    ctx.stroke();

    // Draw pivot
    ctx.fillStyle = '#374151';
    ctx.beginPath();
    ctx.arc(centerX, centerY, 5, 0, Math.PI * 2);
    ctx.fill();

    // Draw bob with glow
    const gradient = ctx.createRadialGradient(bobX, bobY, 0, bobX, bobY, 15);
    gradient.addColorStop(0, '#3b82f6');
    gradient.addColorStop(0.7, '#1d4ed8');
    gradient.addColorStop(1, 'rgba(29, 78, 216, 0.3)');
    
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(bobX, bobY, 15, 0, Math.PI * 2);
    ctx.fill();

    // Add trail effect
    ctx.globalAlpha = 0.1;
    for (let i = 0; i < 10; i++) {
      const trailTime = time - i * 0.05;
      const trailAngle = Math.sin(trailTime * 2) * 0.5;
      const trailX = centerX + Math.sin(trailAngle) * length;
      const trailY = centerY + Math.cos(trailAngle) * length;
      
      ctx.fillStyle = '#3b82f6';
      ctx.beginPath();
      ctx.arc(trailX, trailY, 3, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  };

  const drawWave = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, time: number) => {
    const amplitude = 30;
    const frequency = 0.02;
    const speed = 2;

    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 3;
    ctx.beginPath();

    for (let x = 0; x < canvas.width; x += 2) {
      const y = canvas.height / 2 + amplitude * Math.sin(frequency * x + time * speed);
      if (x === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Draw interference pattern
    ctx.strokeStyle = '#8b5cf6';
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (let x = 0; x < canvas.width; x += 2) {
      const y1 = amplitude * Math.sin(frequency * x + time * speed);
      const y2 = amplitude * 0.7 * Math.sin(frequency * x * 1.3 - time * speed * 0.8);
      const y = canvas.height / 2 + (y1 + y2);
      
      if (x === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  };

  const drawOrbit = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, time: number) => {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const orbitRadius = 60;
    
    // Draw central mass
    const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, 20);
    gradient.addColorStop(0, '#fbbf24');
    gradient.addColorStop(0.7, '#f59e0b');
    gradient.addColorStop(1, 'rgba(245, 158, 11, 0.3)');
    
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(centerX, centerY, 20, 0, Math.PI * 2);
    ctx.fill();

    // Draw orbit path
    ctx.strokeStyle = 'rgba(100, 116, 139, 0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.arc(centerX, centerY, orbitRadius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw orbiting object
    const angle = time * 1.5;
    const objectX = centerX + Math.cos(angle) * orbitRadius;
    const objectY = centerY + Math.sin(angle) * orbitRadius;

    const objectGradient = ctx.createRadialGradient(objectX, objectY, 0, objectX, objectY, 8);
    objectGradient.addColorStop(0, '#3b82f6');
    objectGradient.addColorStop(0.7, '#1d4ed8');
    objectGradient.addColorStop(1, 'rgba(29, 78, 216, 0.3)');
    
    ctx.fillStyle = objectGradient;
    ctx.beginPath();
    ctx.arc(objectX, objectY, 8, 0, Math.PI * 2);
    ctx.fill();

    // Draw velocity vector
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(objectX, objectY);
    ctx.lineTo(objectX - Math.sin(angle) * 20, objectY + Math.cos(angle) * 20);
    ctx.stroke();
  };

  const drawField = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, time: number) => {
    const gridSize = 20;
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    for (let x = gridSize; x < canvas.width; x += gridSize) {
      for (let y = gridSize; y < canvas.height; y += gridSize) {
        const dx = x - centerX;
        const dy = y - centerY;
        const distance = Math.sqrt(dx * dx + dy * dy) + 1;
        const strength = 200 / distance;
        
        const fieldX = (dx / distance) * strength * 0.1;
        const fieldY = (dy / distance) * strength * 0.1;
        
        // Add time-based oscillation
        const oscillation = Math.sin(time * 3 + distance * 0.05) * 0.3;
        
        ctx.strokeStyle = `rgba(59, 130, 246, ${Math.min(strength / 50, 0.8)})`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + fieldX + oscillation, y + fieldY + oscillation);
        ctx.stroke();

        // Draw arrow head
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

    // Draw central charge
    const chargeGradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, 15);
    chargeGradient.addColorStop(0, '#ef4444');
    chargeGradient.addColorStop(0.7, '#dc2626');
    chargeGradient.addColorStop(1, 'rgba(220, 38, 38, 0.3)');
    
    ctx.fillStyle = chargeGradient;
    ctx.beginPath();
    ctx.arc(centerX, centerY, 10, 0, Math.PI * 2);
    ctx.fill();
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

  const TypeIcon = getTypeIcon();

  if (prefersReducedMotion) {
    return (
      <Card 
        variant="physics" 
        domain={currentDomain} 
        className={`w-full rounded-2xl overflow-hidden bg-gradient-to-br from-card/80 via-background/60 to-card/80 backdrop-blur-xl border border-primary/20 shadow-lg ${className}`}
      >
        <div className="p-5 md:p-6 h-full flex flex-col min-h-[260px] md:min-h-[300px]">
          <div className="flex items-center justify-center flex-1 bg-muted/30 rounded-xl border border-muted/40">
            <div className="text-center space-y-3">
              <div
                className="w-12 h-12 mx-auto rounded-xl flex items-center justify-center border-2 shadow-lg"
                style={{
                  backgroundColor: `hsl(var(--semantic-domain-${currentDomain}) / 0.15)`,
                  borderColor: `hsl(var(--semantic-domain-${currentDomain}) / 0.40)`,
                  boxShadow: `0 4px 12px hsl(var(--semantic-domain-${currentDomain}) / 0.25)`
                }}
              >
                <TypeIcon className="w-6 h-6" style={{ color: `hsl(var(--semantic-domain-${currentDomain}))` }} />
              </div>
              <p className="text-sm text-muted-foreground font-medium">
                Animation disabled (reduced motion preference)
              </p>
            </div>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card 
      className={`w-full rounded-3xl overflow-hidden group bg-gradient-to-br from-slate-900/40 via-slate-800/30 to-slate-900/40 backdrop-blur-xl border border-slate-600/30 shadow-xl ${className} ${
        !prefersReducedMotion ? 'hover:shadow-2xl hover:border-slate-500/50 transition-all duration-500' : ''
      }`}
    >
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-0 group-hover:opacity-60 transition-opacity duration-500"
        style={{
          background: `radial-gradient(500px 140px at 50% 0%, hsl(var(--semantic-domain-${currentDomain}) / 0.08), transparent 65%)`
        }}
      />
      <div className="relative p-6 h-full flex flex-col">
        <div className="space-y-4 flex-1">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3 min-w-0">
              <div className="shrink-0 w-10 h-10 rounded-xl flex items-center justify-center border border-slate-500/40 shadow-lg bg-gradient-to-br from-slate-700/50 to-slate-800/50">
                <TypeIcon className="w-5 h-5 text-cyan-300" />
              </div>
              <div className="min-w-0">
                <h3 className="text-base font-semibold capitalize text-balance leading-6 text-slate-200">
                  {type} Simulation
                </h3>
                <div className="flex items-center gap-2 mt-1">
                  {isPlaying && (
                    <span 
                      className="inline-flex h-2.5 w-2.5 rounded-full animate-pulse shadow-lg bg-green-400" 
                      aria-hidden 
                    />
                  )}
                </div>
              </div>
            </div>
            
            <div className="flex items-center gap-2 shrink-0">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  console.log('Reset button clicked');
                  setTime(0);
                }}
                className="h-10 w-10 p-0 border-2 border-orange-400/60 text-orange-400 bg-gradient-to-r from-orange-400/10 to-red-400/10 hover:from-orange-400/20 hover:to-red-400/20 transition-all duration-300 hover:scale-110"
                aria-label="Reset simulation"
              >
                <RotateCcw className="w-5 h-5" />
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  console.log('Play/Pause button clicked, current state:', isPlaying);
                  setIsPlaying(!isPlaying);
                }}
                className="h-10 w-10 p-0 border-2 border-green-400/60 text-green-400 bg-gradient-to-r from-green-400/10 to-blue-400/10 hover:from-green-400/20 hover:to-blue-400/20 transition-all duration-300 hover:scale-110"
                aria-label={isPlaying ? 'Pause simulation' : 'Play simulation'}
              >
                {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
              </Button>
            </div>
          </div>

          <div className="relative bg-gradient-to-br from-slate-900/50 to-slate-800/50 rounded-2xl overflow-hidden h-48 md:h-56 lg:h-64 flex-1 border-2 border-slate-600/30">
            <canvas
              ref={canvasRef}
              className="w-full h-full block"
              style={{ background: 'transparent' }}
            />
            <div
              aria-hidden
              className="pointer-events-none absolute inset-0 opacity-40"
              style={{
                background: `radial-gradient(500px 140px at 50% 0%, hsl(var(--semantic-domain-${currentDomain}) / 0.1), transparent 65%)`
              }}
            />
          </div>

          <div className="text-sm text-cyan-400 text-center font-bold bg-gradient-to-r from-slate-800/50 to-slate-700/50 rounded-lg p-2 border border-cyan-400/30">
            ⏱️ Time: {time.toFixed(1)}s • {isPlaying ? '🟢 Running' : '🔴 Paused'}
          </div>
        </div>
      </div>
    </Card>
  );
};