import React, { useEffect, useRef } from 'react';

interface QuantumParticleFieldProps {
  className?: string;
  density?: 'low' | 'medium' | 'high';
  colors?: string[];
}

export const QuantumParticleField: React.FC<QuantumParticleFieldProps> = ({
  className = '',
  density = 'medium',
  colors = ['#3b82f6', '#8b5cf6', '#6366f1', '#06b6d4']
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const dimensionsRef = useRef({ width: 0, height: 0 });
  const particlesRef = useRef<Array<{
    x: number;
    y: number;
    vx: number;
    vy: number;
    radius: number;
    color: string;
    opacity: number;
    phase: number;
  }>>([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Check for reduced motion preference
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) return;

    const resizeCanvas = () => {
      const rect = canvas.getBoundingClientRect();
      const { width, height } = rect;
      
      // Cache dimensions to avoid repeated getBoundingClientRect calls
      dimensionsRef.current = { width, height };
      
      canvas.width = width * window.devicePixelRatio;
      canvas.height = height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
      canvas.style.width = width + 'px';
      canvas.style.height = height + 'px';
    };

    const createParticles = () => {
      const particleCount = {
        low: 15,
        medium: 25,
        high: 40
      }[density];

      particlesRef.current = [];
      
      const { width, height } = dimensionsRef.current;
      
      for (let i = 0; i < particleCount; i++) {
        particlesRef.current.push({
          x: Math.random() * width,
          y: Math.random() * height,
          vx: (Math.random() - 0.5) * 0.5,
          vy: (Math.random() - 0.5) * 0.5,
          radius: Math.random() * 2 + 1,
          color: colors[Math.floor(Math.random() * colors.length)],
          opacity: Math.random() * 0.8 + 0.2,
          phase: Math.random() * Math.PI * 2
        });
      }
    };

    const animate = (time: number) => {
      const { width, height } = dimensionsRef.current;
      ctx.clearRect(0, 0, width, height);

      particlesRef.current.forEach((particle, index) => {
        // Update position
        particle.x += particle.vx;
        particle.y += particle.vy;
        particle.phase += 0.02;

        // Wrap around edges
        if (particle.x < 0) particle.x = width;
        if (particle.x > width) particle.x = 0;
        if (particle.y < 0) particle.y = height;
        if (particle.y > height) particle.y = 0;

        // Quantum wave behavior
        const waveOffset = Math.sin(particle.phase) * 0.5;
        const finalOpacity = Math.max(0.1, particle.opacity + waveOffset * 0.3);
        const finalRadius = particle.radius + waveOffset * 0.5;

        // Draw particle
        ctx.save();
        ctx.globalAlpha = finalOpacity;
        ctx.fillStyle = particle.color;
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, finalRadius, 0, Math.PI * 2);
        ctx.fill();

        // Draw quantum glow
        const gradient = ctx.createRadialGradient(
          particle.x, particle.y, 0,
          particle.x, particle.y, finalRadius * 3
        );
        gradient.addColorStop(0, particle.color + '40');
        gradient.addColorStop(1, particle.color + '00');
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, finalRadius * 3, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();

        // Draw connections between nearby particles
        particlesRef.current.forEach((otherParticle, otherIndex) => {
          if (index >= otherIndex) return;

          const dx = particle.x - otherParticle.x;
          const dy = particle.y - otherParticle.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < 100) {
            ctx.save();
            ctx.strokeStyle = particle.color + Math.floor((1 - distance / 100) * 40).toString(16).padStart(2, '0');
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(particle.x, particle.y);
            ctx.lineTo(otherParticle.x, otherParticle.y);
            ctx.stroke();
            ctx.restore();
          }
        });
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    resizeCanvas();
    createParticles();
    animate(0);

    const handleResize = () => {
      resizeCanvas();
      createParticles();
    };

    window.addEventListener('resize', handleResize);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      window.removeEventListener('resize', handleResize);
    };
  }, [density, colors]);

  return (
    <canvas
      ref={canvasRef}
      className={`absolute inset-0 pointer-events-none ${className}`}
      style={{
        mixBlendMode: 'screen',
        opacity: 0.6
      }}
    />
  );
};