import React, { useEffect, useRef, useState } from 'react';
import { usePrefersReducedMotion } from '@/lib/accessibility-utils';

interface ParticleSystemProps {
  particleCount?: number;
  className?: string;
  style?: 'quantum' | 'molecular' | 'wave' | 'field';
}

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  opacity: number;
  hue: number;
  energy: number;
  phase: number;
}

export const ParticleSystem: React.FC<ParticleSystemProps> = ({
  particleCount = 50,
  className = '',
  style = 'quantum'
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();
  const particlesRef = useRef<Particle[]>([]);
  const timeRef = useRef(0);
  const prefersReducedMotion = usePrefersReducedMotion();
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => setIsVisible(entry.isIntersecting),
      { threshold: 0.1 }
    );
    
    if (canvasRef.current) {
      observer.observe(canvasRef.current);
    }
    
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (prefersReducedMotion || !isVisible) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const resizeCanvas = () => {
      canvas.width = canvas.offsetWidth * window.devicePixelRatio;
      canvas.height = canvas.offsetHeight * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Initialize particles
    const initParticles = () => {
      particlesRef.current = Array.from({ length: particleCount }, () => ({
        x: Math.random() * canvas.offsetWidth,
        y: Math.random() * canvas.offsetHeight,
        vx: (Math.random() - 0.5) * 2,
        vy: (Math.random() - 0.5) * 2,
        size: Math.random() * 3 + 1,
        opacity: Math.random() * 0.8 + 0.2,
        hue: Math.random() * 60 + 200, // Blue to cyan range
        energy: Math.random() * 10 + 5,
        phase: Math.random() * Math.PI * 2
      }));
    };

    initParticles();

    // Animation loop
    const animate = () => {
      ctx.clearRect(0, 0, canvas.offsetWidth, canvas.offsetHeight);
      timeRef.current += 0.02;

      particlesRef.current.forEach((particle, index) => {
        // Update particle based on style
        switch (style) {
          case 'quantum':
            // Quantum uncertainty principle simulation
            particle.x += particle.vx + Math.sin(timeRef.current + particle.phase) * 0.5;
            particle.y += particle.vy + Math.cos(timeRef.current + particle.phase) * 0.5;
            particle.opacity = 0.5 + 0.3 * Math.sin(timeRef.current * 2 + particle.phase);
            break;
            
          case 'molecular':
            // Brownian motion simulation
            particle.vx += (Math.random() - 0.5) * 0.1;
            particle.vy += (Math.random() - 0.5) * 0.1;
            particle.vx *= 0.99;
            particle.vy *= 0.99;
            particle.x += particle.vx;
            particle.y += particle.vy;
            break;
            
          case 'wave':
            // Wave interference pattern
            const waveX = Math.sin(timeRef.current + particle.x * 0.01) * 2;
            const waveY = Math.cos(timeRef.current + particle.y * 0.01) * 2;
            particle.x += particle.vx + waveX;
            particle.y += particle.vy + waveY;
            break;
            
          case 'field':
            // Electromagnetic field simulation
            const centerX = canvas.offsetWidth / 2;
            const centerY = canvas.offsetHeight / 2;
            const dx = particle.x - centerX;
            const dy = particle.y - centerY;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const force = particle.energy / (distance + 1);
            particle.vx += (dx / distance) * force * 0.001;
            particle.vy += (dy / distance) * force * 0.001;
            particle.x += particle.vx;
            particle.y += particle.vy;
            break;
        }

        // Boundary conditions (wrap around)
        if (particle.x < 0) particle.x = canvas.offsetWidth;
        if (particle.x > canvas.offsetWidth) particle.x = 0;
        if (particle.y < 0) particle.y = canvas.offsetHeight;
        if (particle.y > canvas.offsetHeight) particle.y = 0;

        // Draw particle
        ctx.save();
        ctx.globalAlpha = particle.opacity;
        
        // Create glow effect
        const gradient = ctx.createRadialGradient(
          particle.x, particle.y, 0,
          particle.x, particle.y, particle.size * 3
        );
        gradient.addColorStop(0, `hsl(${particle.hue}, 80%, 60%)`);
        gradient.addColorStop(0.5, `hsl(${particle.hue}, 60%, 40%)`);
        gradient.addColorStop(1, 'transparent');
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size * 3, 0, Math.PI * 2);
        ctx.fill();
        
        // Draw core
        ctx.fillStyle = `hsl(${particle.hue}, 90%, 70%)`;
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.restore();

        // Draw connections for quantum entanglement effect
        if (style === 'quantum') {
          particlesRef.current.forEach((otherParticle, otherIndex) => {
            if (index >= otherIndex) return;
            
            const dx = particle.x - otherParticle.x;
            const dy = particle.y - otherParticle.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < 100) {
              ctx.save();
              ctx.globalAlpha = (1 - distance / 100) * 0.3;
              ctx.strokeStyle = `hsl(${(particle.hue + otherParticle.hue) / 2}, 70%, 50%)`;
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.moveTo(particle.x, particle.y);
              ctx.lineTo(otherParticle.x, otherParticle.y);
              ctx.stroke();
              ctx.restore();
            }
          });
        }
      });

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [particleCount, style, prefersReducedMotion, isVisible]);

  if (prefersReducedMotion) {
    return (
      <div className={`absolute inset-0 ${className}`}>
        <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-accent/5 to-secondary/5" />
      </div>
    );
  }

  return (
    <canvas
      ref={canvasRef}
      className={`absolute inset-0 pointer-events-none ${className}`}
      style={{ background: 'transparent' }}
    />
  );
};