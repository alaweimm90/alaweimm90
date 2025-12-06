import React, { useEffect, useRef, useState } from 'react';
import { usePrefersReducedMotion } from '@/lib/accessibility-utils';

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  maxLife: number;
  type: 'quantum' | 'field' | 'wave';
  phase: number;
  frequency: number;
}

interface QuantumFieldVisualizationProps {
  className?: string;
  intensity?: number;
}

export const QuantumFieldVisualization: React.FC<QuantumFieldVisualizationProps> = ({ 
  className = '', 
  intensity = 1 
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const particlesRef = useRef<Particle[]>([]);
  const prefersReducedMotion = usePrefersReducedMotion();
  const [isVisible, setIsVisible] = useState(false);

  // Initialize particles
  const initializeParticles = (width: number, height: number) => {
    const particles: Particle[] = [];
    const numParticles = Math.min(150 * intensity, prefersReducedMotion ? 30 : 150);
    
    for (let i = 0; i < numParticles; i++) {
      particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        life: Math.random() * 1000,
        maxLife: 1000 + Math.random() * 2000,
        type: ['quantum', 'field', 'wave'][Math.floor(Math.random() * 3)] as 'quantum' | 'field' | 'wave',
        phase: Math.random() * Math.PI * 2,
        frequency: 0.01 + Math.random() * 0.02
      });
    }
    
    particlesRef.current = particles;
  };

  // Animation loop
  const animate = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { width, height } = canvas;
    
    // Clear canvas with subtle glow
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = 'rgba(15, 15, 35, 0.1)';
    ctx.fillRect(0, 0, width, height);

    // Update and draw particles
    const particles = particlesRef.current;
    const time = Date.now() * 0.001;

    particles.forEach((particle, index) => {
      // Update particle physics
      particle.life += 16; // Assume 60fps
      particle.phase += particle.frequency;
      
      // Quantum field influence
      const fieldInfluence = Math.sin(particle.phase) * 0.1;
      particle.vx += fieldInfluence * Math.cos(time * 0.5);
      particle.vy += fieldInfluence * Math.sin(time * 0.3);
      
      // Apply dampening
      particle.vx *= 0.99;
      particle.vy *= 0.99;
      
      // Update position
      particle.x += particle.vx;
      particle.y += particle.vy;
      
      // Boundary wrapping
      if (particle.x < 0) particle.x = width;
      if (particle.x > width) particle.x = 0;
      if (particle.y < 0) particle.y = height;
      if (particle.y > height) particle.y = 0;
      
      // Reset particle if life exceeded
      if (particle.life > particle.maxLife) {
        particle.life = 0;
        particle.x = Math.random() * width;
        particle.y = Math.random() * height;
      }
      
      // Calculate alpha based on life cycle
      const lifeRatio = particle.life / particle.maxLife;
      const alpha = Math.sin(lifeRatio * Math.PI) * 0.6;
      
      // Draw particle based on type
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      
      switch (particle.type) {
        case 'quantum':
          // Quantum dots with interference patterns
          const quantumSize = 1 + Math.sin(particle.phase) * 0.5;
          const gradient = ctx.createRadialGradient(
            particle.x, particle.y, 0,
            particle.x, particle.y, quantumSize * 3
          );
          gradient.addColorStop(0, `rgba(99, 102, 241, ${alpha})`);
          gradient.addColorStop(0.5, `rgba(139, 92, 246, ${alpha * 0.5})`);
          gradient.addColorStop(1, 'rgba(99, 102, 241, 0)');
          
          ctx.fillStyle = gradient;
          ctx.beginPath();
          ctx.arc(particle.x, particle.y, quantumSize * 3, 0, Math.PI * 2);
          ctx.fill();
          break;
          
        case 'field':
          // Field lines with wave motion
          const fieldLength = 20 + Math.sin(particle.phase) * 10;
          const angle = particle.phase + time;
          
          ctx.strokeStyle = `rgba(168, 85, 247, ${alpha * 0.7})`;
          ctx.lineWidth = 0.5;
          ctx.beginPath();
          ctx.moveTo(particle.x, particle.y);
          ctx.lineTo(
            particle.x + Math.cos(angle) * fieldLength,
            particle.y + Math.sin(angle) * fieldLength
          );
          ctx.stroke();
          break;
          
        case 'wave':
          // Wave packets
          const waveSize = 2 + Math.sin(particle.phase * 2) * 1;
          ctx.fillStyle = `rgba(34, 197, 94, ${alpha})`;
          ctx.beginPath();
          ctx.arc(particle.x, particle.y, waveSize, 0, Math.PI * 2);
          ctx.fill();
          
          // Wave interference
          ctx.strokeStyle = `rgba(34, 197, 94, ${alpha * 0.3})`;
          ctx.lineWidth = 0.3;
          ctx.beginPath();
          ctx.arc(particle.x, particle.y, waveSize * 5, 0, Math.PI * 2);
          ctx.stroke();
          break;
      }
      
      ctx.restore();
    });

    // Add quantum field mesh connections
    if (!prefersReducedMotion) {
      ctx.globalCompositeOperation = 'lighter';
      ctx.strokeStyle = 'rgba(99, 102, 241, 0.1)';
      ctx.lineWidth = 0.2;
      
      for (let i = 0; i < particles.length; i += 5) {
        for (let j = i + 1; j < particles.length; j += 5) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          
          if (distance < 100) {
            const alpha = (100 - distance) / 100 * 0.1;
            ctx.strokeStyle = `rgba(99, 102, 241, ${alpha})`;
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();
          }
        }
      }
    }

    if (isVisible) {
      animationRef.current = requestAnimationFrame(animate);
    }
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const updateSize = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
      }
      
      initializeParticles(canvas.width / window.devicePixelRatio, canvas.height / window.devicePixelRatio);
    };

    // Intersection Observer for performance
    const observer = new IntersectionObserver(
      ([entry]) => {
        setIsVisible(entry.isIntersecting);
      },
      { threshold: 0.1 }
    );

    observer.observe(canvas);
    updateSize();
    
    window.addEventListener('resize', updateSize);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      observer.disconnect();
      window.removeEventListener('resize', updateSize);
    };
  }, []);

  useEffect(() => {
    if (isVisible && !prefersReducedMotion) {
      animate();
    }
  }, [isVisible, prefersReducedMotion]);

  return (
    <canvas
      ref={canvasRef}
      className={`absolute inset-0 pointer-events-none -z-10 ${className}`}
      style={{ width: '100%', height: '100%' }}
      aria-hidden="true"
    />
  );
};