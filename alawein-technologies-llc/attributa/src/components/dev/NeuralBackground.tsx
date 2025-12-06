import React, { useEffect, useRef } from 'react';

interface Node {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  alpha: number;
}

interface Connection {
  from: Node;
  to: Node;
  alpha: number;
}

export default function NeuralBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();
  const nodesRef = useRef<Node[]>([]);
  const connectionsRef = useRef<Connection[]>([]);
  const lastFrameTime = useRef<number>(0);
  const isInView = useRef<boolean>(true);
  const prefersReducedMotion = useRef<boolean>(false);

  useEffect(() => {
    // Check for reduced motion preference
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    prefersReducedMotion.current = mediaQuery.matches;
    
    const handleMotionPreference = (e: MediaQueryListEvent) => {
      prefersReducedMotion.current = e.matches;
    };
    mediaQuery.addEventListener('change', handleMotionPreference);

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Performance optimization: throttle animation based on viewport visibility
    const observer = new IntersectionObserver(
      ([entry]) => {
        isInView.current = entry.isIntersecting;
      },
      { threshold: 0.1 }
    );
    observer.observe(canvas);

    // Initialize canvas size
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Initialize nodes
    const initNodes = () => {
      const nodeCount = Math.min(20, Math.floor(canvas.width * canvas.height / 15000));
      nodesRef.current = Array.from({ length: nodeCount }, () => ({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.2,
        vy: (Math.random() - 0.5) * 0.2,
        size: Math.random() * 2 + 1,
        alpha: Math.random() * 0.5 + 0.2
      }));
    };

    // Update connections based on distance
    const updateConnections = () => {
      connectionsRef.current = [];
      const maxDistance = 120;
      
      for (let i = 0; i < nodesRef.current.length; i++) {
        for (let j = i + 1; j < nodesRef.current.length; j++) {
          const nodeA = nodesRef.current[i];
          const nodeB = nodesRef.current[j];
          const distance = Math.sqrt(
            Math.pow(nodeA.x - nodeB.x, 2) + Math.pow(nodeA.y - nodeB.y, 2)
          );
          
          if (distance < maxDistance) {
            connectionsRef.current.push({
              from: nodeA,
              to: nodeB,
              alpha: Math.max(0, (maxDistance - distance) / maxDistance) * 0.3
            });
          }
        }
      }
    };

    // Animation loop with performance optimization
    const animate = (currentTime: number) => {
      // Skip animation if user prefers reduced motion
      if (prefersReducedMotion.current) {
        // Draw static frame once
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        nodesRef.current.forEach(node => {
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.size, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(59, 130, 246, 0.3)`;
          ctx.fill();
        });
        return;
      }
      
      // Skip frame if not in view or throttle to 30fps for performance
      if (!isInView.current || currentTime - lastFrameTime.current < 33) {
        animationFrameRef.current = requestAnimationFrame(animate);
        return;
      }
      
      lastFrameTime.current = currentTime;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Update node positions
      nodesRef.current.forEach(node => {
        node.x += node.vx;
        node.y += node.vy;
        
        // Bounce off edges
        if (node.x < 0 || node.x > canvas.width) node.vx *= -1;
        if (node.y < 0 || node.y > canvas.height) node.vy *= -1;
        
        // Keep in bounds
        node.x = Math.max(0, Math.min(canvas.width, node.x));
        node.y = Math.max(0, Math.min(canvas.height, node.y));
      });

      updateConnections();

      // Draw connections
      connectionsRef.current.forEach(connection => {
        ctx.beginPath();
        ctx.moveTo(connection.from.x, connection.from.y);
        ctx.lineTo(connection.to.x, connection.to.y);
        ctx.strokeStyle = `rgba(59, 130, 246, ${connection.alpha})`;
        ctx.lineWidth = 0.8;
        ctx.stroke();
      });

      // Draw nodes
      nodesRef.current.forEach(node => {
        const gradient = ctx.createRadialGradient(
          node.x, node.y, 0,
          node.x, node.y, node.size * 2
        );
        gradient.addColorStop(0, `rgba(6, 182, 212, ${node.alpha})`);
        gradient.addColorStop(1, `rgba(6, 182, 212, 0)`);
        
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.size, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();
      });

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    initNodes();
    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      mediaQuery.removeEventListener('change', handleMotionPreference);
      observer.disconnect();
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none z-[-1] opacity-10"
      style={{ background: 'transparent' }}
    />
  );
}
