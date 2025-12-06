import { useEffect, useRef } from 'react';

interface Orb {
  x: number;
  y: number;
  radius: number;
  dx: number;
  dy: number;
  opacity: number;
}

export const useCanvasAnimation = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resizeCanvas = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Initialize floating orbs
    const orbs: Orb[] = Array.from({ length: 8 }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      radius: Math.random() * 40 + 20,
      dx: (Math.random() - 0.5) * 0.5,
      dy: (Math.random() - 0.5) * 0.5,
      opacity: Math.random() * 0.15 + 0.05,
    }));

    let animationFrameId: number;

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      orbs.forEach(orb => {
        orb.x += orb.dx;
        orb.y += orb.dy;

        if (orb.x < 0 || orb.x > canvas.width) orb.dx *= -1;
        if (orb.y < 0 || orb.y > canvas.height) orb.dy *= -1;

        const gradient = ctx.createRadialGradient(orb.x, orb.y, 0, orb.x, orb.y, orb.radius);
        gradient.addColorStop(0, `rgba(193, 160, 96, ${orb.opacity})`);
        gradient.addColorStop(1, 'rgba(193, 160, 96, 0)');

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(orb.x, orb.y, orb.radius, 0, Math.PI * 2);
        ctx.fill();
      });

      animationFrameId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return canvasRef;
};
