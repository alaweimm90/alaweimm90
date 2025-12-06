import React, { useRef, useEffect, useState } from 'react';
import { Activity, Atom, Zap } from 'lucide-react';

interface QuantumState {
  theta: number;  // Polar angle (0 to π)
  phi: number;    // Azimuthal angle (0 to 2π)
  alpha: { real: number; imag: number };
  beta: { real: number; imag: number };
  entanglement: number;
  coherence: number;
}

interface AtomicBlochSphereProps {
  state?: QuantumState;
  showOrbitals?: boolean;
  animateTransition?: boolean;
  onStateChange?: (state: QuantumState) => void;
}

export const AtomicBlochSphere: React.FC<AtomicBlochSphereProps> = ({
  state = {
    theta: Math.PI / 4,
    phi: 0,
    alpha: { real: 0.707, imag: 0 },
    beta: { real: 0.707, imag: 0 },
    entanglement: 0,
    coherence: 0.95
  },
  showOrbitals = true,
  animateTransition = true,
  onStateChange
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const [isRotating, setIsRotating] = useState(false);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    };
    resize();

    let rotation = 0;
    let targetRotation = 0;

    const drawSphere = (time: number) => {
      const width = canvas.width / window.devicePixelRatio;
      const height = canvas.height / window.devicePixelRatio;
      const centerX = width / 2;
      const centerY = height / 2;
      const radius = Math.min(width, height) * 0.35;

      // Clear canvas
      ctx.clearRect(0, 0, width, height);

      // Smooth rotation
      if (isRotating) {
        targetRotation += 0.01;
      }
      rotation += (targetRotation - rotation) * 0.1;

      // Draw atomic orbitals (background)
      if (showOrbitals) {
        // S-orbital (spherical)
        ctx.save();
        ctx.globalAlpha = 0.1;
        ctx.strokeStyle = 'rgb(59, 130, 246)'; // orbital-s blue
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 4]);
        
        for (let i = 0; i < 3; i++) {
          ctx.beginPath();
          ctx.arc(centerX, centerY, radius * (1 + i * 0.15), 0, Math.PI * 2);
          ctx.stroke();
        }
        
        // P-orbital (dumbbell)
        ctx.strokeStyle = 'rgb(139, 92, 246)'; // orbital-p purple
        ctx.beginPath();
        ctx.ellipse(centerX - radius * 0.3, centerY, radius * 0.2, radius * 0.5, 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.ellipse(centerX + radius * 0.3, centerY, radius * 0.2, radius * 0.5, 0, 0, Math.PI * 2);
        ctx.stroke();
        
        ctx.restore();
      }

      // Draw main sphere
      ctx.save();
      
      // Sphere gradient
      const gradient = ctx.createRadialGradient(
        centerX - radius * 0.3, 
        centerY - radius * 0.3, 
        0,
        centerX, 
        centerY, 
        radius
      );
      gradient.addColorStop(0, 'rgba(30, 58, 138, 0.1)');
      gradient.addColorStop(0.5, 'rgba(30, 58, 138, 0.05)');
      gradient.addColorStop(1, 'rgba(30, 58, 138, 0.02)');
      
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
      ctx.fill();
      
      // Sphere outline
      ctx.strokeStyle = 'rgba(59, 130, 246, 0.3)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
      ctx.stroke();
      
      // Draw coordinate circles
      ctx.strokeStyle = 'rgba(100, 116, 139, 0.2)';
      ctx.lineWidth = 1;
      
      // Equator (XY plane)
      ctx.beginPath();
      ctx.ellipse(centerX, centerY, radius, radius * Math.sin(rotation), 0, 0, Math.PI * 2);
      ctx.stroke();
      
      // Meridian (XZ plane)
      ctx.beginPath();
      ctx.ellipse(centerX, centerY, radius * Math.cos(rotation), radius, 0, 0, Math.PI * 2);
      ctx.stroke();
      
      // Meridian (YZ plane)
      ctx.save();
      ctx.translate(centerX, centerY);
      ctx.rotate(Math.PI / 2);
      ctx.translate(-centerX, -centerY);
      ctx.beginPath();
      ctx.ellipse(centerX, centerY, radius * Math.sin(rotation + Math.PI / 2), radius, 0, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();

      // Draw axes
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.4)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      
      // X-axis
      ctx.beginPath();
      ctx.moveTo(centerX - radius * 1.2, centerY);
      ctx.lineTo(centerX + radius * 1.2, centerY);
      ctx.stroke();
      
      // Y-axis
      ctx.beginPath();
      ctx.moveTo(centerX, centerY - radius * 1.2);
      ctx.lineTo(centerX, centerY + radius * 1.2);
      ctx.stroke();
      
      ctx.setLineDash([]);
      
      // Draw state vector
      const stateX = centerX + radius * Math.sin(state.theta) * Math.cos(state.phi + rotation);
      const stateY = centerY - radius * Math.cos(state.theta);
      const stateZ = radius * Math.sin(state.theta) * Math.sin(state.phi + rotation);
      
      // State vector line
      ctx.strokeStyle = 'rgb(6, 182, 212)'; // photon-blue
      ctx.lineWidth = 2;
      ctx.shadowBlur = 10;
      ctx.shadowColor = 'rgba(6, 182, 212, 0.5)';
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(stateX, stateY);
      ctx.stroke();
      ctx.shadowBlur = 0;
      
      // State point
      const pulseSize = 4 + Math.sin(time * 0.003) * 2;
      ctx.fillStyle = 'rgb(6, 182, 212)';
      ctx.beginPath();
      ctx.arc(stateX, stateY, pulseSize, 0, Math.PI * 2);
      ctx.fill();
      
      // Electron cloud visualization
      if (state.coherence > 0.5) {
        ctx.save();
        ctx.globalAlpha = state.coherence * 0.3;
        const cloudGradient = ctx.createRadialGradient(stateX, stateY, 0, stateX, stateY, 20);
        cloudGradient.addColorStop(0, 'rgba(6, 182, 212, 0.8)');
        cloudGradient.addColorStop(1, 'rgba(6, 182, 212, 0)');
        ctx.fillStyle = cloudGradient;
        ctx.beginPath();
        ctx.arc(stateX, stateY, 20, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
      
      // Draw quantum state labels
      ctx.fillStyle = 'rgba(148, 163, 184, 0.8)';
      ctx.font = '11px JetBrains Mono';
      
      // |0⟩ state (north pole)
      ctx.fillText('|0⟩', centerX - 8, centerY - radius - 10);
      
      // |1⟩ state (south pole)
      ctx.fillText('|1⟩', centerX - 8, centerY + radius + 20);
      
      // |+⟩ state (positive X)
      ctx.fillText('|+⟩', centerX + radius + 10, centerY + 3);
      
      // |−⟩ state (negative X)
      ctx.fillText('|−⟩', centerX - radius - 25, centerY + 3);
      
      // |i⟩ state (positive Y)
      if (Math.abs(Math.sin(rotation)) > 0.5) {
        ctx.fillText('|i⟩', centerX + radius * 0.7 * Math.sin(rotation), centerY + 3);
      }
      
      ctx.restore();

      // Draw entanglement visualization (if entangled)
      if (state.entanglement > 0.1) {
        ctx.save();
        ctx.strokeStyle = `rgba(236, 72, 153, ${state.entanglement * 0.5})`; // quantum-entangled pink
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        
        // Draw entanglement bonds
        for (let i = 0; i < 3; i++) {
          const angle = (Math.PI * 2 / 3) * i + time * 0.001;
          const x1 = centerX + radius * 0.8 * Math.cos(angle);
          const y1 = centerY + radius * 0.8 * Math.sin(angle);
          const x2 = centerX - radius * 0.8 * Math.cos(angle);
          const y2 = centerY - radius * 0.8 * Math.sin(angle);
          
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.quadraticCurveTo(centerX, centerY, x2, y2);
          ctx.stroke();
        }
        
        ctx.restore();
      }

      animationRef.current = requestAnimationFrame(drawSphere);
    };

    drawSphere(0);

    // Handle mouse interaction
    const handleMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      setMousePos({ x, y });
      
      // Update rotation based on mouse position
      if (isRotating) {
        targetRotation = (x / rect.width - 0.5) * Math.PI;
      }
    };

    const handleClick = () => {
      setIsRotating(!isRotating);
    };

    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('click', handleClick);
    window.addEventListener('resize', resize);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      canvas.removeEventListener('mousemove', handleMouseMove);
      canvas.removeEventListener('click', handleClick);
      window.removeEventListener('resize', resize);
    };
  }, [state, showOrbitals, isRotating]);

  return (
    <div className="relative w-full h-full">
      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-pointer"
        style={{ minHeight: '300px' }}
      />
      
      {/* Quantum State Information Overlay */}
      <div className="absolute bottom-4 left-4 right-4 quantum-glass p-3 rounded-lg">
        <div className="grid grid-cols-3 gap-3 text-center">
          <div>
            <div className="flex items-center justify-center gap-1 mb-1">
              <Atom className="w-3 h-3 text-slate-400" />
              <span className="text-xs text-slate-400">α</span>
            </div>
            <div className="quantum-mono text-xs text-white">
              {state.alpha.real.toFixed(3)} + {state.alpha.imag.toFixed(3)}i
            </div>
          </div>
          <div>
            <div className="flex items-center justify-center gap-1 mb-1">
              <Atom className="w-3 h-3 text-slate-400" />
              <span className="text-xs text-slate-400">β</span>
            </div>
            <div className="quantum-mono text-xs text-white">
              {state.beta.real.toFixed(3)} + {state.beta.imag.toFixed(3)}i
            </div>
          </div>
          <div>
            <div className="flex items-center justify-center gap-1 mb-1">
              <Zap className="w-3 h-3 state-coherent" />
              <span className="text-xs text-slate-400">Coherence</span>
            </div>
            <div className="quantum-mono text-xs state-coherent">
              {(state.coherence * 100).toFixed(0)}%
            </div>
          </div>
        </div>
      </div>
      
      {/* Rotation Indicator */}
      {isRotating && (
        <div className="absolute top-4 right-4 flex items-center gap-2 quantum-glass px-3 py-1.5 rounded-full">
          <Activity className="w-3 h-3 text-quantum-photon-blue animate-spin" />
          <span className="text-xs text-slate-300">Rotating</span>
        </div>
      )}
    </div>
  );
};