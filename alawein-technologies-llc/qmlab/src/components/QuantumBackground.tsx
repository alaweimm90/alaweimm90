import { useEffect, useRef } from 'react';
import { logger } from '@/lib/logger';

// Dynamic Three.js import to avoid loading on reduced motion
const loadThreeJS = async () => {
  const THREE = await import('three');
  return THREE;
};

interface QuantumBackgroundProps {
  className?: string;
  intensity?: 'low' | 'medium' | 'high';
}

export const QuantumBackground: React.FC<QuantumBackgroundProps> = ({ 
  className = "",
  intensity = 'medium'
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<{
    scene: any;
    camera: any;
    renderer: any;
    particles: any[];
    animationId?: number;
    isPaused: boolean;
  }>();

  useEffect(() => {
    if (!mountRef.current) return;

    // Check for reduced motion preference
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const isLowPowerDevice = /Mobile|Android|iPhone|iPad/.test(navigator.userAgent);
    
    // Skip Three.js entirely for reduced motion users
    if (prefersReducedMotion) {
      // Create a simple CSS-based fallback
      const mount = mountRef.current;
      const fallbackDiv = document.createElement('div');
      fallbackDiv.className = 'absolute inset-0 opacity-20 bg-gradient-to-br from-blue-900/10 via-purple-900/5 to-indigo-900/10';
      mount.appendChild(fallbackDiv);
      
      return () => {
        if (mount.contains(fallbackDiv)) {
          mount.removeChild(fallbackDiv);
        }
      };
    }

    // Load Three.js dynamically only when needed
    const initThreeJS = async () => {
      const THREE = await loadThreeJS();
      
      // Adjust intensity based on user preferences and device
      const effectiveIntensity = isLowPowerDevice ? 'low' : intensity;

      const mount = mountRef.current!;
      const width = mount.clientWidth;
      const height = mount.clientHeight;

    // Scene setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ 
      alpha: true, 
      antialias: false,
      powerPreference: 'low-power'
    });
    
    renderer.setSize(width, height);
    renderer.setClearColor(0x000000, 0);
    renderer.domElement.setAttribute('data-animated', (!prefersReducedMotion).toString());
    mount.appendChild(renderer.domElement);

    // Quantum particle systems
    const particles: THREE.Points[] = [];
    
    // Primary quantum field
    const createQuantumField = (count: number, color: number, size: number) => {
      const geometry = new THREE.BufferGeometry();
      const positions = new Float32Array(count * 3);
      const velocities = new Float32Array(count * 3);
      
      for (let i = 0; i < count; i++) {
        positions[i * 3] = (Math.random() - 0.5) * 200;
        positions[i * 3 + 1] = (Math.random() - 0.5) * 200;
        positions[i * 3 + 2] = (Math.random() - 0.5) * 200;
        
        velocities[i * 3] = (Math.random() - 0.5) * 0.02;
        velocities[i * 3 + 1] = (Math.random() - 0.5) * 0.02;
        velocities[i * 3 + 2] = (Math.random() - 0.5) * 0.02;
      }
      
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('velocity', new THREE.BufferAttribute(velocities, 3));
      
      const material = new THREE.PointsMaterial({
        color,
        size,
        transparent: true,
        opacity: 0.6,
        blending: THREE.AdditiveBlending,
        sizeAttenuation: true
      });
      
      return new THREE.Points(geometry, material);
    };

    // Intensity-based particle counts
    const particleCounts = {
      low: { primary: 50, secondary: 30, tertiary: 20 },
      medium: { primary: 100, secondary: 60, tertiary: 40 },
      high: { primary: 200, secondary: 120, tertiary: 80 }
    };

    const counts = particleCounts[effectiveIntensity];

    // Create quantum particle fields
    const primaryField = createQuantumField(counts.primary, 0x00BFFF, 2); // Electric cyan
    const secondaryField = createQuantumField(counts.secondary, 0x9D4EDD, 1.5); // Quantum purple  
    const tertiaryField = createQuantumField(counts.tertiary, 0x06FFA5, 1); // Quantum green

    particles.push(primaryField, secondaryField, tertiaryField);
    particles.forEach(p => scene.add(p));

    // Quantum wave rings
    const createQuantumRing = (radius: number, color: number, segments: number = 64) => {
      const geometry = new THREE.RingGeometry(radius - 0.5, radius, segments);
      const material = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.1,
        side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending
      });
      
      const ring = new THREE.Mesh(geometry, material);
      ring.rotation.x = Math.PI / 2;
      return ring;
    };

    const rings = [
      createQuantumRing(30, 0x00BFFF),
      createQuantumRing(50, 0x9D4EDD),
      createQuantumRing(70, 0x06FFA5)
    ];
    
    rings.forEach(ring => scene.add(ring));

    camera.position.z = 100;

    sceneRef.current = { scene, camera, renderer, particles, isPaused: prefersReducedMotion };

    // Animation loop
    let time = 0;
    const animate = () => {
      // Respect reduced motion and pause when out of view
      if (prefersReducedMotion || sceneRef.current?.isPaused) {
        sceneRef.current!.animationId = requestAnimationFrame(animate);
        return;
      }
      
      time += 0.01;
      
      // Animate particles with quantum behavior
      particles.forEach((particleSystem, index) => {
        const positions = particleSystem.geometry.attributes.position;
        const velocities = particleSystem.geometry.attributes.velocity;
        
        for (let i = 0; i < positions.count; i++) {
          // Quantum tunneling effect
          const x = positions.getX(i);
          const y = positions.getY(i);
          const z = positions.getZ(i);
          
          const vx = velocities.getX(i);
          const vy = velocities.getY(i);
          const vz = velocities.getZ(i);
          
          // Apply quantum wave function
          const wave = Math.sin(time + i * 0.1) * 0.5;
          
          positions.setX(i, x + vx + wave * 0.1);
          positions.setY(i, y + vy + Math.cos(time + i * 0.15) * 0.05);
          positions.setZ(i, z + vz);
          
          // Boundary wrapping
          if (Math.abs(positions.getX(i)) > 100) positions.setX(i, -positions.getX(i));
          if (Math.abs(positions.getY(i)) > 100) positions.setY(i, -positions.getY(i));
          if (Math.abs(positions.getZ(i)) > 100) positions.setZ(i, -positions.getZ(i));
        }
        
        positions.needsUpdate = true;
        
        // Rotate particle system
        particleSystem.rotation.y += 0.001 * (index + 1);
      });

      // Animate quantum rings
      rings.forEach((ring, index) => {
        ring.rotation.z += 0.002 * (index + 1);
        ring.material.opacity = 0.05 + Math.sin(time * 2 + index) * 0.03;
      });

      renderer.render(scene, camera);
      sceneRef.current!.animationId = requestAnimationFrame(animate);
    };

    animate();

    // Handle resize
    const handleResize = () => {
      if (!mount) return;
      const newWidth = mount.clientWidth;
      const newHeight = mount.clientHeight;
      
      camera.aspect = newWidth / newHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, newHeight);
    };

      window.addEventListener('resize', handleResize);

      // Cleanup function
      return () => {
        window.removeEventListener('resize', handleResize);
        if (sceneRef.current?.animationId) {
          cancelAnimationFrame(sceneRef.current.animationId);
        }
        if (mount && renderer.domElement) {
          mount.removeChild(renderer.domElement);
        }
        renderer.dispose();
      };
    };

    // Initialize Three.js asynchronously
    let cleanup: (() => void) | null = null;
    initThreeJS().then((cleanupFn) => {
      cleanup = cleanupFn;
    }).catch((error) => logger.error('Quantum background animation error', { error }));

    // Return cleanup function
    return () => {
      if (cleanup) cleanup();
    };
  }, [intensity]);

  return (
    <div 
      ref={mountRef} 
      className={`absolute inset-0 pointer-events-none ${className}`}
      style={{ zIndex: -1 }}
    />
  );
};