import React, { useRef, useEffect } from 'react';

// Minimal Three.js imports - only load core features needed for Bloch sphere
const loadThreeJSCore = async () => {
  const [sceneModule, cameraModule, rendererModule, colorModule] = await Promise.all([
    import('three/src/scenes/Scene.js'),
    import('three/src/cameras/PerspectiveCamera.js'),
    import('three/src/renderers/WebGLRenderer.js'),
    import('three/src/math/Color.js')
  ]);
  
  const [
    { SphereGeometry, MeshBasicMaterial, Mesh },
    { ArrowHelper, Vector3 }
  ] = await Promise.all([
    Promise.all([
      import('three/src/geometries/SphereGeometry.js').then(m => ({ SphereGeometry: m.SphereGeometry })),
      import('three/src/materials/MeshBasicMaterial.js').then(m => ({ MeshBasicMaterial: m.MeshBasicMaterial })),
      import('three/src/objects/Mesh.js').then(m => ({ Mesh: m.Mesh }))
    ]).then(([sphere, material, mesh]) => ({
      SphereGeometry: sphere.SphereGeometry,
      MeshBasicMaterial: material.MeshBasicMaterial,
      Mesh: mesh.Mesh
    })),
    Promise.all([
      import('three/src/helpers/ArrowHelper.js').then(m => ({ ArrowHelper: m.ArrowHelper })),
      import('three/src/math/Vector3.js').then(m => ({ Vector3: m.Vector3 }))
    ]).then(([arrow, vector]) => ({
      ArrowHelper: arrow.ArrowHelper,
      Vector3: vector.Vector3
    }))
  ]);

  return {
    Scene: sceneModule.Scene,
    PerspectiveCamera: cameraModule.PerspectiveCamera, 
    WebGLRenderer: rendererModule.WebGLRenderer,
    Color: colorModule.Color,
    SphereGeometry, MeshBasicMaterial, Mesh,
    ArrowHelper, Vector3
  };
};

interface BlochCoreProps {
  theta?: number;
  phi?: number;
  animationPhase?: number;
  isPaused?: boolean;
  className?: string;
}

const BlochCore: React.FC<BlochCoreProps> = ({
  theta = 0,
  phi = 0,
  animationPhase = 0,
  isPaused = false,
  className = ''
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<any>(null);
  const sceneRef = useRef<any>(null);

  useEffect(() => {
    let mounted = true;

    const initializeBlochSphere = async () => {
      if (!mountRef.current || !mounted) return;

      try {
        const ThreeJS = await loadThreeJSCore();
        
        // Create scene
        const scene = new ThreeJS.Scene();
        scene.background = new ThreeJS.Color(0x0f0f23);
        sceneRef.current = scene;

        // Create camera
        const camera = new ThreeJS.PerspectiveCamera(
          75,
          mountRef.current.clientWidth / mountRef.current.clientHeight,
          0.1,
          1000
        );
        camera.position.z = 3;

        // Create renderer
        const renderer = new ThreeJS.WebGLRenderer({ 
          antialias: true,
          alpha: true,
          powerPreference: 'low-power' // Better for mobile
        });
        renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // Limit pixel ratio for performance
        rendererRef.current = renderer;

        mountRef.current.appendChild(renderer.domElement);

        // Create Bloch sphere
        const sphereGeometry = new ThreeJS.SphereGeometry(1, 32, 32);
        const sphereMaterial = new ThreeJS.MeshBasicMaterial({ 
          color: 0x1e40af,
          wireframe: true,
          transparent: true,
          opacity: 0.3
        });
        const sphere = new ThreeJS.Mesh(sphereGeometry, sphereMaterial);
        scene.add(sphere);

        // Create state vector
        const stateVector = new ThreeJS.ArrowHelper(
          new ThreeJS.Vector3(0, 0, 1),
          new ThreeJS.Vector3(0, 0, 0),
          1,
          0x10b981
        );
        scene.add(stateVector);

        // Animation loop
        const animate = () => {
          if (!mounted || isPaused) return;

          // Update state vector based on quantum state
          const currentTheta = theta || animationPhase;
          const currentPhi = phi || animationPhase * 1.5;
          
          const x = Math.sin(currentTheta) * Math.cos(currentPhi);
          const y = Math.sin(currentTheta) * Math.sin(currentPhi);
          const z = Math.cos(currentTheta);

          stateVector.setDirection(new ThreeJS.Vector3(x, y, z));
          
          renderer.render(scene, camera);
          if (!isPaused) {
            requestAnimationFrame(animate);
          }
        };

        animate();

        // Handle resize
        const handleResize = () => {
          if (!mountRef.current || !mounted) return;
          
          const width = mountRef.current.clientWidth;
          const height = mountRef.current.clientHeight;
          
          camera.aspect = width / height;
          camera.updateProjectionMatrix();
          renderer.setSize(width, height);
        };

        window.addEventListener('resize', handleResize);

        return () => {
          window.removeEventListener('resize', handleResize);
          if (mountRef.current && renderer.domElement) {
            mountRef.current.removeChild(renderer.domElement);
          }
          renderer.dispose();
          sphereGeometry.dispose();
          sphereMaterial.dispose();
        };

      } catch (error) {
        console.error('Failed to load Bloch sphere:', error);
        if (mountRef.current && mounted) {
          mountRef.current.innerHTML = `
            <div class="flex items-center justify-center h-full bg-slate-900 rounded-lg border border-red-500/30">
              <div class="text-center p-6">
                <div class="text-red-400 mb-2">⚠️ 3D Visualization Error</div>
                <div class="text-slate-400 text-sm">WebGL not supported or failed to load</div>
              </div>
            </div>
          `;
        }
      }
    };

    initializeBlochSphere();

    return () => {
      mounted = false;
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
    };
  }, [theta, phi, animationPhase, isPaused]);

  return (
    <div 
      ref={mountRef} 
      className={`bloch-sphere-container w-full h-96 rounded-lg ${className}`}
      style={{ minHeight: '384px' }}
      role="img"
      aria-label="Interactive Bloch sphere showing quantum state visualization"
    />
  );
};

export default BlochCore;