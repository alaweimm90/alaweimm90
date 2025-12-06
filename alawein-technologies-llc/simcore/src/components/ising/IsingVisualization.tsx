import React, { useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

interface IsingSpinMeshProps {
  lattice: number[][];
  domains?: number[][];
  size: number;
  showDomains: boolean;
  scaleFactor?: number;
}

const IsingSpinMesh: React.FC<IsingSpinMeshProps> = ({ 
  lattice, 
  domains, 
  size, 
  showDomains,
  scaleFactor = 1 
}) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const materialRef = useRef<THREE.MeshPhongMaterial>(null);

  useEffect(() => {
    if (!meshRef.current) return;

    const tempObject = new THREE.Object3D();
    const colors = new Float32Array(size * size * 3);
    
    // Calculate scaling for better visibility
    const spacing = scaleFactor;
    const sphereRadius = Math.min(0.4 * spacing, 0.8);
    
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const index = i * size + j;
        
        // Position with proper spacing
        tempObject.position.set(
          (i - size/2) * spacing, 
          (j - size/2) * spacing, 
          0
        );
        
        // Scale based on spin value for visual effect
        const scale = lattice[i][j] > 0 ? 1.0 : 0.8;
        tempObject.scale.setScalar(scale);
        
        tempObject.updateMatrix();
        meshRef.current.setMatrixAt(index, tempObject.matrix);
        
        let color: THREE.Color;
        if (showDomains && domains) {
          // Color by domain ID with better contrast
          const hue = (domains[i][j] * 137.5) % 360; // Golden angle
          const saturation = 0.8;
          const lightness = lattice[i][j] > 0 ? 0.6 : 0.4;
          color = new THREE.Color().setHSL(hue / 360, saturation, lightness);
        } else {
          // Color by spin: bright red for +1, bright blue for -1
          color = lattice[i][j] > 0 
            ? new THREE.Color(0xff3333) // Bright red
            : new THREE.Color(0x3333ff); // Bright blue
        }
        
        colors[index * 3] = color.r;
        colors[index * 3 + 1] = color.g;
        colors[index * 3 + 2] = color.b;
      }
    }
    
    meshRef.current.instanceMatrix.needsUpdate = true;
    meshRef.current.geometry.setAttribute('color', new THREE.InstancedBufferAttribute(colors, 3));
    
    // Update sphere geometry radius
    if (meshRef.current.geometry) {
      meshRef.current.geometry.dispose();
      meshRef.current.geometry = new THREE.SphereGeometry(sphereRadius, 12, 8);
    }
  }, [lattice, domains, size, showDomains, scaleFactor]);

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, size * size]}>
      <sphereGeometry args={[0.3, 12, 8]} />
      <meshPhongMaterial 
        ref={materialRef}
        vertexColors 
        shininess={100}
        specular={0x222222}
      />
    </instancedMesh>
  );
};

interface IsingVisualizationProps {
  lattice: number[][];
  domains?: number[][];
  size: number;
  showDomains: boolean;
  height?: string;
}

export const IsingVisualization: React.FC<IsingVisualizationProps> = ({
  lattice,
  domains,
  size,
  showDomains,
  height = "400px"
}) => {
  // Calculate camera distance based on lattice size
  const cameraDistance = size * 1.5;
  const scaleFactor = Math.max(1, 64 / size); // Scale up smaller lattices

  return (
    <div style={{ height }} className="bg-background/5 rounded-lg border">
      <Canvas 
        camera={{ 
          position: [cameraDistance * 0.7, cameraDistance * 0.7, cameraDistance * 0.7], 
          fov: 50 
        }}
        gl={{ antialias: true, alpha: true }}
      >
        {/* Enhanced lighting setup */}
        <ambientLight intensity={0.4} />
        <directionalLight 
          position={[10, 10, 5]} 
          intensity={0.8} 
          castShadow
          shadow-mapSize-width={1024}
          shadow-mapSize-height={1024}
        />
        <directionalLight 
          position={[-10, -10, -5]} 
          intensity={0.3} 
          color="#4444ff"
        />
        
        {/* Grid plane for reference */}
        <gridHelper 
          args={[size * scaleFactor * 1.2, size]} 
          position={[0, 0, -1]} 
          rotation={[Math.PI / 2, 0, 0]}
          material={new THREE.LineBasicMaterial({ 
            color: 0x666666, 
            transparent: true, 
            opacity: 0.1 
          })}
        />
        
        {/* Spin lattice */}
        <IsingSpinMesh 
          lattice={lattice} 
          domains={domains}
          size={size} 
          showDomains={showDomains}
          scaleFactor={scaleFactor}
        />
        
        {/* Coordinate axes */}
        <axesHelper args={[size * scaleFactor * 0.6]} />
        
        <OrbitControls 
          enableZoom={true} 
          enablePan={true}
          enableRotate={true}
          autoRotate={false}
          autoRotateSpeed={0.5}
          minDistance={size * scaleFactor * 0.5}
          maxDistance={size * scaleFactor * 3}
        />
      </Canvas>
      
      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-background/80 backdrop-blur-sm rounded-lg p-3 text-sm">
        {showDomains ? (
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-gradient-to-r from-red-500 via-green-500 to-blue-500 rounded"></div>
            <span>Domains by cluster ID</span>
          </div>
        ) : (
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-red-500 rounded-full"></div>
              <span>Spin +1</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
              <span>Spin -1</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default IsingVisualization;