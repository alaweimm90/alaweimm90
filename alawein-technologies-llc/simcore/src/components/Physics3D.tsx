import React, { useRef, useEffect, useState, memo, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, Text } from '@react-three/drei';
import * as THREE from 'three';
import { UNIFIED_SCIENTIFIC_PLOT_SYSTEM } from '@/lib/scientific-plot-system';

// Theme-aware colors extracted from CSS custom properties
const useThemeColors = () => {
  return useMemo(() => {
    // Get computed styles from CSS custom properties
    const root = getComputedStyle(document.documentElement);
    
    return {
      primary: `hsl(${root.getPropertyValue('--primary').trim()})`,
      accent: `hsl(${root.getPropertyValue('--accent').trim()})`,
      muted: `hsl(${root.getPropertyValue('--muted-foreground').trim()})`,
      border: `hsl(${root.getPropertyValue('--border').trim()})`,
      // Fallback to scientific colors if CSS vars not available
      valence: UNIFIED_SCIENTIFIC_PLOT_SYSTEM.colors.physics.valence,
      conduction: UNIFIED_SCIENTIFIC_PLOT_SYSTEM.colors.physics.conduction,
      potential: UNIFIED_SCIENTIFIC_PLOT_SYSTEM.colors.physics.neutral,
    };
  }, []);
};

interface QuantumWavefunctionProps {
  wavefunction: Float32Array;
  potential: Float32Array;
  xRange: [number, number];
  animate?: boolean;
}

export const QuantumWavefunction: React.FC<QuantumWavefunctionProps> = memo(({
  wavefunction,
  potential,
  xRange,
  animate = true
}) => {
  const waveMeshRef = useRef<THREE.Mesh>(null);
  const potentialMeshRef = useRef<THREE.Mesh>(null);
  const themeColors = useThemeColors();
  
  useEffect(() => {
    if (!waveMeshRef.current || !potentialMeshRef.current) return;

    const points = wavefunction.length;
    const dx = (xRange[1] - xRange[0]) / (points - 1);
    
    // Create wave geometry
    const waveGeometry = new THREE.BufferGeometry();
    const wavePositions = new Float32Array(points * 3);
    const waveColors = new Float32Array(points * 3);
    
    // Create potential geometry
    const potentialGeometry = new THREE.BufferGeometry();
    const potentialPositions = new Float32Array(points * 3);
    
    for (let i = 0; i < points; i++) {
      const x = xRange[0] + i * dx;
      
      // Wave positions
      wavePositions[i * 3] = x;
      wavePositions[i * 3 + 1] = wavefunction[i] * 2; // Scale for visibility
      wavePositions[i * 3 + 2] = 0;
      
      // Wave colors (based on amplitude)
      const intensity = Math.abs(wavefunction[i]);
      waveColors[i * 3] = intensity;     // Red
      waveColors[i * 3 + 1] = 0.5;       // Green
      waveColors[i * 3 + 2] = 1 - intensity; // Blue
      
      // Potential positions
      potentialPositions[i * 3] = x;
      potentialPositions[i * 3 + 1] = potential[i] * 0.1; // Scale for visibility
      potentialPositions[i * 3 + 2] = 0;
    }
    
    waveGeometry.setAttribute('position', new THREE.BufferAttribute(wavePositions, 3));
    waveGeometry.setAttribute('color', new THREE.BufferAttribute(waveColors, 3));
    
    potentialGeometry.setAttribute('position', new THREE.BufferAttribute(potentialPositions, 3));
    
    waveMeshRef.current.geometry = waveGeometry;
    potentialMeshRef.current.geometry = potentialGeometry;
  }, [wavefunction, potential, xRange]);

  return (
    <>
      <mesh ref={waveMeshRef}>
        <bufferGeometry />
        <pointsMaterial size={0.05} vertexColors transparent opacity={0.8} />
      </mesh>
      <mesh ref={potentialMeshRef}>
        <bufferGeometry />
        <lineBasicMaterial color={themeColors.potential} linewidth={2} />
      </mesh>
    </>
  );
});

// Display name for debugging
QuantumWavefunction.displayName = 'QuantumWavefunction';

interface CrystalLatticeProps {
  latticeVectors: number[][];
  atomPositions: number[][];
  unitCells: [number, number, number];
  animate?: boolean;
}

export const CrystalLattice: React.FC<CrystalLatticeProps> = memo(({
  latticeVectors,
  atomPositions,
  unitCells,
  animate = true
}) => {
  const groupRef = useRef<THREE.Group>(null);
  const [time, setTime] = useState(0);
  const themeColors = useThemeColors();
  
  useFrame((state, delta) => {
    if (animate) {
      setTime(prev => prev + delta);
      if (groupRef.current) {
        groupRef.current.rotation.y = time * 0.1;
      }
    }
  });

  const atoms = [];
  for (let nx = 0; nx < unitCells[0]; nx++) {
    for (let ny = 0; ny < unitCells[1]; ny++) {
      for (let nz = 0; nz < unitCells[2]; nz++) {
        for (let i = 0; i < atomPositions.length; i++) {
          const position = [
            nx * latticeVectors[0][0] + ny * latticeVectors[1][0] + nz * latticeVectors[2][0] + atomPositions[i][0],
            nx * latticeVectors[0][1] + ny * latticeVectors[1][1] + nz * latticeVectors[2][1] + atomPositions[i][1],
            nx * latticeVectors[0][2] + ny * latticeVectors[1][2] + nz * latticeVectors[2][2] + atomPositions[i][2]
          ];
          
          atoms.push(
            <mesh key={`${nx}-${ny}-${nz}-${i}`} position={[position[0], position[1], position[2]]}>
              <sphereGeometry args={[0.3, 32, 32]} />
              <meshPhongMaterial 
                color={i % 2 === 0 ? themeColors.valence : themeColors.conduction} 
                transparent 
                opacity={0.8}
              />
            </mesh>
          );
        }
      }
    }
  }

  return (
    <group ref={groupRef}>
      {atoms}
      {/* Lattice vector lines */}
      {latticeVectors.map((vector, index) => (
        <mesh key={`vector-${index}`} position={[vector[0]/2, vector[1]/2, vector[2]/2]}>
          <cylinderGeometry args={[0.02, 0.02, Math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)]} />
          <meshBasicMaterial color={themeColors.muted} transparent opacity={0.5} />
        </mesh>
      ))}
    </group>
  );
});

// Display name for debugging
CrystalLattice.displayName = 'CrystalLattice';

interface BandStructurePlotProps {
  kPoints: number[][];
  bands: Float32Array[];
  animate?: boolean;
}

export const BandStructurePlot: React.FC<BandStructurePlotProps> = memo(({
  kPoints,
  bands,
  animate = true
}) => {
  const groupRef = useRef<THREE.Group>(null);
  const themeColors = useThemeColors();
  
  // Memoize band colors for consistent theming
  const bandColors = useMemo(() => {
    return bands.map((_, index) => {
      // Create scientific color palette based on band index
      const hue = (index * 60) % 360; // Spread colors across spectrum
      return `hsl(${hue}, 70%, 60%)`;
    });
  }, [bands.length]);
  
  return (
    <group ref={groupRef}>
      {bands.map((band, bandIndex) => {
        const points = [];
        for (let i = 0; i < kPoints.length; i++) {
          points.push(new THREE.Vector3(
            kPoints[i][0],
            band[i],
            kPoints[i][1]
          ));
        }
        
        const curve = new THREE.CatmullRomCurve3(points);
        const tubeGeometry = new THREE.TubeGeometry(curve, 64, 0.02, 8, false);
        
        return (
          <mesh key={bandIndex} geometry={tubeGeometry}>
            <meshPhongMaterial 
              color={bandColors[bandIndex]}
              transparent 
              opacity={0.8}
            />
          </mesh>
        );
      })}
    </group>
  );
});

// Display name for debugging
BandStructurePlot.displayName = 'BandStructurePlot';

interface Physics3DCanvasProps {
  children: React.ReactNode;
  showGrid?: boolean;
  cameraPosition?: [number, number, number];
}

export const Physics3DCanvas: React.FC<Physics3DCanvasProps> = memo(({
  children,
  showGrid = true,
  cameraPosition = [5, 5, 5] as [number, number, number]
}) => {
  return (
    <div className="w-full h-96 rounded-lg overflow-hidden border border-border/50 bg-card/30">
      <Canvas
        camera={{ position: cameraPosition, fov: 60 }}
        style={{ background: 'transparent' }}
      >
        <ambientLight intensity={0.4} />
        <directionalLight 
          position={[10, 10, 5]} 
          intensity={1} 
          castShadow
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
        />
        <pointLight position={[-10, -10, -10]} intensity={0.5} />
        
        {showGrid && <Grid args={[10, 10]} />}
        
        {children}
        
        <OrbitControls 
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          dampingFactor={0.05}
        />
      </Canvas>
    </div>
  );
});

// Display name for debugging
Physics3DCanvas.displayName = 'Physics3DCanvas';