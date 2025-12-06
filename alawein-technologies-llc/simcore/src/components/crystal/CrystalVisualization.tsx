import React, { useMemo, useRef, Suspense } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Line, Html, Environment, Grid, Sphere, Box } from '@react-three/drei';
import * as THREE from 'three';
import { useCrystalStore, crystalStructures } from '@/lib/crystal-store';

// Atom component with LOD and instancing for performance
function Atom({ position, color, element, scale }: { 
  position: [number, number, number]; 
  color: string; 
  element: string; 
  scale: number; 
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const { camera } = useThree();
  
  // Atomic radii in Angstroms
  const atomicRadii: Record<string, number> = {
    'H': 0.31, 'C': 0.70, 'N': 0.65, 'O': 0.60, 'F': 0.50,
    'Na': 1.86, 'Mg': 1.60, 'Al': 1.43, 'Si': 1.11, 'P': 1.07,
    'S': 1.05, 'Cl': 0.99, 'K': 2.27, 'Ca': 1.97, 'Ti': 1.47,
    'Cr': 1.28, 'Mn': 1.39, 'Fe': 1.25, 'Co': 1.26, 'Ni': 1.24,
    'Cu': 1.28, 'Zn': 1.34, 'Ga': 1.35, 'Ge': 1.31, 'As': 1.21,
    'Se': 1.16, 'Br': 1.14, 'Sr': 2.15, 'Zr': 1.60, 'Mo': 1.39,
    'Ag': 1.45, 'In': 1.67, 'Sn': 1.58, 'Sb': 1.41, 'Te': 1.37,
    'I': 1.33, 'Ba': 2.22, 'W': 1.39, 'Au': 1.44, 'Pb': 1.75,
    'Bi': 1.70
  };

  const radius = (atomicRadii[element] || 1.0) * scale * 0.3;

  useFrame(() => {
    if (meshRef.current) {
      const distance = camera.position.distanceTo(meshRef.current.position);
      // LOD: reduce geometry complexity at distance
      const detail = distance < 10 ? 32 : distance < 20 ? 16 : 8;
      meshRef.current.geometry = new THREE.SphereGeometry(radius, detail, detail);
    }
  });

  return (
    <mesh ref={meshRef} position={position}>
      <sphereGeometry args={[radius, 32, 32]} />
      <meshStandardMaterial color={color} metalness={0.1} roughness={0.7} />
      {scale > 0.8 && (
        <Html distanceFactor={10} occlude>
          <div className="text-xs font-semibold text-foreground bg-background/80 px-1 rounded border">
            {element}
          </div>
        </Html>
      )}
    </mesh>
  );
}

// Bond component
function Bond({ start, end, color = '#cccccc' }: { 
  start: [number, number, number]; 
  end: [number, number, number]; 
  color?: string; 
}) {
  const points = useMemo(() => [
    new THREE.Vector3(...start),
    new THREE.Vector3(...end)
  ], [start, end]);

  return (
    <Line points={points} color={color} lineWidth={2} />
  );
}

// Unit cell wireframe
function UnitCell({ latticeVectors, latticeConstant }: { 
  latticeVectors: number[][]; 
  latticeConstant: number; 
}) {
  const lines = useMemo(() => {
    const origin = [0, 0, 0];
    const a = latticeVectors[0].map(x => x * latticeConstant);
    const b = latticeVectors[1].map(x => x * latticeConstant);
    const c = latticeVectors[2].map(x => x * latticeConstant);

    // 8 vertices of the unit cell
    const vertices = [
      origin,
      a,
      b,
      c,
      [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
      [a[0] + c[0], a[1] + c[1], a[2] + c[2]],
      [b[0] + c[0], b[1] + c[1], b[2] + c[2]],
      [a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]]
    ];

    // 12 edges of the unit cell
    const edges = [
      [0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], 
      [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]
    ];

    return edges.map(([i, j]) => ({
      start: vertices[i] as [number, number, number],
      end: vertices[j] as [number, number, number]
    }));
  }, [latticeVectors, latticeConstant]);

  return (
    <group>
      {lines.map((line, i) => (
        <Bond key={i} start={line.start} end={line.end} color="#3b82f6" />
      ))}
    </group>
  );
}

// Miller plane visualization
function MillerPlane({ indices, color, opacity, latticeVectors, latticeConstant }: {
  indices: [number, number, number];
  color: string;
  opacity: number;
  latticeVectors: number[][];
  latticeConstant: number;
}) {
  const planeGeometry = useMemo(() => {
    const [h, k, l] = indices;
    if (h === 0 && k === 0 && l === 0) return null;

    // Calculate intercepts
    const a = latticeVectors[0].map(x => x * latticeConstant);
    const b = latticeVectors[1].map(x => x * latticeConstant);
    const c = latticeVectors[2].map(x => x * latticeConstant);

    // Plane intercepts
    const intercepts = [
      h !== 0 ? a.map(x => x / h) : [Infinity, Infinity, Infinity],
      k !== 0 ? b.map(x => x / k) : [Infinity, Infinity, Infinity],
      l !== 0 ? c.map(x => x / l) : [Infinity, Infinity, Infinity]
    ];

    // Create plane geometry (simplified for visualization)
    const geometry = new THREE.PlaneGeometry(4, 4);
    
    // Calculate normal vector
    const normal = new THREE.Vector3(h, k, l).normalize();
    
    return { geometry, normal };
  }, [indices, latticeVectors, latticeConstant]);

  if (!planeGeometry) return null;

  return (
    <mesh geometry={planeGeometry.geometry}>
      <meshBasicMaterial 
        color={color} 
        transparent 
        opacity={opacity} 
        side={THREE.DoubleSide} 
      />
    </mesh>
  );
}

// Reciprocal lattice visualization
function ReciprocalLattice({ latticeVectors, latticeConstant }: {
  latticeVectors: number[][];
  latticeConstant: number;
}) {
  const reciprocalPoints = useMemo(() => {
    // Calculate reciprocal lattice vectors
    const a = new THREE.Vector3(...latticeVectors[0]);
    const b = new THREE.Vector3(...latticeVectors[1]);
    const c = new THREE.Vector3(...latticeVectors[2]);
    
    const volume = a.dot(b.clone().cross(c));
    
    const astar = b.clone().cross(c).multiplyScalar(2 * Math.PI / volume);
    const bstar = c.clone().cross(a).multiplyScalar(2 * Math.PI / volume);
    const cstar = a.clone().cross(b).multiplyScalar(2 * Math.PI / volume);

    const points = [];
    for (let i = -2; i <= 2; i++) {
      for (let j = -2; j <= 2; j++) {
        for (let k = -2; k <= 2; k++) {
          const point = astar.clone().multiplyScalar(i)
            .add(bstar.clone().multiplyScalar(j))
            .add(cstar.clone().multiplyScalar(k));
          points.push(point.toArray() as [number, number, number]);
        }
      }
    }

    return points;
  }, [latticeVectors, latticeConstant]);

  return (
    <group>
      {reciprocalPoints.map((point, i) => (
        <Sphere key={i} position={point} args={[0.05]} material-color="#ff6b6b" />
      ))}
    </group>
  );
}

// Brillouin zone visualization
function BrillouinZone({ structure }: { structure: any }) {
  // Simplified Brillouin zone - would need proper implementation
  return (
    <Box args={[2, 2, 2]} material-color="#22c55e" material-transparent material-opacity={0.1} />
  );
}

// Main crystal visualization component
export function CrystalVisualization() {
  const {
    atoms,
    showAtoms,
    showBonds,
    showUnitCell,
    showAxes,
    atomScale,
    bondCutoff,
    millerPlanes,
    showReciprocalLattice,
    showBrillouinZone,
    selectedStructure,
    latticeConstant
  } = useCrystalStore();

  const structure = crystalStructures[selectedStructure];

  // Calculate bonds
  const bonds = useMemo(() => {
    if (!showBonds) return [];
    
    const bondList = [];
    for (let i = 0; i < atoms.length; i++) {
      for (let j = i + 1; j < atoms.length; j++) {
        const distance = Math.sqrt(
          atoms.reduce((sum, _, dim) => 
            sum + Math.pow(atoms[i].position[dim] - atoms[j].position[dim], 2), 0
          )
        );
        
        if (distance < bondCutoff) {
          bondList.push({
            start: atoms[i].position,
            end: atoms[j].position
          });
        }
      }
    }
    return bondList;
  }, [atoms, showBonds, bondCutoff]);

  return (
    <div className="w-full h-full">
      <Canvas camera={{ position: [10, 10, 10], fov: 50 }}>
        <Suspense fallback={null}>
          <Environment preset="studio" />
          <ambientLight intensity={0.6} />
          <directionalLight position={[10, 10, 5]} intensity={0.8} />
          
          <OrbitControls 
            enablePan={true} 
            enableZoom={true} 
            enableRotate={true}
            maxDistance={50}
            minDistance={2}
          />

          {/* Coordinate axes */}
          {showAxes && (
            <group>
              <Line points={[[0, 0, 0], [3, 0, 0]]} color="#ef4444" lineWidth={3} />
              <Line points={[[0, 0, 0], [0, 3, 0]]} color="#22c55e" lineWidth={3} />
              <Line points={[[0, 0, 0], [0, 0, 3]]} color="#3b82f6" lineWidth={3} />
              <Text position={[3.2, 0, 0]} color="#ef4444" fontSize={0.5}>X</Text>
              <Text position={[0, 3.2, 0]} color="#22c55e" fontSize={0.5}>Y</Text>
              <Text position={[0, 0, 3.2]} color="#3b82f6" fontSize={0.5}>Z</Text>
            </group>
          )}

          {/* Atoms */}
          {showAtoms && atoms.map((atom) => (
            <Atom
              key={atom.id}
              position={atom.position}
              color={atom.color}
              element={atom.element}
              scale={atomScale}
            />
          ))}

          {/* Bonds */}
          {bonds.map((bond, i) => (
            <Bond key={i} start={bond.start} end={bond.end} />
          ))}

          {/* Unit cell */}
          {showUnitCell && structure && (
            <UnitCell 
              latticeVectors={structure.latticeVectors} 
              latticeConstant={latticeConstant}
            />
          )}

          {/* Miller planes */}
          {millerPlanes.filter(plane => plane.visible).map((plane, i) => (
            <MillerPlane
              key={i}
              indices={plane.indices}
              color={plane.color}
              opacity={plane.opacity}
              latticeVectors={structure?.latticeVectors || []}
              latticeConstant={latticeConstant}
            />
          ))}

          {/* Reciprocal lattice */}
          {showReciprocalLattice && structure && (
            <ReciprocalLattice 
              latticeVectors={structure.latticeVectors}
              latticeConstant={latticeConstant}
            />
          )}

          {/* Brillouin zone */}
          {showBrillouinZone && structure && (
            <BrillouinZone structure={structure} />
          )}

          {/* Grid */}
          <Grid 
            args={[10, 10]} 
            cellSize={1} 
            cellThickness={0.5} 
            cellColor="#6b7280" 
            sectionSize={5} 
            sectionThickness={1.5} 
            sectionColor="#374151"
            fadeDistance={25}
            fadeStrength={1}
            followCamera={false}
            infiniteGrid={true}
          />
        </Suspense>
      </Canvas>
    </div>
  );
}