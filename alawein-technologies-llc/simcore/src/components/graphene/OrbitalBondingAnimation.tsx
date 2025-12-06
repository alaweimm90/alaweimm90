import React, { useRef, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text } from '@react-three/drei';
import * as THREE from 'three';

interface AtomProps {
  position: [number, number, number];
  label: string;
  type: 'carbon';
  isHighlighted?: boolean;
}

function Atom({ position, label, type, isHighlighted = false }: AtomProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  
  useFrame((state) => {
    if (meshRef.current && isHighlighted) {
      meshRef.current.scale.setScalar(1 + 0.05 * Math.sin(state.clock.elapsedTime * 3));
    }
  });

  return (
    <group position={position}>
      <mesh ref={meshRef}>
        <sphereGeometry args={[0.2, 32, 32]} />
        <meshPhongMaterial 
          color={isHighlighted ? "#ff4444" : "#333333"} 
          emissive={isHighlighted ? "#441100" : "#111111"}
          emissiveIntensity={0.1}
          shininess={100}
        />
      </mesh>
      <Text
        position={[0, 0.4, 0]}
        fontSize={0.12}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
        font="/fonts/helvetiker_regular.typeface.json"
      >
        {label}
      </Text>
    </group>
  );
}

interface SOrbitalProps {
  position: [number, number, number];
  phase: number;
  visible: boolean;
}

function SOrbital({ position, phase, visible }: SOrbitalProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  
  useFrame((state) => {
    if (meshRef.current && visible) {
      const time = state.clock.elapsedTime + phase;
      const pulsation = 1 + 0.1 * Math.sin(time * 2);
      meshRef.current.scale.setScalar(pulsation);
    }
  });

  if (!visible) return null;

  return (
    <mesh ref={meshRef} position={position}>
      <sphereGeometry args={[0.15, 16, 16]} />
      <meshPhongMaterial 
        color="#4FC3F7"
        transparent
        opacity={0.4}
        emissive="#0277BD"
        emissiveIntensity={0.2}
      />
    </mesh>
  );
}

interface POrbitalProps {
  position: [number, number, number];
  direction: 'x' | 'y' | 'z';
  phase: number;
  visible: boolean;
}

function POrbital({ position, direction, phase, visible }: POrbitalProps) {
  const groupRef = useRef<THREE.Group>(null);
  
  useFrame((state) => {
    if (groupRef.current && visible) {
      const time = state.clock.elapsedTime + phase;
      const rotation = time * 0.5;
      if (direction === 'z') {
        groupRef.current.rotation.z = rotation;
      }
      const pulsation = 1 + 0.15 * Math.sin(time * 3);
      groupRef.current.scale.setScalar(pulsation);
    }
  });

  if (!visible) return null;

  const getGeometry = () => {
    switch (direction) {
      case 'z':
        return <sphereGeometry args={[0.12, 8, 16]} />;
      case 'x':
        return <cylinderGeometry args={[0.08, 0.08, 0.3, 8]} />;
      case 'y':
        return <cylinderGeometry args={[0.08, 0.08, 0.3, 8]} />;
      default:
        return <sphereGeometry args={[0.1, 8, 16]} />;
    }
  };

  const getRotation = (): [number, number, number] => {
    switch (direction) {
      case 'x':
        return [0, 0, Math.PI / 2];
      case 'y':
        return [0, 0, 0];
      case 'z':
        return [Math.PI / 2, 0, 0];
      default:
        return [0, 0, 0];
    }
  };

  return (
    <group ref={groupRef} position={position}>
      {/* Positive lobe */}
      <mesh position={direction === 'z' ? [0, 0, 0.15] : direction === 'x' ? [0.15, 0, 0] : [0, 0.15, 0]} rotation={getRotation()}>
        {getGeometry()}
        <meshPhongMaterial 
          color="#E91E63"
          transparent
          opacity={0.7}
          emissive="#AD1457"
          emissiveIntensity={0.3}
        />
      </mesh>
      
      {/* Negative lobe */}
      <mesh position={direction === 'z' ? [0, 0, -0.15] : direction === 'x' ? [-0.15, 0, 0] : [0, -0.15, 0]} rotation={getRotation()}>
        {getGeometry()}
        <meshPhongMaterial 
          color="#2196F3"
          transparent
          opacity={0.7}
          emissive="#1565C0"
          emissiveIntensity={0.3}
        />
      </mesh>
    </group>
  );
}

interface BondProps {
  start: [number, number, number];
  end: [number, number, number];
  bondType: 'sigma' | 'pi';
  strength: number;
}

function Bond({ start, end, bondType, strength }: BondProps) {
  const ref = useRef<THREE.Mesh>(null);
  
  const direction = new THREE.Vector3(end[0] - start[0], end[1] - start[1], end[2] - start[2]);
  const length = direction.length();
  const center = new THREE.Vector3(
    (start[0] + end[0]) / 2,
    (start[1] + end[1]) / 2,
    (start[2] + end[2]) / 2
  );
  
  direction.normalize();
  const quaternion = new THREE.Quaternion();
  quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);

  useFrame((state) => {
    if (ref.current) {
      const intensity = bondType === 'sigma' ? 0.4 : 0.2 + 0.3 * Math.sin(state.clock.elapsedTime * 2) * strength;
      (ref.current.material as THREE.MeshPhongMaterial).emissiveIntensity = intensity;
    }
  });

  const radius = bondType === 'sigma' ? 0.04 : 0.02;
  const color = bondType === 'sigma' ? "#FFC107" : "#9C27B0";
  const emissive = bondType === 'sigma' ? "#F57C00" : "#6A1B9A";

  return (
    <mesh ref={ref} position={center} quaternion={quaternion}>
      <cylinderGeometry args={[radius, radius, length, 8]} />
      <meshPhongMaterial 
        color={color}
        emissive={emissive}
        emissiveIntensity={0.3}
        transparent
        opacity={bondType === 'sigma' ? 0.9 : 0.6}
      />
    </mesh>
  );
}

function GrapheneLattice() {
  const [showS] = useState(true);
  const [showP] = useState(true);
  
  // Honeycomb lattice - only central unit cell for clarity
  const a = 1.8; // Scaled lattice constant
  const positions: Array<{ 
    pos: [number, number, number]; 
    label: string; 
    sublattice: 'A' | 'B';
    index: number;
  }> = [
    // Central unit cell
    { pos: [0, 0, 0], label: 'A₁', sublattice: 'A', index: 0 },
    { pos: [a * Math.sqrt(3)/2, a * 0.5, 0], label: 'B₁', sublattice: 'B', index: 1 },
    { pos: [a * Math.sqrt(3)/2, -a * 0.5, 0], label: 'B₂', sublattice: 'B', index: 2 },
    { pos: [0, -a, 0], label: 'A₂', sublattice: 'A', index: 3 },
    { pos: [-a * Math.sqrt(3)/2, -a * 0.5, 0], label: 'B₃', sublattice: 'B', index: 4 },
    { pos: [-a * Math.sqrt(3)/2, a * 0.5, 0], label: 'B₄', sublattice: 'B', index: 5 },
    { pos: [0, a, 0], label: 'A₃', sublattice: 'A', index: 6 },
  ];
  
  // Generate bonds - only nearest neighbors
  const bonds: Array<{ start: [number, number, number]; end: [number, number, number]; type: 'sigma' | 'pi'; strength: number }> = [];
  
  for (let i = 0; i < positions.length; i++) {
    for (let j = i + 1; j < positions.length; j++) {
      const atom1 = positions[i];
      const atom2 = positions[j];
      
      if (atom1.sublattice !== atom2.sublattice) {
        const dist = Math.sqrt(
          Math.pow(atom1.pos[0] - atom2.pos[0], 2) +
          Math.pow(atom1.pos[1] - atom2.pos[1], 2)
        );
        
        // Only nearest neighbors
        if (dist < a * 0.8) {
          bonds.push({
            start: atom1.pos,
            end: atom2.pos,
            type: 'sigma',
            strength: 1
          });
        }
      }
    }
  }

  return (
    <group>
      {/* Control panel */}
      <group position={[0, 2.5, 0]}>
        <Text
          fontSize={0.15}
          color="#ffffff"
          anchorX="center"
        >
          Carbon Atoms with sp² Hybridization
        </Text>
      </group>
      
      <group position={[0, -2.8, 0]}>
        <Text
          fontSize={0.1}
          color="#94a3b8"
          anchorX="center"
        >
          Blue: s orbitals | Red/Blue lobes: p orbitals | Yellow: σ bonds
        </Text>
      </group>

      {/* Render σ bonds */}
      {bonds.map((bond, index) => (
        <Bond
          key={`bond-${index}`}
          start={bond.start}
          end={bond.end}
          bondType="sigma"
          strength={bond.strength}
        />
      ))}
      
      {/* Render atoms */}
      {positions.map((atom, index) => (
        <Atom
          key={`atom-${index}`}
          position={atom.pos}
          label={atom.label}
          type="carbon"
          isHighlighted={index < 3}
        />
      ))}
      
      {/* Render s orbitals */}
      {showS && positions.map((atom, index) => (
        <SOrbital
          key={`s-orbital-${index}`}
          position={atom.pos}
          phase={index * Math.PI / 3}
          visible={showS}
        />
      ))}
      
      {/* Render p orbitals - sp² hybridization uses px, py */}
      {showP && positions.slice(0, 4).map((atom, index) => (
        <React.Fragment key={`p-orbitals-${index}`}>
          <POrbital
            position={[atom.pos[0], atom.pos[1], atom.pos[2] + 0.1]}
            direction="x"
            phase={index * Math.PI / 4}
            visible={showP}
          />
          <POrbital
            position={[atom.pos[0] + 0.1, atom.pos[1], atom.pos[2]]}
            direction="y"
            phase={index * Math.PI / 4 + Math.PI / 2}
            visible={showP}
          />
          {/* pz orbital for π bonding */}
          <POrbital
            position={[atom.pos[0], atom.pos[1], atom.pos[2] + 0.3]}
            direction="z"
            phase={index * Math.PI / 4 + Math.PI}
            visible={showP}
          />
        </React.Fragment>
      ))}
    </group>
  );
}

export function OrbitalBondingAnimation() {
  return (
    <div className="w-full h-[600px] bg-background border border-border rounded-lg overflow-hidden relative">
      <Canvas
        camera={{ position: [4, 3, 4], fov: 45 }}
        style={{ background: 'transparent' }}
      >
        <ambientLight intensity={0.3} />
        <directionalLight position={[5, 5, 5]} intensity={0.7} castShadow />
        <pointLight position={[-3, 3, 3]} intensity={0.4} color="#4FC3F7" />
        <pointLight position={[3, -3, 3]} intensity={0.4} color="#E91E63" />
        
        <GrapheneLattice />
        
        <OrbitControls 
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          autoRotate={true}
          autoRotateSpeed={1}
          maxDistance={8}
          minDistance={2}
        />
      </Canvas>
      
      {/* Legend */}
      <div className="absolute top-4 left-4 bg-background/80 backdrop-blur-sm rounded-lg p-3 text-sm space-y-1">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-blue-400"></div>
          <span>s orbitals</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-pink-500"></div>
          <span>p orbitals (+)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-blue-600"></div>
          <span>p orbitals (-)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
          <span>σ bonds</span>
        </div>
      </div>
    </div>
  );
}