import React, { useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Line, Text, Html } from '@react-three/drei';
import * as THREE from 'three';
import { useLLGStore, vec3 } from '@/lib/llg-store';

// Enhanced magnetization visualization with torque
function MagnetizationVisualization() {
  const {
    magnetization,
    effectiveField,
    torque,
    magnetizationHistory,
    showEffectiveField,
    showTorque,
    showTrajectory,
    isRunning,
    stepSimulation
  } = useLLGStore();

  // Animation loop for real-time simulation
  useFrame((state, delta) => {
    if (isRunning) {
      // Run multiple steps for smooth animation
      const stepsPerFrame = Math.max(1, Math.floor(delta * 1000));
      for (let i = 0; i < stepsPerFrame; i++) {
        stepSimulation();
      }
    }
  });

  // Trajectory points (last 500 for performance)
  const trajectoryPoints = showTrajectory 
    ? magnetizationHistory.slice(-500)
    : [];

  return (
    <group>
      {/* Coordinate system */}
      <Line points={[[-1.2, 0, 0], [1.2, 0, 0]]} color="#ef4444" lineWidth={2} />
      <Line points={[[0, -1.2, 0], [0, 1.2, 0]]} color="#22c55e" lineWidth={2} />
      <Line points={[[0, 0, -1.2], [0, 0, 1.2]]} color="#3b82f6" lineWidth={2} />
      
      <Text position={[1.3, 0, 0]} fontSize={0.12} color="#ef4444">X</Text>
      <Text position={[0, 1.3, 0]} fontSize={0.12} color="#22c55e">Y</Text>
      <Text position={[0, 0, 1.3]} fontSize={0.12} color="#3b82f6">Z</Text>
      
      {/* Unit sphere */}
      <mesh>
        <sphereGeometry args={[1, 64, 64]} />
        <meshBasicMaterial 
          color="#374151" 
          transparent 
          opacity={0.1} 
          wireframe 
        />
      </mesh>
      
      {/* Magnetization vector */}
      <group>
        <mesh position={magnetization}>
          <sphereGeometry args={[0.03, 16, 16]} />
          <meshPhongMaterial color="#dc2626" emissive="#dc2626" emissiveIntensity={0.2} />
        </mesh>
        
        <Line 
          points={[[0, 0, 0], magnetization]} 
          color="#dc2626" 
          lineWidth={6}
        />
        
        <Html position={[magnetization[0] + 0.1, magnetization[1] + 0.1, magnetization[2] + 0.1]}>
          <div className="text-sm font-bold text-destructive bg-background/80 px-2 py-1 rounded border">
            m
          </div>
        </Html>
      </group>
      
      {/* Effective field vector */}
      {showEffectiveField && (
        <group>
          <mesh position={vec3.scale(effectiveField, 0.3)}>
            <sphereGeometry args={[0.02, 12, 12]} />
            <meshPhongMaterial color="#16a34a" />
          </mesh>
          
          <Line 
            points={[[0, 0, 0], vec3.scale(effectiveField, 0.3)]} 
            color="#16a34a" 
            lineWidth={4}
          />
          
          <Html position={vec3.add(vec3.scale(effectiveField, 0.3), [0.1, 0.1, 0.1])}>
            <div className="text-sm font-bold text-accent bg-background/80 px-2 py-1 rounded border">
              H_eff
            </div>
          </Html>
        </group>
      )}
      
      {/* Torque vector */}
      {showTorque && (
        <group>
          <mesh position={vec3.scale(torque, 0.5)}>
            <sphereGeometry args={[0.02, 12, 12]} />
            <meshPhongMaterial color="#f59e0b" />
          </mesh>
          
          <Line 
            points={[[0, 0, 0], vec3.scale(torque, 0.5)]} 
            color="#f59e0b" 
            lineWidth={4}
          />
          
          <Html position={vec3.add(vec3.scale(torque, 0.5), [0.1, 0.1, 0.1])}>
            <div className="text-sm font-bold text-accent bg-background/80 px-2 py-1 rounded border">
              τ
            </div>
          </Html>
        </group>
      )}
      
      {/* Trajectory */}
      {showTrajectory && trajectoryPoints.length > 1 && (
        <Line 
          points={trajectoryPoints} 
          color="#8b5cf6" 
          lineWidth={3}
        />
      )}
      
      {/* Trajectory points as small spheres for better visibility */}
      {showTrajectory && trajectoryPoints.slice(-50).map((point, index) => (
        <mesh key={index} position={point}>
          <sphereGeometry args={[0.008, 8, 8]} />
          <meshBasicMaterial 
            color="#8b5cf6" 
            transparent 
            opacity={0.6 + 0.4 * (index / 50)} 
          />
        </mesh>
      ))}
      
      {/* Performance indicator */}
      <Html position={[-1.4, 1.4, 0]}>
        <div className="text-xs text-muted-foreground bg-background/80 px-2 py-1 rounded border">
          |m| = {vec3.magnitude(magnetization).toFixed(6)}
        </div>
      </Html>
    </group>
  );
}

export function LLG3DVisualization() {
  return (
    <div className="w-full h-full">
      <Canvas camera={{ position: [2.5, 2.5, 2.5], fov: 45 }}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} intensity={0.7} />
        <directionalLight position={[-5, -5, -5]} intensity={0.3} />
        
        <OrbitControls 
          enablePan={true} 
          enableZoom={true} 
          enableRotate={true}
          maxDistance={8}
          minDistance={1.5}
        />
        
        <MagnetizationVisualization />
      </Canvas>
    </div>
  );
}