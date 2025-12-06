import React, { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

interface DiracCone3DProps {
  fermiVelocity: number; // m/s, used to scale visually
  scale?: number;        // visual scale factor
  colorTop?: string;
  colorBottom?: string;
}

function DoubleConeMesh({ alpha = 1, colorTop = '#3b82f6', colorBottom = '#ef4444' }: { alpha?: number; colorTop?: string; colorBottom?: string }) {
  const topRef = useRef<THREE.Mesh>(null);
  const botRef = useRef<THREE.Mesh>(null);

  // Build a radial grid and displace along z: z = ± alpha * r
  const segments = 100;
  const radius = 1; // normalized radius

  const geometry = useMemo(() => {
    const geom = new THREE.PlaneGeometry(2 * radius, 2 * radius, segments, segments);
    const pos = geom.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const y = pos.getY(i);
      const r = Math.hypot(x, y);
      const z = alpha * r;
      pos.setZ(i, z);
    }
    pos.needsUpdate = true;
    geom.computeVertexNormals();
    return geom;
  }, [alpha]);

  const materialTop = useMemo(() => new THREE.MeshStandardMaterial({ color: colorTop, metalness: 0.1, roughness: 0.5, side: THREE.DoubleSide, transparent: true, opacity: 0.9 }), [colorTop]);
  const materialBottom = useMemo(() => new THREE.MeshStandardMaterial({ color: colorBottom, metalness: 0.1, roughness: 0.5, side: THREE.DoubleSide, transparent: true, opacity: 0.9 }), [colorBottom]);

  useFrame((_, dt) => {
    if (topRef.current) topRef.current.rotation.z += dt * 0.05;
    if (botRef.current) botRef.current.rotation.z += dt * 0.05;
  });

  return (
    <group>
      <mesh ref={topRef} geometry={geometry} material={materialTop} rotation={[-Math.PI / 2, 0, 0]} />
      <mesh ref={botRef} geometry={geometry} material={materialBottom} rotation={[Math.PI / 2, 0, 0]} />
    </group>
  );
}

export default function DiracCone3D({ fermiVelocity, scale = 1, colorTop, colorBottom }: DiracCone3DProps) {
  // Map Fermi velocity (~1e6 m/s) to visual slope alpha
  const alpha = useMemo(() => {
    // Normalize around 1e6 m/s -> alpha ~ 0.8
    const norm = Math.min(Math.max(fermiVelocity / 1e6, 0.2), 2);
    return 0.8 * norm * scale;
  }, [fermiVelocity, scale]);

  return (
    <Canvas camera={{ position: [2.5, 2.5, 2.5], fov: 45 }}>
      <ambientLight intensity={0.6} />
      <directionalLight position={[3, 5, 2]} intensity={0.9} />
      <gridHelper args={[4, 20, '#555', '#333']} />
      <axesHelper args={[1.5]} />

      <DoubleConeMesh alpha={alpha} colorTop={colorTop} colorBottom={colorBottom} />

      <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
    </Canvas>
  );
}
