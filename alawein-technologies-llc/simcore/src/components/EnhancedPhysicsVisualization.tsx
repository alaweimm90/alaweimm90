import React, { useRef, useEffect, useState, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Sphere, Line, Points, PointMaterial } from '@react-three/drei';
import * as THREE from 'three';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  Settings, 
  Eye,
  Zap,
  Atom,
  Waves
} from 'lucide-react';

interface Particle {
  position: THREE.Vector3;
  velocity: THREE.Vector3;
  mass: number;
  charge: number;
  color: string;
  trail: THREE.Vector3[];
}

// Enhanced Particle System Component
function ParticleSystem({ 
  particleCount = 100, 
  isPlaying = true, 
  fieldStrength = 1.0,
  interactionType = 'electromagnetic'
}: {
  particleCount?: number;
  isPlaying?: boolean;
  fieldStrength?: number;
  interactionType?: 'electromagnetic' | 'gravitational' | 'quantum';
}) {
  const pointsRef = useRef<THREE.Points>(null);
  const [particles, setParticles] = useState<Particle[]>([]);

  // Initialize particles
  useEffect(() => {
    const newParticles: Particle[] = [];
    for (let i = 0; i < particleCount; i++) {
      newParticles.push({
        position: new THREE.Vector3(
          (Math.random() - 0.5) * 10,
          (Math.random() - 0.5) * 10,
          (Math.random() - 0.5) * 10
        ),
        velocity: new THREE.Vector3(
          (Math.random() - 0.5) * 2,
          (Math.random() - 0.5) * 2,
          (Math.random() - 0.5) * 2
        ),
        mass: 1 + Math.random() * 2,
        charge: Math.random() > 0.5 ? 1 : -1,
        color: Math.random() > 0.5 ? '#ff4444' : '#4444ff',
        trail: []
      });
    }
    setParticles(newParticles);
  }, [particleCount]);

  // Animation loop
  useFrame((state, delta) => {
    if (!isPlaying || !pointsRef.current) return;

    const positions = pointsRef.current.geometry.attributes.position.array as Float32Array;
    const colors = pointsRef.current.geometry.attributes.color.array as Float32Array;

    setParticles(currentParticles => {
      return currentParticles.map((particle, i) => {
        // Physics simulation based on interaction type
        let force = new THREE.Vector3();

        switch (interactionType) {
          case 'electromagnetic':
            // Coulomb force simulation
            currentParticles.forEach((other, j) => {
              if (i !== j) {
                const distance = particle.position.distanceTo(other.position);
                if (distance > 0.1) {
                  const direction = new THREE.Vector3()
                    .subVectors(particle.position, other.position)
                    .normalize();
                  const forceMagnitude = (particle.charge * other.charge * fieldStrength) / (distance * distance);
                  force.add(direction.multiplyScalar(forceMagnitude * 0.01));
                }
              }
            });
            break;

          case 'gravitational':
            // Gravitational attraction
            currentParticles.forEach((other, j) => {
              if (i !== j) {
                const distance = particle.position.distanceTo(other.position);
                if (distance > 0.1) {
                  const direction = new THREE.Vector3()
                    .subVectors(other.position, particle.position)
                    .normalize();
                  const forceMagnitude = (particle.mass * other.mass * fieldStrength) / (distance * distance);
                  force.add(direction.multiplyScalar(forceMagnitude * 0.001));
                }
              }
            });
            break;

          case 'quantum':
            // Quantum harmonic oscillator
            const center = new THREE.Vector3(0, 0, 0);
            const displacement = new THREE.Vector3().subVectors(particle.position, center);
            force = displacement.multiplyScalar(-fieldStrength * 0.1);
            
            // Add quantum tunneling effect
            if (Math.random() < 0.001) {
              particle.position.add(new THREE.Vector3(
                (Math.random() - 0.5) * 2,
                (Math.random() - 0.5) * 2,
                (Math.random() - 0.5) * 2
              ));
            }
            break;
        }

        // Update velocity and position
        particle.velocity.add(force.multiplyScalar(delta / particle.mass));
        particle.velocity.multiplyScalar(0.99); // Damping
        particle.position.add(particle.velocity.clone().multiplyScalar(delta));

        // Boundary conditions
        if (particle.position.length() > 15) {
          particle.position.normalize().multiplyScalar(15);
          particle.velocity.multiplyScalar(-0.8);
        }

        // Update trail
        particle.trail.push(particle.position.clone());
        if (particle.trail.length > 50) {
          particle.trail.shift();
        }

        // Update geometry arrays
        const idx = i * 3;
        positions[idx] = particle.position.x;
        positions[idx + 1] = particle.position.y;
        positions[idx + 2] = particle.position.z;

        // Color based on velocity magnitude
        const speed = particle.velocity.length();
        const intensity = Math.min(speed / 5, 1);
        colors[idx] = particle.charge > 0 ? intensity : 0.2;
        colors[idx + 1] = 0.2;
        colors[idx + 2] = particle.charge < 0 ? intensity : 0.2;

        return particle;
      });
    });

    pointsRef.current.geometry.attributes.position.needsUpdate = true;
    pointsRef.current.geometry.attributes.color.needsUpdate = true;
  });

  const positions = useMemo(() => {
    const pos = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    
    for (let i = 0; i < particleCount; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 10;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 10;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 10;
      
      colors[i * 3] = Math.random();
      colors[i * 3 + 1] = 0.2;
      colors[i * 3 + 2] = Math.random();
    }
    
    return { positions: pos, colors };
  }, [particleCount]);

  return (
    <Points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions.positions, 3]}
        />
        <bufferAttribute
          attach="attributes-color"
          args={[positions.colors, 3]}
        />
      </bufferGeometry>
      <PointMaterial
        size={0.1}
        vertexColors
        transparent
        opacity={0.8}
        sizeAttenuation
      />
    </Points>
  );
}

// Wave Function Visualization
function WaveFunction({ 
  amplitude = 1, 
  frequency = 1, 
  time = 0,
  waveType = 'sine'
}: {
  amplitude?: number;
  frequency?: number;
  time?: number;
  waveType?: 'sine' | 'square' | 'sawtooth';
}) {
  const points = useMemo(() => {
    const pts: THREE.Vector3[] = [];
    for (let x = -10; x <= 10; x += 0.1) {
      let y: number;
      switch (waveType) {
        case 'square':
          y = amplitude * Math.sign(Math.sin(2 * Math.PI * frequency * x + time));
          break;
        case 'sawtooth':
          y = amplitude * (2 * ((frequency * x + time / (2 * Math.PI)) % 1) - 1);
          break;
        default:
          y = amplitude * Math.sin(2 * Math.PI * frequency * x + time);
      }
      pts.push(new THREE.Vector3(x, y, 0));
    }
    return pts;
  }, [amplitude, frequency, time, waveType]);

  return (
    <Line
      points={points}
      color="#00ff88"
      lineWidth={2}
    />
  );
}

// Quantum State Visualization
function QuantumState({ 
  stateType = 'superposition',
  time = 0 
}: {
  stateType?: 'superposition' | 'entangled' | 'coherent';
  time?: number;
}) {
  const sphereRef = useRef<THREE.Mesh>(null);
  
  useFrame(() => {
    if (sphereRef.current) {
      sphereRef.current.rotation.x = time * 0.5;
      sphereRef.current.rotation.y = time * 0.3;
      
      // Quantum state evolution
      const scale = 1 + 0.2 * Math.sin(time * 2);
      sphereRef.current.scale.setScalar(scale);
    }
  });

  const material = useMemo(() => {
    return new THREE.ShaderMaterial({
      uniforms: {
        time: { value: time },
        stateType: { value: stateType === 'superposition' ? 1 : stateType === 'entangled' ? 2 : 3 }
      },
      vertexShader: `
        varying vec2 vUv;
        varying vec3 vPosition;
        uniform float time;
        
        void main() {
          vUv = uv;
          vPosition = position;
          
          vec3 pos = position;
          pos += normal * sin(time * 2.0 + position.x * 5.0) * 0.1;
          
          gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        }
      `,
      fragmentShader: `
        varying vec2 vUv;
        varying vec3 vPosition;
        uniform float time;
        uniform int stateType;
        
        void main() {
          vec3 color = vec3(0.5);
          
          if (stateType == 1) {
            // Superposition - interference pattern
            float interference = sin(vPosition.x * 10.0 + time) * sin(vPosition.y * 10.0 + time);
            color = vec3(0.5 + interference * 0.5, 0.2, 0.8);
          } else if (stateType == 2) {
            // Entangled - correlated patterns
            float pattern = sin(vPosition.x * 5.0 + time) * cos(vPosition.z * 5.0 + time);
            color = vec3(0.8, 0.2, 0.5 + pattern * 0.5);
          } else {
            // Coherent - uniform oscillation
            float coherence = 0.5 + 0.5 * sin(time * 3.0);
            color = vec3(0.2, 0.8, coherence);
          }
          
          gl_FragColor = vec4(color, 0.8);
        }
      `,
      transparent: true
    });
  }, [time, stateType]);

  return (
    <Sphere ref={sphereRef} args={[2, 32, 32]} material={material} />
  );
}

export function EnhancedPhysicsVisualization() {
  const [isPlaying, setIsPlaying] = useState(true);
  const [visualizationType, setVisualizationType] = useState<'particles' | 'waves' | 'quantum'>('particles');
  const [particleCount, setParticleCount] = useState(50);
  const [fieldStrength, setFieldStrength] = useState(1.0);
  const [interactionType, setInteractionType] = useState<'electromagnetic' | 'gravitational' | 'quantum'>('electromagnetic');
  const [waveAmplitude, setWaveAmplitude] = useState(1);
  const [waveFrequency, setWaveFrequency] = useState(1);
  const [time, setTime] = useState(0);

  useEffect(() => {
    if (isPlaying) {
      const interval = setInterval(() => {
        setTime(t => t + 0.1);
      }, 50);
      return () => clearInterval(interval);
    }
  }, [isPlaying]);

  const resetSimulation = () => {
    setTime(0);
    setIsPlaying(false);
    setTimeout(() => setIsPlaying(true), 100);
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Enhanced Physics Visualization
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs value={visualizationType} onValueChange={(value: any) => setVisualizationType(value)}>
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="particles" className="flex items-center gap-2">
                <Atom className="h-4 w-4" />
                Particles
              </TabsTrigger>
              <TabsTrigger value="waves" className="flex items-center gap-2">
                <Waves className="h-4 w-4" />
                Waves
              </TabsTrigger>
              <TabsTrigger value="quantum" className="flex items-center gap-2">
                <Eye className="h-4 w-4" />
                Quantum
              </TabsTrigger>
            </TabsList>

            <div className="mt-6">
              {/* Controls */}
              <div className="flex flex-wrap gap-4 mb-6">
                <Button
                  onClick={() => setIsPlaying(!isPlaying)}
                  variant={isPlaying ? "default" : "outline"}
                >
                  {isPlaying ? <Pause className="h-4 w-4 mr-2" /> : <Play className="h-4 w-4 mr-2" />}
                  {isPlaying ? 'Pause' : 'Play'}
                </Button>
                
                <Button onClick={resetSimulation} variant="outline">
                  <RotateCcw className="h-4 w-4 mr-2" />
                  Reset
                </Button>

                <Badge variant="outline" className="flex items-center gap-2">
                  <Settings className="h-3 w-3" />
                  Time: {time.toFixed(1)}s
                </Badge>
              </div>

              {/* Parameter Controls */}
              <TabsContent value="particles" className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                  <div>
                    <label className="text-sm font-medium mb-2 block">Particle Count: {particleCount}</label>
                    <Slider
                      value={[particleCount]}
                      onValueChange={(value) => setParticleCount(value[0])}
                      min={10}
                      max={200}
                      step={10}
                    />
                  </div>
                  
                  <div>
                    <label className="text-sm font-medium mb-2 block">Field Strength: {fieldStrength.toFixed(1)}</label>
                    <Slider
                      value={[fieldStrength]}
                      onValueChange={(value) => setFieldStrength(value[0])}
                      min={0.1}
                      max={3.0}
                      step={0.1}
                    />
                  </div>

                  <div>
                    <label className="text-sm font-medium mb-2 block">Interaction Type</label>
                    <select 
                      value={interactionType} 
                      onChange={(e) => setInteractionType(e.target.value as any)}
                      className="w-full p-2 rounded border"
                    >
                      <option value="electromagnetic">Electromagnetic</option>
                      <option value="gravitational">Gravitational</option>
                      <option value="quantum">Quantum</option>
                    </select>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="waves" className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                  <div>
                    <label className="text-sm font-medium mb-2 block">Amplitude: {waveAmplitude.toFixed(1)}</label>
                    <Slider
                      value={[waveAmplitude]}
                      onValueChange={(value) => setWaveAmplitude(value[0])}
                      min={0.1}
                      max={3.0}
                      step={0.1}
                    />
                  </div>
                  
                  <div>
                    <label className="text-sm font-medium mb-2 block">Frequency: {waveFrequency.toFixed(1)}</label>
                    <Slider
                      value={[waveFrequency]}
                      onValueChange={(value) => setWaveFrequency(value[0])}
                      min={0.1}
                      max={5.0}
                      step={0.1}
                    />
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="quantum" className="space-y-4">
                <div className="text-sm text-muted-foreground mb-4">
                  Visualizing quantum superposition, entanglement, and coherence effects.
                  The sphere's behavior represents different quantum states.
                </div>
              </TabsContent>

              {/* 3D Visualization */}
              <div className="h-96 border rounded-lg overflow-hidden bg-background">
                <Canvas camera={{ position: [15, 15, 15], fov: 60 }}>
                  <ambientLight intensity={0.3} />
                  <pointLight position={[10, 10, 10]} intensity={0.8} />
                  <pointLight position={[-10, -10, -10]} intensity={0.5} color="#4444ff" />
                  
                  {visualizationType === 'particles' && (
                    <ParticleSystem
                      particleCount={particleCount}
                      isPlaying={isPlaying}
                      fieldStrength={fieldStrength}
                      interactionType={interactionType}
                    />
                  )}
                  
                  {visualizationType === 'waves' && (
                    <>
                      <WaveFunction
                        amplitude={waveAmplitude}
                        frequency={waveFrequency}
                        time={time}
                        waveType="sine"
                      />
                      <WaveFunction
                        amplitude={waveAmplitude * 0.7}
                        frequency={waveFrequency * 1.5}
                        time={time * 1.2}
                        waveType="sine"
                      />
                    </>
                  )}
                  
                  {visualizationType === 'quantum' && (
                    <QuantumState stateType="superposition" time={time} />
                  )}
                  
                  <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
                  
                  {/* Coordinate system */}
                  <Line points={[[-10, 0, 0], [10, 0, 0]]} color="#ff0000" lineWidth={1} />
                  <Line points={[[0, -10, 0], [0, 10, 0]]} color="#00ff00" lineWidth={1} />
                  <Line points={[[0, 0, -10], [0, 0, 10]]} color="#0000ff" lineWidth={1} />
                </Canvas>
              </div>
            </div>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}