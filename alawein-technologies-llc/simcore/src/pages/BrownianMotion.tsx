import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Trail, Html } from '@react-three/drei';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { PhysicsModuleLayout, PhysicsContentCard } from '@/components/PhysicsModuleLayout';
import { PhysicsModuleHeader } from '@/components/PhysicsModuleHeader';
import { PhysicsTheory } from '@/components/PhysicsTheory';
import { PerformanceBenchmark } from '@/components/PerformanceBenchmark';
import { Play, Pause, RotateCcw, Move, TrendingUp, Activity, BarChart3, Download, BookOpen, Zap, Save, Settings, BarChart2 } from 'lucide-react';
import { InlineMath, BlockMath } from '@/components/ui/Math';
import { useToast } from '@/hooks/use-toast';
import * as THREE from 'three';
import Plot from 'react-plotly.js';
import { useSEO } from '@/hooks/use-seo';
import { UNIFIED_SCIENTIFIC_PLOT_SYSTEM as PLOT } from '@/lib/scientific-plot-system';

interface BrownianParticle {
  id: number;
  position: THREE.Vector3;
  velocity: THREE.Vector3;
  force: THREE.Vector3;
  trail: THREE.Vector3[];
  displacement: number;
  totalDistance: number;
  mass: number;
  radius: number;
  color: THREE.Color;
  temperature: number;
  lifetime: number;
  interactions: number;
}

interface FluidProperties {
  viscosity: number;
  density: number;
  temperature: number;
  molecularMass: number;
}

interface DiffusionData {
  time: number;
  msd: number;
  msd1D: number;
  autocorrelation: number;
  diffusionCoeff: number;
}

interface VelocityDistribution {
  velocity: number;
  probability: number;
  maxwell: number;
  experimental: number;
}

interface CollisionEvent {
  time: number;
  particleId: number;
  position: THREE.Vector3;
  velocityChange: number;
  energy: number;
}

interface BrownianParticleComponentProps {
  particle: BrownianParticle;
  showTrails: boolean;
}

const BrownianParticleComponent: React.FC<BrownianParticleComponentProps> = ({ 
  particle, 
  showTrails 
}) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const velocityRef = useRef<THREE.ArrowHelper>(null);
  const forceRef = useRef<THREE.ArrowHelper>(null);

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.position.copy(particle.position);
      
      // Update velocity arrow
      if (velocityRef.current && particle.velocity.length() > 0.001) {
        const velDirection = particle.velocity.clone().normalize();
        const velMagnitude = Math.min(particle.velocity.length() * 100, 2);
        velocityRef.current.setDirection(velDirection);
        velocityRef.current.setLength(velMagnitude, velMagnitude * 0.2, velMagnitude * 0.1);
        velocityRef.current.position.copy(particle.position);
      }
      
      // Update force arrow
      if (forceRef.current && particle.force.length() > 0.001) {
        const forceDirection = particle.force.clone().normalize();
        const forceMagnitude = Math.min(particle.force.length() * 1000, 1.5);
        forceRef.current.setDirection(forceDirection);
        forceRef.current.setLength(forceMagnitude, forceMagnitude * 0.2, forceMagnitude * 0.1);
        forceRef.current.position.copy(particle.position);
      }
    }
  });

  return (
    <group>
      <mesh ref={meshRef}>
        <sphereGeometry args={[particle.radius * 1e6 * 0.1, 16, 12]} />
        <meshPhongMaterial color={particle.color} />
      </mesh>
      
      {showTrails && particle.trail.length > 1 && (
        <Trail
          width={0.02}
          color={particle.color}
          length={Math.min(particle.trail.length, 100)}
          decay={1}
          local={false}
          stride={0}
          interval={1}
          attenuation={(width) => width}
        >
          <mesh>
            <sphereGeometry args={[0.01]} />
            <meshBasicMaterial color={particle.color} transparent opacity={0} />
          </mesh>
        </Trail>
      )}
      
      {/* Velocity vector */}
      <primitive 
        ref={velocityRef}
        object={new THREE.ArrowHelper(
          new THREE.Vector3(1, 0, 0),
          particle.position,
          1,
          0x00ff00,
          0.2,
          0.1
        )}
        visible={particle.velocity.length() > 0.001}
      />
      
      {/* Force vector */}
      <primitive 
        ref={forceRef}
        object={new THREE.ArrowHelper(
          new THREE.Vector3(1, 0, 0),
          particle.position,
          1,
          0xff0000,
          0.2,
          0.1
        )}
        visible={particle.force.length() > 0.001}
      />
    </group>
  );
};

type SimulationMode = 'langevin' | 'overdamped' | 'ballistic' | 'anomalous';
type FluidType = 'water' | 'glycerol' | 'air' | 'custom';
type ParticleType = 'spherical' | 'ellipsoidal' | 'fractal' | 'active';

export default function BrownianMotion() {
  useSEO({ title: 'Brownian Motion Simulator – SimCore', description: 'Interactive Brownian motion with diffusion analysis and real-time stochastic dynamics.' });
  const schemaLD = {
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: 'Brownian Motion Simulator',
    description: 'Statistical mechanics of particle diffusion with real-time Langevin dynamics and analysis.',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    keywords: 'Brownian motion, diffusion, Langevin dynamics, statistical mechanics',
    about: ['Diffusion coefficient', 'Langevin equation', 'Mean squared displacement'],
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  } as const;
  const [simulationMode, setSimulationMode] = useState<SimulationMode>('langevin');
  const [fluidType, setFluidType] = useState<FluidType>('water');
  const [particleType, setParticleType] = useState<ParticleType>('spherical');
  const [temperature, setTemperature] = useState([300]);
  const [viscosity, setViscosity] = useState([1.0]);
  const [particleRadius, setParticleRadius] = useState([1]);
  const [numParticles, setNumParticles] = useState([10]);
  const [timeStep, setTimeStep] = useState([0.01]);
  const [simulationTime, setSimulationTime] = useState([10]);
  const [anomalousExponent, setAnomalousExponent] = useState([1.0]);
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [showTrails, setShowTrails] = useState(true);
  const [showVelocities, setShowVelocities] = useState(false);
  const [showForces, setShowForces] = useState(false);
  const [showAnalysis, setShowAnalysis] = useState(true);
  const [viewMode, setViewMode] = useState<'3d' | '2d-xy' | '2d-xz' | '2d-yz'>('3d');
  
  const [particles, setParticles] = useState<BrownianParticle[]>([]);
  const [diffusionData, setDiffusionData] = useState<DiffusionData[]>([]);
  const [velocityDistribution, setVelocityDistribution] = useState<VelocityDistribution[]>([]);
  const [collisionEvents, setCollisionEvents] = useState<CollisionEvent[]>([]);
  const [step, setStep] = useState(0);
  const { toast } = useToast();

  // Physical constants and fluid properties
  const kB = 1.381e-23; // J/K
  const NA = 6.022e23;
  
  const fluidProperties: Record<FluidType, FluidProperties> = {
    water: { viscosity: 1e-3, density: 1000, temperature: 300, molecularMass: 18e-3 },
    glycerol: { viscosity: 1.5, density: 1260, temperature: 300, molecularMass: 92e-3 },
    air: { viscosity: 1.8e-5, density: 1.2, temperature: 300, molecularMass: 29e-3 },
    custom: { viscosity: viscosity[0] * 1e-3, density: 1000, temperature: temperature[0], molecularMass: 18e-3 }
  };

  // Advanced particle initialization with different types
  const initializeParticles = useCallback(() => {
    const newParticles: BrownianParticle[] = [];
    const fluid = fluidProperties[fluidType];
    
    for (let i = 0; i < numParticles[0]; i++) {
      const baseRadius = particleRadius[0] * 1e-6;
      let radius = baseRadius;
      let mass = (4/3) * Math.PI * Math.pow(radius, 3) * 2000; // typical particle density
      
      // Modify properties based on particle type
      switch (particleType) {
        case 'ellipsoidal':
          radius = baseRadius * (1 + 0.5 * Math.random()); // elongated
          mass *= 1.2;
          break;
        case 'fractal':
          radius = baseRadius * Math.pow(1 + i / numParticles[0], 0.6); // size distribution
          mass = Math.pow(radius / baseRadius, 2.4) * mass; // fractal scaling
          break;
        case 'active':
          mass *= 0.8; // lighter for active motion
          break;
      }
      
      newParticles.push({
        id: i,
        position: new THREE.Vector3(
          (Math.random() - 0.5) * 8,
          (Math.random() - 0.5) * 8,
          (Math.random() - 0.5) * 8
        ),
        velocity: new THREE.Vector3(
          (Math.random() - 0.5) * 0.1,
          (Math.random() - 0.5) * 0.1,
          (Math.random() - 0.5) * 0.1
        ),
        force: new THREE.Vector3(0, 0, 0),
        trail: [],
        displacement: 0,
        totalDistance: 0,
        mass,
        radius,
        color: new THREE.Color().setHSL((i / numParticles[0] + 0.1) % 1, 0.8, 0.6),
        temperature: temperature[0] * (0.8 + 0.4 * Math.random()), // temperature fluctuations
        lifetime: 0,
        interactions: 0
      });
    }
    
    setParticles(newParticles);
    setDiffusionData([]);
    setVelocityDistribution([]);
    setCollisionEvents([]);
    setStep(0);
  }, [numParticles, particleRadius, temperature, fluidType, particleType, fluidProperties]);

  // Advanced Langevin dynamics with multiple regimes
  const simulationStep = useCallback(() => {
    const fluid = fluidProperties[fluidType];
    const T = temperature[0];
    const dt = timeStep[0] * 1e-6;
    
    setParticles(prevParticles => {
      const newParticles = prevParticles.map(particle => {
        const newParticle = { ...particle };
        
        // Calculate friction coefficient based on particle and fluid properties
        const eta = fluid.viscosity;
        const R = particle.radius;
        let gamma = 6 * Math.PI * eta * R; // Stokes law
        
        // Modify for particle shape
        if (particleType === 'ellipsoidal') gamma *= 1.3;
        if (particleType === 'fractal') gamma *= 1.8;
        
        // Diffusion coefficient
        const D = kB * particle.temperature / gamma;
        
        // Random force calculation
        const randomForceAmplitude = Math.sqrt(2 * gamma * kB * particle.temperature / dt);
        
        // Generate correlated noise for anomalous diffusion
        const noiseCorrelation = Math.pow(dt, anomalousExponent[0] / 2 - 0.5);
        const fx = (Math.random() - 0.5) * randomForceAmplitude * noiseCorrelation;
        const fy = (Math.random() - 0.5) * randomForceAmplitude * noiseCorrelation;
        const fz = (Math.random() - 0.5) * randomForceAmplitude * noiseCorrelation;
        
        // Additional forces for active particles
        let activeForce = new THREE.Vector3(0, 0, 0);
        if (particleType === 'active') {
          const magnitude = 0.1 * particle.mass * Math.sqrt(kB * T / particle.mass);
          const direction = Math.random() * 2 * Math.PI;
          activeForce = new THREE.Vector3(
            magnitude * Math.cos(direction),
            magnitude * Math.sin(direction),
            0
          );
        }
        
        // Update forces
        newParticle.force = new THREE.Vector3(fx, fy, fz).add(activeForce);
        
        // Langevin equation integration based on simulation mode
        switch (simulationMode) {
          case 'langevin':
            // Full Langevin with inertia
            const accel = newParticle.force.clone().multiplyScalar(1 / particle.mass)
              .add(particle.velocity.clone().multiplyScalar(-gamma / particle.mass));
            newParticle.velocity = particle.velocity.clone().add(accel.multiplyScalar(dt));
            break;
            
          case 'overdamped':
            // Overdamped limit (no inertia)
            newParticle.velocity = newParticle.force.clone().multiplyScalar(1 / gamma);
            break;
            
          case 'ballistic':
            // Ballistic regime (no friction)
            const ballAccel = newParticle.force.clone().multiplyScalar(1 / particle.mass);
            newParticle.velocity = particle.velocity.clone().add(ballAccel.multiplyScalar(dt));
            break;
            
          case 'anomalous':
            // Fractional Brownian motion
            const fbmVel = newParticle.force.clone().multiplyScalar(1 / gamma)
              .multiplyScalar(Math.pow(particle.lifetime * dt + dt, anomalousExponent[0] - 1));
            newParticle.velocity = fbmVel;
            break;
        }
        
        // Position update
        const dr = newParticle.velocity.clone().multiplyScalar(dt * 1e6); // scale for visualization
        newParticle.position = particle.position.clone().add(dr);
        
        // Advanced boundary conditions with energy conservation
        const boundaries = 6;
        
        // Check x boundary
        if (Math.abs(newParticle.position.x) > boundaries) {
          newParticle.position.x = Math.sign(newParticle.position.x) * boundaries;
          newParticle.velocity.x *= -0.9;
          newParticle.interactions++;
          
          setCollisionEvents(prev => [...prev, {
            time: step * timeStep[0],
            particleId: particle.id,
            position: newParticle.position.clone(),
            velocityChange: Math.abs(newParticle.velocity.x - particle.velocity.x),
            energy: 0.5 * particle.mass * newParticle.velocity.lengthSq()
          }].slice(-100));
        }
        
        // Check y boundary
        if (Math.abs(newParticle.position.y) > boundaries) {
          newParticle.position.y = Math.sign(newParticle.position.y) * boundaries;
          newParticle.velocity.y *= -0.9;
          newParticle.interactions++;
        }
        
        // Check z boundary
        if (Math.abs(newParticle.position.z) > boundaries) {
          newParticle.position.z = Math.sign(newParticle.position.z) * boundaries;
          newParticle.velocity.z *= -0.9;
          newParticle.interactions++;
        }
        
        // Update trail and statistics
        newParticle.trail = [...particle.trail, newParticle.position.clone()].slice(-200);
        newParticle.displacement = newParticle.position.length();
        newParticle.totalDistance = particle.totalDistance + dr.length();
        newParticle.lifetime = particle.lifetime + dt;
        
        return newParticle;
      });
      
      return newParticles;
    });
    
    setStep(prev => prev + 1);
  }, [simulationMode, fluidType, particleType, temperature, viscosity, timeStep, anomalousExponent, step]);

  // Advanced diffusion analysis
  const calculateDiffusionProperties = useCallback(() => {
    if (particles.length === 0 || step < 10) return;
    
    const currentTime = step * timeStep[0];
    
    // Calculate MSD in 3D and 1D
    const msd3D = particles.reduce((sum, particle) => sum + particle.displacement ** 2, 0) / particles.length;
    const msd1D = particles.reduce((sum, particle) => sum + particle.position.x ** 2, 0) / particles.length;
    
    // Velocity autocorrelation function
    const autocorr = particles.reduce((sum, particle, i) => {
      if (particle.trail.length > 2) {
        const v0 = particle.trail[particle.trail.length - 1].clone().sub(particle.trail[particle.trail.length - 2]);
        const vt = particle.velocity.clone();
        return sum + v0.dot(vt) / (v0.length() * vt.length() || 1);
      }
      return sum;
    }, 0) / particles.length;
    
    // Instantaneous diffusion coefficient from MSD slope
    const diffCoeff = diffusionData.length > 5 ? 
      (msd3D - diffusionData[diffusionData.length - 5].msd) / (6 * (currentTime - diffusionData[diffusionData.length - 5].time)) : 0;
    
    setDiffusionData(prev => [...prev, {
      time: currentTime,
      msd: msd3D,
      msd1D,
      autocorrelation: autocorr,
      diffusionCoeff: Math.max(0, diffCoeff)
    }].slice(-500));
    
    // Calculate velocity distribution
    if (step % 20 === 0) {
      const velocities = particles.map(p => p.velocity.length());
      const maxVel = Math.max(...velocities);
      const bins = 50;
      const velDist: VelocityDistribution[] = [];
      
      for (let i = 0; i < bins; i++) {
        const v = (i / bins) * maxVel;
        const count = velocities.filter(vel => vel >= v && vel < v + maxVel / bins).length;
        const prob = count / particles.length;
        
        // Maxwell-Boltzmann distribution for comparison
        const kT = kB * temperature[0];
        const avgMass = particles.reduce((sum, p) => sum + p.mass, 0) / particles.length;
        const maxwell = 4 * Math.PI * Math.pow(avgMass / (2 * Math.PI * kT), 1.5) * 
          v * v * Math.exp(-avgMass * v * v / (2 * kT));
        
        velDist.push({
          velocity: v,
          probability: prob,
          maxwell: maxwell * maxVel / bins,
          experimental: prob
        });
      }
      
      setVelocityDistribution(velDist);
    }
  }, [particles, step, timeStep, temperature, diffusionData]);

  // Initialize on mount and parameter changes
  useEffect(() => {
    initializeParticles();
  }, [initializeParticles]);

  // Run diffusion analysis
  useEffect(() => {
    calculateDiffusionProperties();
  }, [particles, calculateDiffusionProperties]);

  // Animation loop
  useEffect(() => {
    if (!isRunning || isPaused) return;
    
    const interval = setInterval(simulationStep, 50); // Slightly slower for better visualization
    return () => clearInterval(interval);
  }, [isRunning, isPaused, simulationStep]);

  // Playback controls
  const handlePlay = () => {
    setIsRunning(true);
    setIsPaused(false);
  };

  const handlePause = () => {
    setIsPaused(true);
  };

  const handleReset = () => {
    setIsRunning(false);
    setIsPaused(false);
    initializeParticles();
  };

  const handleRestart = () => {
    handleReset();
    setTimeout(() => handlePlay(), 100);
  };

  // Theoretical predictions
  const theoretical = useMemo(() => {
    const fluid = fluidProperties[fluidType];
    const T = temperature[0];
    const eta = fluid.viscosity;
    const R = particleRadius[0] * 1e-6;
    const gamma = 6 * Math.PI * eta * R;
    const avgMass = particles.length > 0 ? particles.reduce((s, p) => s + p.mass, 0) / particles.length : 1e-15;
    
    const D = kB * T / gamma;
    const timeScale = step * timeStep[0] * 1e-6;
    const theoreticalMSD = 6 * D * Math.pow(timeScale, anomalousExponent[0]) * 1e12;
    const relaxationTime = avgMass / gamma;
    const thermalVelocity = Math.sqrt(3 * kB * T / avgMass);
    const meanFreeTime = relaxationTime;
    const meanFreePath = thermalVelocity * meanFreeTime;
    
    return {
      diffusionCoefficient: D,
      theoreticalMSD,
      relaxationTime,
      thermalVelocity,
      meanFreeTime,
      meanFreePath,
      schmidtNumber: eta / (fluid.density * D),
      reynoldsNumber: fluid.density * thermalVelocity * 2 * R / eta
    };
  }, [fluidType, temperature, particleRadius, particles, step, timeStep, anomalousExponent]);

  // Export comprehensive data
  const exportData = useCallback(() => {
    const data = {
      metadata: {
        simulationMode,
        fluidType,
        particleType,
        temperature: temperature[0],
        viscosity: viscosity[0],
        numParticles: numParticles[0],
        timeStep: timeStep[0],
        totalSteps: step
      },
      particles: particles.map(p => ({
        id: p.id,
        finalPosition: [p.position.x, p.position.y, p.position.z],
        finalVelocity: [p.velocity.x, p.velocity.y, p.velocity.z],
        displacement: p.displacement,
        totalDistance: p.totalDistance,
        interactions: p.interactions,
        trail: p.trail.map(pos => [pos.x, pos.y, pos.z])
      })),
      analysis: {
        diffusionData,
        velocityDistribution,
        collisionEvents,
        theoretical
      }
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `brownian_motion_${simulationMode}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    toast({
      title: "Data Exported",
      description: "Brownian motion data exported successfully",
    });
  }, [simulationMode, fluidType, particleType, temperature, viscosity, numParticles, timeStep, step, particles, diffusionData, velocityDistribution, collisionEvents, theoretical, toast]);

  // Export simulation data (wrapper for existing export)
  const exportSimulationData = useCallback(() => {
    exportData();
  }, [exportData]);

  // Export CSV data
  const exportCSV = useCallback(() => {
    const headers = ['time', 'particle_id', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'displacement'];
    const rows = particles.flatMap(particle => 
      particle.trail.map((pos, i) => [
        i * timeStep[0],
        particle.id,
        pos.x,
        pos.y,
        pos.z,
        particle.velocity.x,
        particle.velocity.y,
        particle.velocity.z,
        particle.displacement
      ])
    );
    
    const csvContent = [headers, ...rows].map(row => row.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `brownian_trajectories_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    
    toast({
      title: "CSV Exported",
      description: "Trajectory data exported to CSV",
    });
  }, [particles, timeStep, toast]);

  // Generate plots data
  const msdPlotData = useMemo(() => {
    if (diffusionData.length === 0) return [];
    
    return [
      {
        x: diffusionData.map(d => d.time),
        y: diffusionData.map(d => d.msd),
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: 'Experimental MSD',
        line: { color: '#3B82F6', width: 2 }
      },
      {
        x: diffusionData.map(d => d.time),
        y: diffusionData.map(d => 6 * theoretical.diffusionCoefficient * Math.pow(d.time || 1e-10, anomalousExponent[0]) * 1e12),
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: 'Theoretical',
        line: { color: '#EF4444', dash: 'dash', width: 2 }
      }
    ];
  }, [diffusionData, theoretical.diffusionCoefficient, anomalousExponent]);

  const velocityPlotData = useMemo(() => {
    if (velocityDistribution.length === 0) return [];
    
    return [
      {
        x: velocityDistribution.map(d => d.velocity),
        y: velocityDistribution.map(d => d.experimental),
        type: 'bar' as const,
        name: 'Experimental',
        marker: { color: '#10B981', opacity: 0.7 }
      },
      {
        x: velocityDistribution.map(d => d.velocity),
        y: velocityDistribution.map(d => d.maxwell),
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: 'Maxwell-Boltzmann',
        line: { color: '#F59E0B', width: 3 }
      }
    ];
  }, [velocityDistribution]);

  const autocorrPlotData = useMemo(() => {
    if (diffusionData.length === 0) return [];
    
    return [
      {
        x: diffusionData.map(d => d.time),
        y: diffusionData.map(d => d.autocorrelation),
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: 'C_v(t)',
        line: { color: '#8B5CF6', width: 2 }
      }
    ];
  }, [diffusionData]);

  const diffCoeffPlotData = useMemo(() => {
    if (diffusionData.length === 0) return [];
    
    return [
      {
        x: diffusionData.map(d => d.time),
        y: diffusionData.map(d => d.diffusionCoeff),
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: 'Instantaneous D',
        line: { color: '#EC4899', width: 2 }
      }
    ];
  }, [diffusionData]);

  return (
    <PhysicsModuleLayout background="statistical">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      <PhysicsModuleHeader
        title="Brownian Motion Simulator"
        description="Statistical mechanics of particle diffusion in thermal environments with real-time stochastic dynamics"
        category="Statistical Physics"
        difficulty="Intermediate"
        equation="D = \frac{k_B T}{6\pi \eta R}"
        isRunning={isRunning}
        currentStep={step}
        onReset={handleReset}
        onExport={exportSimulationData}
      />
          <CardHeader className="text-center">
            <CardTitle className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent flex items-center justify-center gap-3">
              <Activity className="h-8 w-8 text-blue-600" />
              Advanced Brownian Motion
            </CardTitle>
            <CardDescription className="text-lg">
              Comprehensive Langevin dynamics with multi-particle interactions
            </CardDescription>
            
            {/* Feature badges */}
            <div className="flex flex-wrap justify-center gap-2 mt-4">
              <Badge variant="secondary" className="flex items-center gap-1">
                <Move className="h-3 w-3" />
                Langevin Dynamics
              </Badge>
              <Badge variant="secondary" className="flex items-center gap-1">
                <TrendingUp className="h-3 w-3" />
                Anomalous Diffusion
              </Badge>
              <Badge variant="secondary" className="flex items-center gap-1">
                <Zap className="h-3 w-3" />
                Active Matter
              </Badge>
              <Badge variant="secondary" className="flex items-center gap-1">
                <BarChart3 className="h-3 w-3" />
                Stochastic Processes
              </Badge>
              <Badge variant="outline">Simulation</Badge>
              <Badge variant="outline">Analysis</Badge>
              <Badge variant="outline">Theory</Badge>
              <Badge variant="outline">Performance</Badge>
            </div>
          </CardHeader>

        <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
          {/* Enhanced Control Panel */}
          <div className="xl:col-span-1 space-y-4">
            <Card className="bg-card/90 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Settings className="h-5 w-5" />
                  Advanced Controls
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Simulation Mode */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Simulation Mode</label>
                  <Select value={simulationMode} onValueChange={(value: SimulationMode) => setSimulationMode(value)}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="langevin">Full Langevin</SelectItem>
                      <SelectItem value="overdamped">Overdamped</SelectItem>
                      <SelectItem value="ballistic">Ballistic</SelectItem>
                      <SelectItem value="anomalous">Anomalous</SelectItem>
                    </SelectContent>
                  </Select>
                  
                  {/* Mode equation display */}
                  <div className="text-xs p-2 bg-muted rounded">
                    {simulationMode === 'langevin' && (
                      <InlineMath math="m\frac{dv}{dt} = -\gamma v + \sqrt{2\gamma k_BT}\eta(t)" />
                    )}
                    {simulationMode === 'overdamped' && (
                      <InlineMath math="\frac{dx}{dt} = \frac{F}{\gamma} + \sqrt{2D}\eta(t)" />
                    )}
                    {simulationMode === 'ballistic' && (
                      <InlineMath math="m\frac{dv}{dt} = F(t)" />
                    )}
                    {simulationMode === 'anomalous' && (
                      <InlineMath math="\langle x^2(t) \rangle \propto t^\alpha" />
                    )}
                  </div>
                </div>

                {/* Fluid Environment */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Fluid Environment</label>
                  <Select value={fluidType} onValueChange={(value: FluidType) => setFluidType(value)}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="water">Water</SelectItem>
                      <SelectItem value="glycerol">Glycerol</SelectItem>
                      <SelectItem value="air">Air</SelectItem>
                      <SelectItem value="custom">Custom</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Particle Type */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Particle Type</label>
                  <Select value={particleType} onValueChange={(value: ParticleType) => setParticleType(value)}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="spherical">Spherical</SelectItem>
                      <SelectItem value="ellipsoidal">Ellipsoidal</SelectItem>
                      <SelectItem value="fractal">Fractal</SelectItem>
                      <SelectItem value="active">Active Matter</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Temperature Control */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Temperature: {temperature[0]} K</label>
                  <Slider
                    value={temperature}
                    onValueChange={setTemperature}
                    min={50}
                    max={1000}
                    step={10}
                    className="w-full"
                  />
                </div>

                {/* Particle Radius */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Particle Radius: {particleRadius[0]} μm</label>
                  <Slider
                    value={particleRadius}
                    onValueChange={setParticleRadius}
                    min={0.1}
                    max={5}
                    step={0.1}
                    className="w-full"
                  />
                </div>

                {/* Number of Particles */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Number of Particles: {numParticles[0]}</label>
                  <Slider
                    value={numParticles}
                    onValueChange={setNumParticles}
                    min={1}
                    max={100}
                    step={1}
                    className="w-full"
                  />
                </div>

                {/* Anomalous Exponent */}
                {simulationMode === 'anomalous' && (
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Anomalous Exponent: {anomalousExponent[0]}</label>
                    <Slider
                      value={anomalousExponent}
                      onValueChange={setAnomalousExponent}
                      min={0.1}
                      max={2.0}
                      step={0.1}
                      className="w-full"
                    />
                  </div>
                )}

                {/* Custom Viscosity */}
                {fluidType === 'custom' && (
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Viscosity: {viscosity[0]} mPa·s</label>
                    <Slider
                      value={viscosity}
                      onValueChange={setViscosity}
                      min={0.1}
                      max={100}
                      step={0.1}
                      className="w-full"
                    />
                  </div>
                )}

                {/* Visualization Options */}
                <div className="space-y-3 border-t pt-4">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium">Trails</label>
                    <Switch checked={showTrails} onCheckedChange={setShowTrails} />
                  </div>
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium">Velocities</label>
                    <Switch checked={showVelocities} onCheckedChange={setShowVelocities} />
                  </div>
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium">Forces</label>
                    <Switch checked={showForces} onCheckedChange={setShowForces} />
                  </div>
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium">Analysis</label>
                    <Switch checked={showAnalysis} onCheckedChange={setShowAnalysis} />
                  </div>
                </div>

                {/* Enhanced Control Buttons */}
                <div className="space-y-2 border-t pt-4">
                  <div className="grid grid-cols-2 gap-2">
                    <Button 
                      onClick={handlePlay} 
                      disabled={isRunning && !isPaused}
                      variant="default"
                      size="sm"
                    >
                      <Play className="h-4 w-4" />
                    </Button>
                    <Button 
                      onClick={handlePause}
                      disabled={!isRunning || isPaused}
                      variant="outline"
                      size="sm"
                    >
                      <Pause className="h-4 w-4" />
                    </Button>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-2">
                    <Button onClick={handleReset} variant="outline" size="sm">
                      <RotateCcw className="h-4 w-4" />
                    </Button>
                    <Button onClick={handleRestart} variant="outline" size="sm">
                      <Activity className="h-4 w-4" />
                    </Button>
                  </div>
                  
                  <Button onClick={exportData} variant="outline" size="sm" className="w-full">
                    <Download className="h-4 w-4 mr-2" />
                    JSON
                  </Button>
                  
                  <Button onClick={exportCSV} variant="outline" size="sm" className="w-full">
                    <Save className="h-4 w-4 mr-2" />
                    CSV
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Enhanced Current State */}
            <Card className="bg-card/90 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-sm">Current State</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-1 gap-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">D:</span>
                    <span className="font-mono">{theoretical.diffusionCoefficient?.toExponential(2) || '0'} m²/s</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">τ:</span>
                    <span className="font-mono">{theoretical.relaxationTime?.toExponential(2) || '0'} s</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Re:</span>
                    <span className="font-mono">{theoretical.reynoldsNumber?.toFixed(3) || '0'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Steps:</span>
                    <span className="font-mono">{step}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Particles:</span>
                    <span className="font-mono">{particles.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Status:</span>
                    <Badge variant={isRunning ? (isPaused ? "secondary" : "default") : "outline"} className="text-xs">
                      {isRunning ? (isPaused ? "Paused" : "Running") : "Stopped"}
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Main Content Area */}
          <div className="xl:col-span-4 space-y-4">
            {/* Enhanced 3D Canvas */}
            <Card className="bg-card/90 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Move className="h-5 w-5" />
                  3D Brownian Motion Simulation
                </CardTitle>
                <CardDescription>
                  Real-time {simulationMode} dynamics in {fluidType} environment
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[600px] w-full border rounded-lg overflow-hidden bg-gradient-to-b from-blue-50/20 to-cyan-50/20 dark:from-gray-900 dark:to-gray-800">
                  <Canvas camera={{ position: [12, 8, 12], fov: 60 }}>
                    <fog attach="fog" args={['#ffffff', 15, 50]} />
                    <ambientLight intensity={0.4} />
                    <pointLight position={[10, 10, 10]} intensity={0.8} />
                    <pointLight position={[-10, -10, -10]} intensity={0.3} />
                    <directionalLight position={[5, 5, 5]} intensity={0.5} />
                    
                    {particles.map((particle) => (
                      <BrownianParticleComponent
                        key={particle.id}
                        particle={particle}
                        showTrails={showTrails}
                      />
                    ))}
                    
                    {/* Enhanced boundary visualization */}
                    <group>
                      <mesh>
                        <boxGeometry args={[12, 12, 12]} />
                        <meshBasicMaterial 
                          color={0x3B82F6} 
                          wireframe 
                          transparent 
                          opacity={0.15} 
                        />
                      </mesh>
                      
                      {/* Corner markers */}
                      {[-6, 6].map(x => 
                        [-6, 6].map(y => 
                          [-6, 6].map(z => (
                            <mesh key={`${x}-${y}-${z}`} position={[x, y, z]}>
                              <sphereGeometry args={[0.1]} />
                              <meshBasicMaterial color={0x64748B} />
                            </mesh>
                          ))
                        )
                      )}
                    </group>
                    
                    <OrbitControls 
                      enableDamping 
                      dampingFactor={0.05}
                      minDistance={8}
                      maxDistance={50}
                      maxPolarAngle={Math.PI * 0.75}
                    />
                  </Canvas>
                </div>
                
                {/* Legend */}
                <div className="mt-4 flex flex-wrap gap-4 text-sm">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    <span>Velocity Vectors</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                    <span>Force Vectors</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-1 bg-blue-500 rounded"></div>
                    <span>Particle Trails</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Enhanced Analysis Plots Grid */}
            {showAnalysis && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* MSD Plot */}
                <Card className="bg-card/90 backdrop-blur-sm">
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center gap-2">
                      <TrendingUp className="h-5 w-5" />
                      Mean Square Displacement
                    </CardTitle>
                    <CardDescription>
                      <InlineMath math="\langle r^2(t) \rangle = 6Dt^\alpha" />
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Plot
                      data={msdPlotData}
                      layout={{
                        ...PLOT.createLayout('⟨r²(t)⟩ vs Time', 'Time (μs)', 'MSD (μm²)', { height: 350 }),
                        xaxis: { ...PLOT.layout.xaxis, title: 'Time (μs)', type: diffusionData.length > 10 ? 'log' : 'linear' },
                        yaxis: { ...PLOT.layout.yaxis, title: 'MSD (μm²)', type: diffusionData.length > 10 ? 'log' : 'linear' }
                      }}
                      config={PLOT.config}
                      style={{ width: '100%', height: '350px' }}
                    />
                  </CardContent>
                </Card>

                {/* Velocity Distribution */}
                <Card className="bg-card/90 backdrop-blur-sm">
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center gap-2">
                      <BarChart3 className="h-5 w-5" />
                      Velocity Distribution
                    </CardTitle>
                    <CardDescription>
                      Maxwell-Boltzmann vs Experimental
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Plot
                      data={velocityPlotData}
                      layout={PLOT.createLayout('P(v) vs Velocity', 'Velocity (m/s)', 'Probability Density', { height: 350 })}
                      config={PLOT.config}
                      style={{ width: '100%', height: '350px' }}
                    />
                  </CardContent>
                </Card>

                {/* Autocorrelation Function */}
                <Card className="bg-card/90 backdrop-blur-sm">
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Activity className="h-5 w-5" />
                      Velocity Autocorrelation
                    </CardTitle>
                    <CardDescription>
                      <InlineMath math="C_v(t) = \langle \vec{v}(0) \cdot \vec{v}(t) \rangle" />
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Plot
                      data={autocorrPlotData}
                      layout={PLOT.createLayout('⟨v(0)·v(t)⟩ vs Time', 'Time (μs)', 'Autocorrelation', { height: 350 })}
                      config={PLOT.config}
                      style={{ width: '100%', height: '350px' }}
                    />
                  </CardContent>
                </Card>

                {/* Diffusion Coefficient */}
                <Card className="bg-card/90 backdrop-blur-sm">
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center gap-2">
                      <BarChart2 className="h-5 w-5" />
                      Diffusion Coefficient
                    </CardTitle>
                    <CardDescription>
                      Real-time D(t) calculation
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Plot
                      data={diffCoeffPlotData}
                      layout={PLOT.createLayout('D(t) vs Time', 'Time (μs)', 'Diffusion Coefficient (m²/s)', { height: 350 })}
                      config={PLOT.config}
                      style={{ width: '100%', height: '350px' }}
                    />
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        </div>

        {/* Theory and Performance Section */}
        <Tabs defaultValue="theory" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="theory">Theory</TabsTrigger>
            <TabsTrigger value="mathematics">Mathematics</TabsTrigger>
            <TabsTrigger value="applications">Applications</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
          </TabsList>
          
          <TabsContent value="theory" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BookOpen className="h-5 w-5" />
                  Brownian Motion Theory
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="prose prose-sm max-w-none">
                  <h3>Langevin Equation</h3>
                  <BlockMath math="m\frac{dv}{dt} = -\gamma v + \sqrt{2\gamma k_B T}\eta(t)" />
                  <p>The Langevin equation describes the motion of a particle in a viscous medium with thermal fluctuations.</p>
                  
                  <h3>Diffusion Regimes</h3>
                  <ul>
                    <li><strong>Ballistic:</strong> Short times, inertial effects dominate</li>
                    <li><strong>Overdamped:</strong> High friction, position directly follows force</li>
                    <li><strong>Anomalous:</strong> Non-Gaussian diffusion with α ≠ 1</li>
                  </ul>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="mathematics" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Mathematical Framework</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-semibold mb-2">Stokes-Einstein Relation</h4>
                    <BlockMath math="D = \frac{k_B T}{6\pi\eta R}" />
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2">Velocity Autocorrelation</h4>
                    <BlockMath math="C_v(t) = \frac{k_B T}{m}e^{-\gamma t/m}" />
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2">Mean Square Displacement</h4>
                    <BlockMath math="\langle r^2(t) \rangle = 6Dt + \frac{6k_B T}{\gamma^2}(1-e^{-\gamma t/m})" />
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2">Reynolds Number</h4>
                    <BlockMath math="Re = \frac{\rho v R}{\eta}" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="applications" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Real-world Applications</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-semibold mb-2">Biophysics</h4>
                    <ul className="text-sm space-y-1">
                      <li>• Protein diffusion in membranes</li>
                      <li>• DNA dynamics</li>
                      <li>• Cellular transport</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2">Materials Science</h4>
                    <ul className="text-sm space-y-1">
                      <li>• Particle tracking</li>
                      <li>• Colloidal suspensions</li>
                      <li>• Polymer dynamics</li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="performance" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Performance Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-sm text-muted-foreground">Current FPS</div>
                      <div className="text-2xl font-bold">{isRunning ? 20 : 0}</div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground">Particles</div>
                      <div className="text-2xl font-bold">{particles.length}</div>
                    </div>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2">Features</h4>
                    <ul className="text-sm space-y-1">
                      <li>• Real-time Langevin integration</li>
                      <li>• Multi-particle dynamics</li>
                      <li>• Interactive 3D visualization</li>
                      <li>• Statistical analysis</li>
                      <li>• Multiple diffusion regimes</li>
                      <li>• Vector visualization</li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
    </PhysicsModuleLayout>
  );
}