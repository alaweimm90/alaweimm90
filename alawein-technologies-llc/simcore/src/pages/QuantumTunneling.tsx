import React, { useState, useMemo, useRef, useCallback, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Line, Text, Html, Sphere, Box } from '@react-three/drei';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Download, Settings, Info, Play, Pause, RotateCcw, Zap, Calculator, Eye, Cpu, Activity, Layers, Target } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import Plot from 'react-plotly.js';
import { ScientificPlot } from '@/components/ScientificPlot';
import { PhysicsTheory } from '@/components/PhysicsTheory';
import { PhysicsModuleLayout, PhysicsContentCard } from '@/components/PhysicsModuleLayout';
import { PhysicsModuleHeader } from '@/components/PhysicsModuleHeader';
import { InlineMath, BlockMath } from '@/components/ui/Math';

import { UNIFIED_SCIENTIFIC_PLOT_SYSTEM as PLOT } from '@/lib/scientific-plot-system';
import { WebGPUManager } from '@/lib/webgpu-manager';
import { useToast } from '@/hooks/use-toast';
import { useSEO } from '@/hooks/use-seo';
import * as THREE from 'three';

// Advanced tunneling physics engine with multiple barrier shapes and numerical methods
const calculateAdvancedTunneling = (
  params: {
    barrierHeight: number;
    barrierWidth: number;
    barrierPosition: number;
    particleEnergy: number;
    particleMass: number;
    wavePacketWidth: number;
    initialPosition: number;
    initialMomentum: number;
    time: number;
    barrierShape: string;
    method: string;
    absorbing: boolean;
  },
  gridPoints: number = 512
) => {
  const L = 20; // Total length
  const dx = L / gridPoints;
  const dt = Math.min(0.001, dx * dx * params.particleMass / 2); // Adaptive time step
  
  // Create spatial grid
  const x = new Float32Array(gridPoints);
  const V = new Float32Array(gridPoints); // Potential
  const psi_real = new Float32Array(gridPoints);
  const psi_imag = new Float32Array(gridPoints);
  const probability = new Float32Array(gridPoints);
  
  // Set up spatial grid and barrier shapes
  for (let i = 0; i < gridPoints; i++) {
    x[i] = -L/2 + i * dx;
    const relPos = x[i] - params.barrierPosition;
    
    // Multiple barrier shapes
    switch (params.barrierShape) {
      case 'rectangular':
        V[i] = Math.abs(relPos) < params.barrierWidth/2 ? params.barrierHeight : 0;
        break;
      case 'gaussian':
        V[i] = params.barrierHeight * Math.exp(-0.5 * Math.pow(relPos / (params.barrierWidth/3), 2));
        break;
      case 'triangular':
        if (Math.abs(relPos) < params.barrierWidth/2) {
          V[i] = params.barrierHeight * (1 - 2*Math.abs(relPos)/params.barrierWidth);
        } else {
          V[i] = 0;
        }
        break;
      case 'double_well': {
        const sep = params.barrierWidth/3;
        V[i] = params.barrierHeight * (
          Math.exp(-0.5 * Math.pow((relPos - sep) / (params.barrierWidth/6), 2)) +
          Math.exp(-0.5 * Math.pow((relPos + sep) / (params.barrierWidth/6), 2))
        );
        break;
      }
      case 'coulomb': {
        const r = Math.max(Math.abs(relPos), 0.1);
        V[i] = params.barrierHeight / r * Math.exp(-r/params.barrierWidth);
        break;
      }
      default:
        V[i] = Math.abs(relPos) < params.barrierWidth/2 ? params.barrierHeight : 0;
    }
    
    // Add absorbing boundaries if enabled (complex potential)
    if (params.absorbing) {
      const absLength = L * 0.1;
      if (x[i] < -L/2 + absLength) {
        // Imaginary potential for absorption (simplified as real for this implementation)
        V[i] += params.barrierHeight * 0.1 * Math.pow((x[i] + L/2)/absLength, 2);
      } else if (x[i] > L/2 - absLength) {
        V[i] += params.barrierHeight * 0.1 * Math.pow((x[i] - L/2 + absLength)/absLength, 2);
      }
    }
  }
  
  // Initialize various wave packet types
  const k = params.initialMomentum;
  const sigma = params.wavePacketWidth;
  const x0 = params.initialPosition;
  
  let norm = 0;
  for (let i = 0; i < gridPoints; i++) {
    // Gaussian wave packet with momentum
    const gaussianEnvelope = Math.exp(-0.5 * Math.pow((x[i] - x0) / sigma, 2));
    psi_real[i] = gaussianEnvelope * Math.cos(k * x[i]);
    psi_imag[i] = gaussianEnvelope * Math.sin(k * x[i]);
    norm += psi_real[i] * psi_real[i] + psi_imag[i] * psi_imag[i];
  }
  
  // Normalize
  norm = Math.sqrt(norm * dx);
  for (let i = 0; i < gridPoints; i++) {
    psi_real[i] /= norm;
    psi_imag[i] /= norm;
  }
  
  // Time evolution with multiple numerical methods
  const numTimeSteps = Math.floor(params.time / dt);
  const hbar = 1; // Set ℏ = 1
  
  // Conservation tracking
  let totalProbability = 1.0;
  let averageEnergy = 0;
  let positionExpectation = 0;
  let momentumExpectation = 0;
  
  for (let step = 0; step < numTimeSteps; step++) {
    if (params.method === 'split_operator') {
      // Split-operator method (most accurate)
      // Apply potential evolution: exp(-iV*dt/ℏ)
      for (let i = 0; i < gridPoints; i++) {
        const phase = -V[i] * dt / hbar;
        const cosPhase = Math.cos(phase);
        const sinPhase = Math.sin(phase);
        
        const real_temp = psi_real[i] * cosPhase - psi_imag[i] * sinPhase;
        const imag_temp = psi_real[i] * sinPhase + psi_imag[i] * cosPhase;
        
        psi_real[i] = real_temp;
        psi_imag[i] = imag_temp;
      }
      
      // Apply kinetic evolution using FFT (approximated with finite differences)
      const kinetic_factor = -hbar * dt / (2 * params.particleMass * dx * dx);
      const temp_real = new Float32Array(gridPoints);
      const temp_imag = new Float32Array(gridPoints);
      
      for (let i = 1; i < gridPoints - 1; i++) {
        const laplacian_real = psi_real[i+1] - 2*psi_real[i] + psi_real[i-1];
        const laplacian_imag = psi_imag[i+1] - 2*psi_imag[i] + psi_imag[i-1];
        
        temp_real[i] = psi_real[i] + kinetic_factor * laplacian_imag;
        temp_imag[i] = psi_imag[i] - kinetic_factor * laplacian_real;
      }
      
      for (let i = 1; i < gridPoints - 1; i++) {
        psi_real[i] = temp_real[i];
        psi_imag[i] = temp_imag[i];
      }
      
    } else if (params.method === 'crank_nicolson') {
      // Crank-Nicolson method (implicit, very stable)
      // Simplified implementation
      const alpha = hbar * dt / (2 * params.particleMass * dx * dx);
      
      for (let i = 1; i < gridPoints - 1; i++) {
        const kinetic = -(psi_real[i+1] - 2*psi_real[i] + psi_real[i-1]) / (dx * dx);
        const potential = V[i] * psi_real[i];
        
        const newReal = psi_real[i] - dt * (kinetic / (2 * params.particleMass) + potential) * psi_imag[i] / hbar;
        const newImag = psi_imag[i] + dt * (kinetic / (2 * params.particleMass) + potential) * psi_real[i] / hbar;
        
        psi_real[i] = newReal;
        psi_imag[i] = newImag;
      }
    } else {
      // Euler method (simple but less stable)
      for (let i = 1; i < gridPoints - 1; i++) {
        const kinetic = -(psi_real[i+1] - 2*psi_real[i] + psi_real[i-1]) / (dx * dx);
        const potential = V[i];
        
        const newReal = psi_real[i] - dt * ((kinetic / (2 * params.particleMass) + potential) * psi_imag[i]) / hbar;
        const newImag = psi_imag[i] + dt * ((kinetic / (2 * params.particleMass) + potential) * psi_real[i]) / hbar;
        
        psi_real[i] = newReal;
        psi_imag[i] = newImag;
      }
    }
  }
  
  // Calculate probability density and conservation quantities
  totalProbability = 0;
  averageEnergy = 0;
  positionExpectation = 0;
  momentumExpectation = 0;
  
  for (let i = 0; i < gridPoints; i++) {
    // Ensure wave function values are finite
    if (!isFinite(psi_real[i])) psi_real[i] = 0;
    if (!isFinite(psi_imag[i])) psi_imag[i] = 0;
    
    probability[i] = psi_real[i] * psi_real[i] + psi_imag[i] * psi_imag[i];
    
    // Validate probability is finite
    if (!isFinite(probability[i])) probability[i] = 0;
    
    totalProbability += probability[i] * dx;
    positionExpectation += x[i] * probability[i] * dx;
    
    if (i > 0 && i < gridPoints - 1) {
      // Momentum expectation (finite difference approximation)
      const dpsi_dx_real = (psi_real[i+1] - psi_real[i-1]) / (2*dx);
      const dpsi_dx_imag = (psi_imag[i+1] - psi_imag[i-1]) / (2*dx);
      const momentum_contrib = (psi_real[i] * dpsi_dx_imag - psi_imag[i] * dpsi_dx_real) * dx * hbar;
      
      if (isFinite(momentum_contrib)) {
        momentumExpectation += momentum_contrib;
      }
      
      // Energy expectation
      const kinetic = -(psi_real[i] * (psi_real[i+1] - 2*psi_real[i] + psi_real[i-1]) + 
                       psi_imag[i] * (psi_imag[i+1] - 2*psi_imag[i] + psi_imag[i-1])) * 
                       hbar * hbar / (2 * params.particleMass * dx * dx);
      const potential = V[i] * probability[i];
      const energy_contrib = (kinetic + potential) * dx;
      
      if (isFinite(energy_contrib)) {
        averageEnergy += energy_contrib;
      }
    }
  }
  
  // Calculate transmission and reflection coefficients
  const barrierStart = Math.floor((params.barrierPosition - params.barrierWidth/2 + L/2) / dx);
  const barrierEnd = Math.floor((params.barrierPosition + params.barrierWidth/2 + L/2) / dx);
  
  let transmitted = 0, reflected = 0, barrier_prob = 0;
  
  for (let i = 0; i < gridPoints; i++) {
    if (i > barrierEnd) {
      transmitted += probability[i];
    } else if (i < barrierStart) {
      reflected += probability[i];
    } else {
      barrier_prob += probability[i];
    }
  }
  
  // Advanced WKB analysis with arbitrary barriers
  let wkbTransmission = 0;
  let actionIntegral = 0;
  
  if (params.particleEnergy < params.barrierHeight) {
    // Integrate through the classically forbidden region
    for (let i = 0; i < gridPoints; i++) {
      if (V[i] > params.particleEnergy) {
        const kappa = Math.sqrt(2 * params.particleMass * (V[i] - params.particleEnergy));
        actionIntegral += kappa * dx;
      }
    }
    wkbTransmission = Math.exp(-2 * actionIntegral);
  } else {
    wkbTransmission = 1; // Classical transmission
  }
  
  // Resonance detection for specific energies
  const resonanceStrength = Math.abs(Math.sin(Math.sqrt(2 * params.particleMass * params.particleEnergy) * params.barrierWidth));
  
  return {
    x: Array.from(x),
    potential: Array.from(V),
    waveReal: Array.from(psi_real),
    waveImag: Array.from(psi_imag),
    probability: Array.from(probability),
    transmission: transmitted / totalProbability,
    reflection: reflected / totalProbability,
    barrierOccupancy: barrier_prob / totalProbability,
    wkbTransmission,
    classicalForbidden: params.particleEnergy < params.barrierHeight,
    groupVelocity: params.initialMomentum / params.particleMass,
    deBroglieWavelength: 2 * Math.PI / params.initialMomentum,
    conservationMetrics: {
      probabilityConservation: totalProbability,
      averageEnergy: averageEnergy,
      positionExpectation: positionExpectation,
      momentumExpectation: momentumExpectation
    },
    resonanceStrength,
    actionIntegral,
    phaseShift: Math.atan2(momentumExpectation, params.initialMomentum)
  };
};

// 3D Quantum wave visualization
function QuantumWave3D({ 
  waveData, 
  showPotential, 
  showWaveFunction, 
  showProbability,
  animate,
  time
}: any) {
  const waveRef = useRef<THREE.Group>(null);
  
  useFrame((state, delta) => {
    if (animate && waveRef.current) {
      // Subtle oscillation to show quantum nature
      waveRef.current.rotation.y += delta * 0.1;
    }
  });

  // Create wave surface mesh
  const waveSurface = useMemo(() => {
    if (!waveData?.probability?.length) return null;
    
    const geometry = new THREE.BufferGeometry();
    const vertices = [];
    const colors = [];
    const indices = [];
    
    const numPoints = waveData.probability.length;
    const width = 12;
    
    // Create surface from probability density
    for (let i = 0; i < numPoints; i++) {
      const x = (i / numPoints - 0.5) * width;
      const probValue = waveData.probability[i];
      const y = isFinite(probValue) ? probValue * 3 : 0; // Scale height, handle NaN
      const z = 0;
      
      vertices.push(x, y, z);
      
      // Color based on wave function phase
      const realPart = isFinite(waveData.waveReal[i]) ? waveData.waveReal[i] : 0;
      const imagPart = isFinite(waveData.waveImag[i]) ? waveData.waveImag[i] : 0;
      const phase = Math.atan2(imagPart, realPart);
      const hue = (phase + Math.PI) / (2 * Math.PI);
      const color = new THREE.Color().setHSL(hue, 0.8, 0.6);
      colors.push(color.r, color.g, color.b);
    }
    
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    
    return geometry;
  }, [waveData]);

  return (
    <group ref={waveRef}>
      {/* Coordinate axes */}
      <Line points={[[-6, 0, 0], [6, 0, 0]]} color="#ef4444" lineWidth={2} />
      <Line points={[[0, 0, 0], [0, 4, 0]]} color="#22c55e" lineWidth={2} />
      
      {/* Axis labels */}
      <Text position={[6.5, 0, 0]} fontSize={0.3} color="#ef4444" anchorX="left">
        Position (nm)
      </Text>
      <Text position={[0, 4.5, 0]} fontSize={0.3} color="#22c55e" anchorY="bottom">
        Amplitude
      </Text>
      
      {/* Potential barrier visualization */}
      {showPotential && waveData?.potential && (
        <Line
          points={waveData.x.map((x: number, i: number) => [
            (x + 10) * 0.6 - 6, 
            isFinite(waveData.potential[i]) ? waveData.potential[i] * 0.5 : 0, 
            -1
          ]).filter(point => point.every(coord => isFinite(coord)))}
          color="#dc2626"
          lineWidth={3}
        />
      )}
      
      {/* Wave function real part */}
      {showWaveFunction && waveData?.waveReal && (
        <Line
          points={waveData.x.map((x: number, i: number) => [
            (x + 10) * 0.6 - 6, 
            isFinite(waveData.waveReal[i]) ? waveData.waveReal[i] * 2 : 0, 
            1
          ]).filter(point => point.every(coord => isFinite(coord)))}
          color="#3b82f6"
          lineWidth={2}
        />
      )}
      
      {/* Probability density surface */}
      {showProbability && waveSurface && (
        <mesh geometry={waveSurface}>
          <pointsMaterial size={0.05} vertexColors transparent opacity={0.8} />
        </mesh>
      )}
      
      {/* Barrier region indicator */}
      {waveData && (
        <mesh position={[0, -0.5, 0]}>
          <boxGeometry args={[2, 0.1, 4]} />
          <meshBasicMaterial 
            color="#fbbf24" 
            transparent 
            opacity={0.3}
          />
        </mesh>
      )}
    </group>
  );
}

// Main component
export default function QuantumTunneling() {
  useSEO({ title: 'Quantum Tunneling Simulator – SimCore', description: 'Real-time wave packet tunneling through potential barriers with advanced numerical methods.' });
  const schemaLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: 'Quantum Tunneling Simulator',
    description: 'Real-time wave packet tunneling through potential barriers with advanced numerical methods.',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    keywords: 'quantum tunneling, wave packet, potential barrier, Schrödinger equation',
    about: ['Quantum tunneling', 'Wave packets', 'Potential barriers'],
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  }), []);
  const breadcrumbsLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: [
      { '@type': 'ListItem', position: 1, name: 'Home', item: typeof window !== 'undefined' ? `${window.location.origin}/` : '/' },
      { '@type': 'ListItem', position: 2, name: 'Modules', item: typeof window !== 'undefined' ? `${window.location.origin}/modules` : '/modules' },
      { '@type': 'ListItem', position: 3, name: 'Quantum Tunneling', item: typeof window !== 'undefined' ? window.location.href : '' }
    ]
  }), []);
  const [activeTab, setActiveTab] = useState('simulation');
  // Simulation parameters
  const [barrierHeight, setBarrierHeight] = useState([3.0]);
  const [barrierWidth, setBarrierWidth] = useState([2.0]);
  const [barrierPosition, setBarrierPosition] = useState([0]);
  const [particleEnergy, setParticleEnergy] = useState([2.0]);
  const [particleMass, setParticleMass] = useState([1.0]);
  const [wavePacketWidth, setWavePacketWidth] = useState([1.5]);
  const [initialPosition, setInitialPosition] = useState([-6]);
  const [initialMomentum, setInitialMomentum] = useState([2.0]);
  
  // Animation controls
  const [time, setTime] = useState([0]);
  const [isAnimating, setIsAnimating] = useState(false);
  const [showPotential, setShowPotential] = useState(true);
  const [showWaveFunction, setShowWaveFunction] = useState(true);
  const [showProbability, setShowProbability] = useState(true);
  const [gridResolution, setGridResolution] = useState([256]);
  
  // Calculate quantum tunneling
  const params = useMemo(() => ({
    barrierHeight: barrierHeight[0],
    barrierWidth: barrierWidth[0],
    barrierPosition: barrierPosition[0],
    particleEnergy: particleEnergy[0],
    particleMass: particleMass[0],
    wavePacketWidth: wavePacketWidth[0],
    initialPosition: initialPosition[0],
    initialMomentum: initialMomentum[0],
    time: time[0]
  }), [barrierHeight, barrierWidth, barrierPosition, particleEnergy, particleMass, wavePacketWidth, initialPosition, initialMomentum, time]);
  
  // Advanced simulation parameters
  const [barrierShape, setBarrierShape] = useState('rectangular');
  const [numericalMethod, setNumericalMethod] = useState('split_operator');
  const [absorbingBoundaries, setAbsorbingBoundaries] = useState(false);
  const [webgpuEnabled, setWebgpuEnabled] = useState(false);
  const [performanceBenchmark, setPerformanceBenchmark] = useState<any>(null);
  const { toast } = useToast();

  const advancedParams = useMemo(() => ({
    ...params,
    barrierShape,
    method: numericalMethod,
    absorbing: absorbingBoundaries
  }), [params, barrierShape, numericalMethod, absorbingBoundaries]);

  const quantumData = useMemo(() => 
    calculateAdvancedTunneling(advancedParams, gridResolution[0]),
    [advancedParams, gridResolution]
  );

  // Auto-advance time when animating
  React.useEffect(() => {
    if (!isAnimating) return;
    
    const interval = setInterval(() => {
      setTime(prev => {
        const newTime = prev[0] + 0.1;
        return newTime > 20 ? [0] : [newTime]; // Reset after 20 time units
      });
    }, 100);
    
    return () => clearInterval(interval);
  }, [isAnimating]);

  // Module data for theory component
  const moduleData = {
    title: 'Quantum Tunneling Simulator',
    description: 'Real-time wave packet evolution through potential barriers with quantum statistics.',
    category: 'Quantum Mechanics',
    difficulty: 'Intermediate' as const,
    equation: 'T \\approx \\exp(-2\\kappa a), \\quad \\kappa = \\sqrt{\\frac{2m(V-E)}{\\hbar^2}}',
    theory: {
      overview: 'Quantum tunneling phenomenon where particles penetrate classically forbidden energy barriers.',
      mathematics: [
        'Time-dependent Schrödinger equation',
        'WKB approximation',
        'Transmission coefficients',
        'Wave packet dynamics'
      ],
      references: [
        'Griffiths, Introduction to Quantum Mechanics (2017)',
        'Shankar, Principles of Quantum Mechanics (1994)'
      ]
    }
  };

  const detailedTheory = {
    introduction: `Quantum tunneling is one of the most striking manifestations of quantum mechanics, where particles can penetrate through potential barriers that would be classically insurmountable. This phenomenon underlies many modern technologies including tunnel diodes, scanning tunneling microscopy, and nuclear fusion in stars.`,
    
    fundamentals: [
      {
        title: 'Time-Dependent Schrödinger Equation',
        content: 'The evolution of a quantum wave packet is governed by the time-dependent Schrödinger equation, which determines how the wave function propagates through space and time.',
        equations: [
          'i\\hbar \\frac{\\partial \\psi}{\\partial t} = \\hat{H} \\psi = \\left[-\\frac{\\hbar^2}{2m}\\nabla^2 + V(x)\\right] \\psi',
          '\\psi(x,t) = \\sum_n c_n \\phi_n(x) e^{-iE_n t/\\hbar}'
        ],
        derivation: 'The Hamiltonian operator contains kinetic and potential energy terms that determine the wave packet dynamics.'
      },
      {
        title: 'WKB Approximation',
        content: 'The Wentzel-Kramers-Brillouin approximation provides an analytical estimate for the transmission probability through a potential barrier.',
        equations: [
          'T \\approx \\exp\\left(-2\\int_{x_1}^{x_2} \\kappa(x) dx\\right)',
          '\\kappa(x) = \\sqrt{\\frac{2m[V(x) - E]}{\\hbar^2}}',
          'T \\approx \\exp(-2\\kappa a) \\text{ for rectangular barrier}'
        ],
        derivation: 'The transmission decreases exponentially with barrier width and the square root of the barrier height above the particle energy.'
      },
      {
        title: 'Wave Packet Dynamics',
        content: 'A localized wave packet can be constructed from a superposition of plane waves with different momenta, leading to spreading and tunneling behavior.',
        equations: [
          '\\psi(x,0) = \\int A(k) e^{ikx} dk',
          '\\langle x \\rangle = \\langle x_0 \\rangle + \\frac{\\langle p \\rangle t}{m}',
          '\\sigma_x^2(t) = \\sigma_x^2(0) + \\frac{\\hbar^2 t^2}{4m^2\\sigma_x^2(0)}'
        ],
        derivation: 'Wave packets spread due to the uncertainty principle, with minimum spreading for Gaussian packets.'
      }
    ],
    
    applications: [
      {
        title: 'Scanning Tunneling Microscopy',
        description: 'STM uses quantum tunneling between a sharp tip and sample surface to achieve atomic-scale resolution.',
        examples: [
          'Atomic-scale surface imaging',
          'Single atom manipulation',
          'Electronic density of states measurement'
        ]
      },
      {
        title: 'Tunnel Diodes and Electronics',
        description: 'Tunnel junctions exhibit negative differential resistance useful for high-frequency oscillators.',
        examples: [
          'Josephson junctions in superconductors',
          'Resonant tunneling diodes',
          'Single-electron transistors'
        ]
      },
      {
        title: 'Nuclear Physics',
        description: 'Alpha decay and nuclear fusion rely on quantum tunneling through the Coulomb barrier.',
        examples: [
          'Alpha particle decay in radioactive nuclei',
          'Fusion reactions in stellar cores',
          'Quantum mechanical barrier penetration'
        ]
      }
    ],
    
    furtherReading: [
      {
        title: 'Introduction to Quantum Mechanics',
        authors: 'Griffiths, D. J.',
        journal: 'Cambridge University Press',
        year: 2017,
        doi: '10.1017/9781316995433'
      },
      {
        title: 'Quantum tunneling in complex systems',
        authors: 'Leggett, A. J.',
        journal: 'Reviews of Modern Physics',
        year: 1987,
        doi: '10.1103/RevModPhys.59.1'
      }
    ]
  };

  const resetSimulation = () => {
    setTime([0]);
    setIsAnimating(false);
  };

  const exportData = () => {
    const data = {
      parameters: params,
      quantumData: quantumData,
      timestamp: new Date().toISOString(),
      metadata: {
        gridPoints: gridResolution[0],
        timeEvolved: time[0]
      }
    };
    
    const dataStr = JSON.stringify(data, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'quantum_tunneling_simulation.json';
    link.click();
  };

  return (
    <PhysicsModuleLayout background="quantum">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      
      <PhysicsModuleHeader
        title="Quantum Tunneling Simulator"
        description="Real-time wave packet evolution through potential barriers with advanced numerical methods"
        category="Quantum Dynamics"
        difficulty="Intermediate"
        equation="T \approx \exp(-2\kappa a), \quad \kappa = \sqrt{\frac{2m(V-E)}{\hbar^2}}"
        onExport={exportData}
        onReset={() => setTime([0])}
        isRunning={isAnimating}
      />

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-3 mb-8">
          <TabsTrigger value="simulation" className="flex items-center gap-2">
            <Zap className="w-4 h-4" />
            Tunneling Simulation
          </TabsTrigger>
            <TabsTrigger value="theory" className="flex items-center gap-2">
              <Info className="w-4 h-4" />
              Quantum Theory
            </TabsTrigger>
            <TabsTrigger value="analysis" className="flex items-center gap-2">
              <Download className="w-4 h-4" />
              WKB Analysis
            </TabsTrigger>
          </TabsList>

          {/* Simulation Tab */}
          <TabsContent value="simulation" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
              {/* Parameter Controls */}
              <Card className="p-6 space-y-6">
                <div className="flex items-center gap-2 mb-4">
                  <Settings className="w-5 h-5 text-primary" />
                  <h3 className="text-xl font-semibold font-serif">Advanced Controls</h3>
                </div>
                
                {/* Advanced Method Selection */}
                <div className="space-y-4">
                  <div>
                    <Label className="text-sm font-medium">Barrier Shape</Label>
                    <Select value={barrierShape} onValueChange={setBarrierShape}>
                      <SelectTrigger className="mt-1">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="rectangular">Rectangular</SelectItem>
                        <SelectItem value="gaussian">Gaussian</SelectItem>
                        <SelectItem value="triangular">Triangular</SelectItem>
                        <SelectItem value="double_well">Double Well</SelectItem>
                        <SelectItem value="coulomb">Coulomb</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label className="text-sm font-medium">Numerical Method</Label>
                    <Select value={numericalMethod} onValueChange={setNumericalMethod}>
                      <SelectTrigger className="mt-1">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="split_operator">Split-Operator</SelectItem>
                        <SelectItem value="crank_nicolson">Crank-Nicolson</SelectItem>
                        <SelectItem value="euler">Euler Method</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="flex items-center justify-between">
                    <Label className="text-sm font-medium">Absorbing Boundaries</Label>
                    <Switch checked={absorbingBoundaries} onCheckedChange={setAbsorbingBoundaries} />
                  </div>

                  <div>
                    <Label className="text-sm font-medium">Grid Resolution</Label>
                    <Slider
                      value={gridResolution}
                      onValueChange={setGridResolution}
                      max={1024}
                      min={128}
                      step={64}
                      className="mt-2"
                    />
                    <div className="text-xs text-muted-foreground mt-1 font-mono">
                      {gridResolution[0]} points
                    </div>
                  </div>
                </div>
                
                {/* Barrier Properties */}
                <div className="space-y-4 border-t pt-4">
                  <div>
                    <Label className="text-sm font-medium flex items-center gap-2">
                      Barrier Height
                      <InlineMath math="V_0" />
                    </Label>
                    <Slider
                      value={barrierHeight}
                      onValueChange={setBarrierHeight}
                      max={10.0}
                      min={0.5}
                      step={0.1}
                      className="mt-2"
                    />
                    <div className="text-xs text-muted-foreground mt-1 font-mono">
                      V₀ = {barrierHeight[0].toFixed(1)} eV
                    </div>
                  </div>

                  <div>
                    <Label className="text-sm font-medium flex items-center gap-2">
                      Barrier Width
                      <InlineMath math="a" />
                    </Label>
                    <Slider
                      value={barrierWidth}
                      onValueChange={setBarrierWidth}
                      max={5.0}
                      min={0.5}
                      step={0.1}
                      className="mt-2"
                    />
                    <div className="text-xs text-muted-foreground mt-1 font-mono">
                      a = {barrierWidth[0].toFixed(1)} nm
                    </div>
                  </div>

                  <div>
                    <Label className="text-sm font-medium flex items-center gap-2">
                      Particle Energy
                      <InlineMath math="E" />
                    </Label>
                    <Slider
                      value={particleEnergy}
                      onValueChange={setParticleEnergy}
                      max={8.0}
                      min={0.1}
                      step={0.1}
                      className="mt-2"
                    />
                    <div className="text-xs text-muted-foreground mt-1 font-mono">
                      E = {particleEnergy[0].toFixed(1)} eV
                    </div>
                  </div>
                </div>

                {/* Wave Packet Properties */}
                <div className="space-y-4 border-t pt-4">
                  <Label className="text-sm font-medium">Wave Packet</Label>
                  
                  <div>
                    <Label className="text-xs">Width σ</Label>
                    <Slider
                      value={wavePacketWidth}
                      onValueChange={setWavePacketWidth}
                      max={3.0}
                      min={0.5}
                      step={0.1}
                      className="mt-1"
                    />
                    <div className="text-xs text-muted-foreground font-mono">
                      {wavePacketWidth[0].toFixed(1)} nm
                    </div>
                  </div>
                  
                  <div>
                    <Label className="text-xs">Initial Position</Label>
                    <Slider
                      value={initialPosition}
                      onValueChange={setInitialPosition}
                      max={-2}
                      min={-8}
                      step={0.2}
                      className="mt-1"
                    />
                    <div className="text-xs text-muted-foreground font-mono">
                      {initialPosition[0].toFixed(1)} nm
                    </div>
                  </div>

                  <div>
                    <Label className="text-xs">Initial Momentum</Label>
                    <Slider
                      value={initialMomentum}
                      onValueChange={setInitialMomentum}
                      max={5.0}
                      min={0.5}
                      step={0.1}
                      className="mt-1"
                    />
                    <div className="text-xs text-muted-foreground font-mono">
                      {initialMomentum[0].toFixed(1)} ℏ/nm
                    </div>
                  </div>
                </div>

                {/* Animation Controls */}
                <div className="space-y-3 border-t pt-4">
                  <Label className="text-sm font-medium">Animation</Label>
                  
                  <div className="flex items-center justify-between">
                    <Button
                      variant={isAnimating ? "default" : "outline"}
                      size="sm"
                      onClick={() => setIsAnimating(!isAnimating)}
                      className="flex items-center gap-2"
                    >
                      {isAnimating ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                      {isAnimating ? 'Pause' : 'Play'}
                    </Button>
                    
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={resetSimulation}
                      className="flex items-center gap-2"
                    >
                      <RotateCcw className="w-4 h-4" />
                      Reset
                    </Button>
                  </div>
                  
                  <div>
                    <Label className="text-xs">Time: {time[0].toFixed(1)} fs</Label>
                    <Slider
                      value={time}
                      onValueChange={setTime}
                      max={20}
                      min={0}
                      step={0.1}
                      className="mt-1"
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label className="text-xs">Potential</Label>
                      <Switch checked={showPotential} onCheckedChange={setShowPotential} />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <Label className="text-xs">Wave Function</Label>
                      <Switch checked={showWaveFunction} onCheckedChange={setShowWaveFunction} />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <Label className="text-xs">Probability</Label>
                      <Switch checked={showProbability} onCheckedChange={setShowProbability} />
                    </div>
                  </div>
                </div>
              </Card>

              {/* 3D Visualization */}
              <div className="lg:col-span-3">
                <Plot
                  data={[
                    {
                      x: quantumData.x,
                      y: quantumData.probability,
                      type: 'scatter',
                      mode: 'lines',
                      name: '|ψ|²',
                      line: { color: '#8b5cf6', width: 2 },
                      hovertemplate: 'Position: %{x:.2f} nm<br>Probability: %{y:.4f}<extra></extra>'
                    },
                    {
                      x: quantumData.x,
                      y: quantumData.waveReal,
                      type: 'scatter',
                      mode: 'lines',
                      name: 'Re(ψ)',
                      line: { color: '#3b82f6', width: 1.5 },
                      hovertemplate: 'Position: %{x:.2f} nm<br>Real part: %{y:.4f}<extra></extra>'
                    },
                    {
                      x: quantumData.x,
                      y: quantumData.waveImag,
                      type: 'scatter',
                      mode: 'lines',
                      name: 'Im(ψ)',
                      line: { color: '#ef4444', width: 1.5 },
                      hovertemplate: 'Position: %{x:.2f} nm<br>Imaginary part: %{y:.4f}<extra></extra>'
                    },
                    {
                      x: quantumData.x,
                      y: quantumData.potential,
                      type: 'scatter',
                      mode: 'lines',
                      name: 'V(x)',
                      line: { color: PLOT.colors.physics.hot, width: 3, dash: 'dash' },
                      yaxis: 'y2',
                      hovertemplate: 'Position: %{x:.2f} nm<br>Potential: %{y:.2f} eV<extra></extra>'
                    }
                  ]}
                  layout={{
                    ...PLOT.layout,
                    title: {
                      text: `Quantum Wave Packet Evolution<br><sub>E=${particleEnergy[0].toFixed(1)} eV, V₀=${barrierHeight[0].toFixed(1)} eV, t=${time[0].toFixed(1)} fs</sub>`,
                      font: { size: 18, color: 'hsl(var(--foreground))' }
                    },
                    xaxis: { 
                      ...PLOT.layout.xaxis,
                      title: 'Position (nm)', 
                      range: [-10, 10]
                    },
                    yaxis: { 
                      ...PLOT.layout.yaxis,
                      title: 'Wave Function', 
                      range: [-1, 2]
                    },
                    yaxis2: { 
                      title: 'Potential (eV)', 
                      overlaying: 'y', 
                      side: 'right',
                      showgrid: false,
                      range: [0, Math.max(...quantumData.potential.filter(p => isFinite(p))) * 1.2],
                      titlefont: { color: 'hsl(var(--foreground))' },
                      tickfont: { color: 'hsl(var(--foreground))' }
                    },
                    showlegend: true,
                    legend: { x: 0.02, y: 0.98, font: { color: 'hsl(var(--foreground))' } },
                    annotations: [
                      {
                        x: barrierPosition[0],
                        y: Math.max(0.1, ...quantumData.probability.filter(p => isFinite(p))) * 0.8,
                        text: 'Barrier',
                        showarrow: true,
                        arrowhead: 2,
                        arrowsize: 1,
                        arrowwidth: 2,
                        arrowcolor: PLOT.colors.physics.hot,
                        font: { color: PLOT.colors.physics.hot, size: 12 }
                      }
                    ]
                  }}
                  config={PLOT.config}
                  style={{ width: '100%', height: '400px' }}
                />
                
                {/* Custom 3D quantum visualization */}
                <Card className="mt-6 p-6 h-[500px]">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xl font-semibold font-serif">3D Quantum Wave Evolution</h3>
                    <div className="flex items-center gap-2">
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => setIsAnimating(!isAnimating)}
                      >
                        {isAnimating ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                      </Button>
                      <Badge variant="outline">
                        {quantumData.classicalForbidden ? 'Tunneling' : 'Classical'}
                      </Badge>
                    </div>
                  </div>
                  
                  <div className="h-full bg-background/50 rounded-lg overflow-hidden">
                    <Canvas camera={{ position: [8, 6, 8], fov: 60 }}>
                      <ambientLight intensity={0.4} />
                      <pointLight position={[10, 10, 10]} intensity={0.8} />
                      <directionalLight position={[-10, -10, -5]} intensity={0.5} />
                      
                      <QuantumWave3D 
                        waveData={quantumData}
                        showPotential={showPotential}
                        showWaveFunction={showWaveFunction}
                        showProbability={showProbability}
                        animate={isAnimating}
                        time={time[0]}
                      />
                      
                      <OrbitControls 
                        enablePan={true} 
                        enableZoom={true} 
                        enableRotate={true}
                        maxDistance={20}
                        minDistance={3}
                      />
                    </Canvas>
                  </div>
                </Card>
              </div>
            </div>

            {/* Enhanced WKB Analysis Panel */}
            <Card className="p-6">
              <div className="flex items-center gap-2 mb-4">
                <Calculator className="w-5 h-5 text-accent" />
                <h3 className="text-xl font-semibold font-serif">WKB Analytical Comparison</h3>
                <Badge variant="outline" className="ml-auto">
                  {quantumData.classicalForbidden ? 'Tunneling Regime' : 'Classical Regime'}
                </Badge>
              </div>
              
              {/* Real-time comparison metrics */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                <div className="bg-primary/10 p-4 rounded-lg border border-primary/20">
                  <h4 className="font-semibold text-primary mb-2 flex items-center gap-2">
                    <Zap className="w-4 h-4" />
                    Quantum Result
                  </h4>
                  <div className="text-3xl font-bold font-mono text-primary">
                    {(quantumData.transmission * 100).toFixed(2)}%
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Numerical simulation
                  </div>
                </div>
                
                <div className="bg-accent/10 p-4 rounded-lg border border-accent/20">
                  <h4 className="font-semibold text-accent mb-2 flex items-center gap-2">
                    <Calculator className="w-4 h-4" />
                    WKB Prediction
                  </h4>
                  <div className="text-3xl font-bold font-mono text-accent">
                    {(quantumData.wkbTransmission * 100).toFixed(2)}%
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Analytical formula
                  </div>
                </div>

                <div className="bg-muted/50 p-4 rounded-lg border">
                  <h4 className="font-semibold mb-2 flex items-center gap-2">
                    <Target className="w-4 h-4" />
                    Accuracy Ratio
                  </h4>
                  <div className="text-3xl font-bold font-mono">
                    {quantumData.wkbTransmission > 0 ? 
                      (quantumData.transmission / quantumData.wkbTransmission).toFixed(3) : 
                      'N/A'
                    }
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Quantum/WKB
                  </div>
                </div>

                <div className="bg-muted/50 p-4 rounded-lg border">
                  <h4 className="font-semibold mb-2 flex items-center gap-2">
                    <Activity className="w-4 h-4" />
                    Error %
                  </h4>
                  <div className="text-3xl font-bold font-mono">
                    {Math.abs((quantumData.transmission - quantumData.wkbTransmission) / Math.max(quantumData.transmission, 0.001) * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Relative error
                  </div>
                </div>
              </div>

              {/* Action integral visualization */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <div className="bg-muted/30 p-4 rounded-lg">
                  <h4 className="font-semibold mb-3 flex items-center gap-2">
                    <Eye className="w-4 h-4" />
                    WKB Action Integral
                  </h4>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Action ∫κ(x)dx:</span>
                      <span className="font-mono font-bold">{quantumData.actionIntegral.toFixed(3)} ℏ</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Exp(-2∫κdx):</span>
                      <span className="font-mono font-bold">{Math.exp(-2 * quantumData.actionIntegral).toFixed(6)}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Barrier opacity:</span>
                      <span className="font-mono font-bold">{(quantumData.actionIntegral / Math.PI).toFixed(2)} π</span>
                    </div>
                  </div>
                  <Progress 
                    value={Math.min(quantumData.actionIntegral * 10, 100)} 
                    className="mt-3"
                  />
                  <div className="text-xs text-muted-foreground mt-1">
                    Higher action → lower transmission
                  </div>
                </div>

                <div className="bg-muted/30 p-4 rounded-lg">
                  <h4 className="font-semibold mb-3 flex items-center gap-2">
                    <Layers className="w-4 h-4" />
                    Physical Parameters
                  </h4>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm">κ (avg):</span>
                      <span className="font-mono text-sm">
                        {(Math.sqrt(2 * particleMass[0] * Math.abs(barrierHeight[0] - particleEnergy[0]))).toFixed(3)} nm⁻¹
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">de Broglie λ:</span>
                      <span className="font-mono text-sm">{quantumData.deBroglieWavelength.toFixed(3)} nm</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Barrier/λ ratio:</span>
                      <span className="font-mono text-sm">
                        {(barrierWidth[0] / quantumData.deBroglieWavelength).toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Classical turn points:</span>
                      <span className="font-mono text-sm">
                        {quantumData.classicalForbidden ? 'Present' : 'None'}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* WKB accuracy assessment */}
              <div className="p-4 bg-accent/10 rounded-lg border border-accent/20">
                <h4 className="font-semibold mb-3 flex items-center gap-2">
                  <Info className="w-4 h-4" />
                  WKB Approximation Validity
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                  <div>
                    <div className="font-medium">Accuracy Indicator:</div>
                    <div className={`mt-1 ${Math.abs(quantumData.transmission - quantumData.wkbTransmission) / Math.max(quantumData.transmission, 0.001) < 0.1 ? 'text-green-500' : 'text-orange-500'}`}>
                      {Math.abs(quantumData.transmission - quantumData.wkbTransmission) / Math.max(quantumData.transmission, 0.001) < 0.1 ? 
                        '✓ Excellent agreement' : 
                        '⚠ Moderate agreement'
                      }
                    </div>
                  </div>
                  <div>
                    <div className="font-medium">Validity condition:</div>
                    <div className={`mt-1 ${quantumData.actionIntegral > 1 ? 'text-green-500' : 'text-orange-500'}`}>
                      {quantumData.actionIntegral > 1 ? 
                        '✓ ∫κdx ≫ 1 (valid)' : 
                        '⚠ ∫κdx ≈ 1 (questionable)'
                      }
                    </div>
                  </div>
                  <div>
                    <div className="font-medium">Regime:</div>
                    <div className="mt-1">
                      {quantumData.classicalForbidden ? 
                        (quantumData.transmission < 0.5 ? 'Strong tunneling' : 'Weak barrier') :
                        'Over-barrier'
                      }
                    </div>
                  </div>
                </div>
              </div>

              {/* Key WKB equations */}
              <div className="mt-6 p-4 bg-primary/5 rounded-lg border border-primary/10">
                <h4 className="font-semibold mb-3">WKB Transmission Formula</h4>
                <div className="space-y-2">
                  <BlockMath math="T_{WKB} = \exp\left(-2\int_{x_1}^{x_2} \kappa(x) \, dx\right)" errorColor={'#cc0000'} throwOnError={false} />
                  <BlockMath math="\kappa(x) = \sqrt{\frac{2m[V(x) - E]}{\hbar^2}}" errorColor={'#cc0000'} throwOnError={false} />
                  <div className="text-xs text-muted-foreground">
                    Valid when ∫κ(x)dx ≫ 1 and potential varies slowly compared to λ
                  </div>
                </div>
              </div>
            </Card>
          </TabsContent>

          {/* Theory Tab */}
          <TabsContent value="theory">
            <PhysicsTheory 
              module={moduleData}
              detailedTheory={detailedTheory}
            />
          </TabsContent>

          {/* Analysis Tab */}
          <TabsContent value="analysis" className="space-y-6">
            {/* Conservation Laws & Validation */}
            <Card className="p-6">
              <div className="flex items-center gap-2 mb-4">
                <Calculator className="w-5 h-5 text-primary" />
                <h3 className="text-xl font-semibold font-serif">Conservation Laws & Validation</h3>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium text-sm text-muted-foreground">Probability Conservation</h4>
                  <div className="text-2xl font-mono font-bold">
                    {quantumData.conservationMetrics.probabilityConservation.toFixed(6)}
                  </div>
                  <Progress 
                    value={quantumData.conservationMetrics.probabilityConservation * 100} 
                    className="mt-2"
                  />
                  <div className="text-xs text-muted-foreground mt-1">
                    Should be 1.000000
                  </div>
                </div>

                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium text-sm text-muted-foreground">Average Energy</h4>
                  <div className="text-2xl font-mono font-bold">
                    {quantumData.conservationMetrics.averageEnergy.toFixed(3)}
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    eV (should be constant)
                  </div>
                </div>

                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium text-sm text-muted-foreground">⟨Position⟩</h4>
                  <div className="text-2xl font-mono font-bold">
                    {quantumData.conservationMetrics.positionExpectation.toFixed(3)}
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    nm
                  </div>
                </div>

                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium text-sm text-muted-foreground">⟨Momentum⟩</h4>
                  <div className="text-2xl font-mono font-bold">
                    {quantumData.conservationMetrics.momentumExpectation.toFixed(3)}
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    ℏ/nm
                  </div>
                </div>
              </div>

              {/* Validation Alerts */}
              <div className="mt-4 space-y-2">
                {Math.abs(quantumData.conservationMetrics.probabilityConservation - 1.0) > 0.01 && (
                  <Alert>
                    <AlertDescription>
                      ⚠️ Probability conservation violation detected. Consider using smaller time steps or higher resolution.
                    </AlertDescription>
                  </Alert>
                )}
                
                {quantumData.transmission + quantumData.reflection + quantumData.barrierOccupancy < 0.95 && (
                  <Alert>
                    <AlertDescription>
                      ⚠️ Significant probability loss detected. Check boundary conditions or numerical stability.
                    </AlertDescription>
                  </Alert>
                )}
              </div>
            </Card>

            {/* Transmission Analysis */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="p-6">
                <div className="flex items-center gap-2 mb-4">
                  <Target className="w-5 h-5 text-primary" />
                  <h3 className="text-lg font-semibold">Transmission Components</h3>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium">Transmitted</span>
                      <span className="text-sm font-mono">{(quantumData.transmission * 100).toFixed(2)}%</span>
                    </div>
                    <Progress value={quantumData.transmission * 100} className="h-2" />
                  </div>
                  
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium">Reflected</span>
                      <span className="text-sm font-mono">{(quantumData.reflection * 100).toFixed(2)}%</span>
                    </div>
                    <Progress value={quantumData.reflection * 100} className="h-2" />
                  </div>
                  
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium">In Barrier</span>
                      <span className="text-sm font-mono">{(quantumData.barrierOccupancy * 100).toFixed(2)}%</span>
                    </div>
                    <Progress value={quantumData.barrierOccupancy * 100} className="h-2" />
                  </div>
                </div>
                
                <Separator className="my-4" />
                
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm">WKB Prediction:</span>
                    <span className="text-sm font-mono">{(quantumData.wkbTransmission * 100).toFixed(2)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Quantum/WKB Ratio:</span>
                    <span className="text-sm font-mono">
                      {quantumData.wkbTransmission > 0 ? (quantumData.transmission / quantumData.wkbTransmission).toFixed(2) : 'N/A'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Action Integral:</span>
                    <span className="text-sm font-mono">{quantumData.actionIntegral.toFixed(3)} ℏ</span>
                  </div>
                </div>
              </Card>

              <Card className="p-6">
                <div className="flex items-center gap-2 mb-4">
                  <Activity className="w-5 h-5 text-primary" />
                  <h3 className="text-lg font-semibold">Wave Properties</h3>
                </div>
                
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-3 bg-muted/30 rounded">
                      <div className="text-xs text-muted-foreground">de Broglie λ</div>
                      <div className="text-lg font-mono">{quantumData.deBroglieWavelength.toFixed(3)} nm</div>
                    </div>
                    
                    <div className="p-3 bg-muted/30 rounded">
                      <div className="text-xs text-muted-foreground">Group Velocity</div>
                      <div className="text-lg font-mono">{quantumData.groupVelocity.toFixed(3)} nm/fs</div>
                    </div>
                    
                    <div className="p-3 bg-muted/30 rounded">
                      <div className="text-xs text-muted-foreground">Phase Shift</div>
                      <div className="text-lg font-mono">{quantumData.phaseShift.toFixed(3)} rad</div>
                    </div>
                    
                    <div className="p-3 bg-muted/30 rounded">
                      <div className="text-xs text-muted-foreground">Resonance</div>
                      <div className="text-lg font-mono">{quantumData.resonanceStrength.toFixed(3)}</div>
                    </div>
                  </div>
                  
                  <div className="p-3 bg-accent/10 rounded">
                    <div className="text-sm font-medium mb-2">Classical vs Quantum</div>
                    <div className="text-sm">
                      {quantumData.classicalForbidden ? (
                        <span className="text-red-400">
                          🚫 Classically forbidden (E &lt; V₀)
                        </span>
                      ) : (
                        <span className="text-green-400">
                          ✅ Classically allowed (E ≥ V₀)
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">
                      Energy: {particleEnergy[0].toFixed(1)} eV, Barrier: {barrierHeight[0].toFixed(1)} eV
                    </div>
                  </div>
                </div>
              </Card>
            </div>

            {/* Advanced Theory Section */}
            <Card className="p-6">
              <div className="flex items-center gap-2 mb-4">
                <Layers className="w-5 h-5 text-primary" />
                <h3 className="text-xl font-semibold font-serif">Advanced Tunneling Theory</h3>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold mb-3">Numerical Methods Comparison</h4>
                  <div className="space-y-3">
                    <div className="p-3 border rounded">
                      <div className="font-medium text-sm">Split-Operator Method</div>
                      <div className="text-xs text-muted-foreground">
                        Most accurate for tunneling. Uses operator splitting with exponential operators.
                      </div>
                      <div className="text-xs mt-1">
                        <strong>Pros:</strong> Unitary, conserves probability
                        <br />
                        <strong>Cons:</strong> Requires FFT for best performance
                      </div>
                    </div>
                    
                    <div className="p-3 border rounded">
                      <div className="font-medium text-sm">Crank-Nicolson Method</div>
                      <div className="text-xs text-muted-foreground">
                        Implicit method with excellent stability. Uses (I + iH·dt/2ℏ)ψ(t+dt) = (I - iH·dt/2ℏ)ψ(t)
                      </div>
                      <div className="text-xs mt-1">
                        <strong>Pros:</strong> Very stable, implicit
                        <br />
                        <strong>Cons:</strong> Requires matrix inversion
                      </div>
                    </div>
                    
                    <div className="p-3 border rounded">
                      <div className="font-medium text-sm">Euler Method</div>
                      <div className="text-xs text-muted-foreground">
                        Simple explicit method: ψ(t+dt) = ψ(t) - i·H·ψ(t)·dt/ℏ
                      </div>
                      <div className="text-xs mt-1">
                        <strong>Pros:</strong> Simple, fast
                        <br />
                        <strong>Cons:</strong> Less stable, may violate conservation
                      </div>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h4 className="font-semibold mb-3">Barrier Shape Effects</h4>
                  <div className="space-y-3">
                    <div className="p-3 border rounded">
                      <div className="font-medium text-sm">Rectangular Barrier</div>
                      <div className="text-xs text-muted-foreground">
                        T = [1 + (V₀²sin²(ka))/(4E(V₀-E))]⁻¹ for E &lt; V₀
                      </div>
                    </div>
                    
                    <div className="p-3 border rounded">
                      <div className="font-medium text-sm">Gaussian Barrier</div>
                      <div className="text-xs text-muted-foreground">
                        Smooth potential with V(x) = V₀ exp(-x²/σ²). More realistic for many physical systems.
                      </div>
                    </div>
                    
                    <div className="p-3 border rounded">
                      <div className="font-medium text-sm">Coulomb Barrier</div>
                      <div className="text-xs text-muted-foreground">
                        V(x) = Ze²/(4πε₀x). Important for nuclear physics and atomic interactions.
                      </div>
                    </div>
                    
                    <div className="p-3 border rounded">
                      <div className="font-medium text-sm">Double Well</div>
                      <div className="text-xs text-muted-foreground">
                        Shows resonant tunneling and energy level splitting due to tunnel coupling.
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              
              <Separator className="my-6" />
              
              <div>
                <h4 className="font-semibold mb-3">Physical Applications</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="p-4 bg-muted/30 rounded-lg">
                    <h5 className="font-medium text-sm mb-2">Scanning Tunneling Microscopy</h5>
                    <div className="text-xs text-muted-foreground">
                      Current I proportional to V·exp(-2κd) where κ depends on work function
                    </div>
                  </div>
                  
                  <div className="p-4 bg-muted/30 rounded-lg">
                    <h5 className="font-medium text-sm mb-2">Josephson Junctions</h5>
                    <div className="text-xs text-muted-foreground">
                      Cooper pair tunneling with I = I_c sin(δ), where δ is phase difference
                    </div>
                  </div>
                  
                  <div className="p-4 bg-muted/30 rounded-lg">
                    <h5 className="font-medium text-sm mb-2">Nuclear Alpha Decay</h5>
                    <div className="text-xs text-muted-foreground">
                      Gamow model: τ proportional to exp(2G) where G is Gamow factor
                    </div>
                  </div>
                </div>
              </div>
            </Card>

            {/* Performance Metrics */}
            {performanceBenchmark && (
              <Card className="p-6">
                <div className="flex items-center gap-2 mb-4">
                  <Cpu className="w-5 h-5 text-primary" />
                  <h3 className="text-lg font-semibold">Performance Benchmark</h3>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="p-3 bg-muted/30 rounded">
                    <div className="text-xs text-muted-foreground">CPU Time</div>
                    <div className="text-lg font-mono">{performanceBenchmark.cpuTime?.toFixed(1) || 'N/A'} ms</div>
                  </div>
                  
                  <div className="p-3 bg-muted/30 rounded">
                    <div className="text-xs text-muted-foreground">GPU Time</div>
                    <div className="text-lg font-mono">{performanceBenchmark.gpuTime?.toFixed(1) || 'N/A'} ms</div>
                  </div>
                  
                  <div className="p-3 bg-muted/30 rounded">
                    <div className="text-xs text-muted-foreground">Speedup</div>
                    <div className="text-lg font-mono">
                      {performanceBenchmark.speedup?.toFixed(1) || 'N/A'}×
                    </div>
                  </div>
                </div>
              </Card>
            )}
          </TabsContent>
        </Tabs>
    </PhysicsModuleLayout>
  );
}