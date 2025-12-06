import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { StatusChip } from '@/components/ui/status-chip';
import { useAccessibilityContext } from '@/components/AccessibilityProvider';
import { Pause, Play, RotateCcw, Eye, Activity, Orbit, GraduationCap, Settings, BookOpen } from 'lucide-react';
import { trackQuantumEvents } from '@/lib/analytics';

type LearningMode = 'tutorial' | 'exploration' | 'advanced';

export const BlochSphere = () => {
  const [animationPhase, setAnimationPhase] = useState(0);
  const [isPaused, setIsPaused] = useState(false);
  const [reduceMotion, setReduceMotion] = useState(false);
  const [interactiveMode, setInteractiveMode] = useState(false);
  const [customTheta, setCustomTheta] = useState(0);
  const [customPhi, setCustomPhi] = useState(0);
  const [learningMode, setLearningMode] = useState<LearningMode>('tutorial');
  const [showExplanations, setShowExplanations] = useState(true);
  const { announce } = useAccessibilityContext();

  // Check for reduced motion preference
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReduceMotion(mediaQuery.matches);
    
    const handleChange = () => setReduceMotion(mediaQuery.matches);
    mediaQuery.addEventListener('change', handleChange);
    
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  // Animation loop
  useEffect(() => {
    if (isPaused || reduceMotion) return;

    const interval = setInterval(() => {
      setAnimationPhase((prev) => (prev + 0.02) % (2 * Math.PI));
    }, 32); // ~30fps

    return () => clearInterval(interval);
  }, [isPaused, reduceMotion]);

  // Calculate state vector position on Bloch sphere
  const theta = interactiveMode ? customTheta : animationPhase;
  const phi = interactiveMode ? customPhi : animationPhase * 1.5;
  
  const x = Math.sin(theta) * Math.cos(phi);
  const y = Math.sin(theta) * Math.sin(phi);
  const z = Math.cos(theta);

  // Convert to 2D coordinates for SVG
  const x2d = x * 40;
  const y2d = -z * 40;
  const depth = y; // For visual depth indication

  // Calculate probabilities
  const prob0 = (1 + z) / 2;
  const prob1 = (1 - z) / 2;
  const phase = phi % (2 * Math.PI);

  const handlePause = () => {
    setIsPaused(!isPaused);
    announce(isPaused ? 'Bloch sphere animation resumed' : 'Bloch sphere animation paused');
    trackQuantumEvents.blochInteraction(isPaused ? 'play' : 'pause');
  };

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === ' ' || event.key === 'Enter') {
      event.preventDefault();
      handlePause();
    } else if (event.key === 'r' || event.key === 'R') {
      event.preventDefault();
      handleReset();
    }
  };

  const handleReset = () => {
    setAnimationPhase(0);
    setCustomTheta(0);
    setCustomPhi(0);
    setInteractiveMode(false);
    announce('Bloch sphere reset to initial state');
    trackQuantumEvents.blochInteraction('reset');
  };

  const handleSphereClick = (event: React.MouseEvent<SVGSVGElement>) => {
    const svg = event.currentTarget;
    const rect = svg.getBoundingClientRect();
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    
    // Get click coordinates relative to center
    const clickX = (event.clientX - rect.left - centerX) / (rect.width / 4);
    const clickY = -(event.clientY - rect.top - centerY) / (rect.height / 4);
    
    // Convert to spherical coordinates
    const r = Math.min(1, Math.sqrt(clickX * clickX + clickY * clickY));
    const newTheta = Math.acos(Math.max(-1, Math.min(1, clickY / Math.max(0.1, r))));
    const newPhi = Math.atan2(clickX, Math.max(0.001, Math.sqrt(Math.max(0, 1 - clickY * clickY))));
    
    setCustomTheta(newTheta);
    setCustomPhi(newPhi);
    setInteractiveMode(true);
    setIsPaused(true);
    
    announce(`Set quantum state: theta ${Math.round(newTheta * 180 / Math.PI)} degrees, phi ${Math.round(newPhi * 180 / Math.PI)} degrees`);
    trackQuantumEvents.blochInteraction('reset');
  };

  const toggleInteractiveMode = () => {
    setInteractiveMode(!interactiveMode);
    if (!interactiveMode) {
      setIsPaused(true);
    }
    announce(interactiveMode ? 'Switched to automatic animation' : 'Switched to interactive mode');
  };

  // Quantum state presets for different learning modes
  const quantumPresets = {
    tutorial: [
      { name: '|0⟩', theta: 0, phi: 0, description: 'Ground state - qubit is definitely in |0⟩' },
      { name: '|1⟩', theta: Math.PI, phi: 0, description: 'Excited state - qubit is definitely in |1⟩' },
      { name: '|+⟩', theta: Math.PI/2, phi: 0, description: 'Superposition - equal chance of |0⟩ and |1⟩' },
    ],
    exploration: [
      { name: '|0⟩', theta: 0, phi: 0, description: 'Ground state' },
      { name: '|1⟩', theta: Math.PI, phi: 0, description: 'Excited state' },
      { name: '|+⟩', theta: Math.PI/2, phi: 0, description: 'Positive superposition' },
      { name: '|-⟩', theta: Math.PI/2, phi: Math.PI, description: 'Negative superposition' },
      { name: '|+i⟩', theta: Math.PI/2, phi: Math.PI/2, description: 'Positive imaginary state' },
    ],
    advanced: [
      { name: '|0⟩', theta: 0, phi: 0, description: 'Computational basis |0⟩' },
      { name: '|1⟩', theta: Math.PI, phi: 0, description: 'Computational basis |1⟩' },
      { name: '|+⟩', theta: Math.PI/2, phi: 0, description: 'Hadamard basis |+⟩ = (|0⟩+|1⟩)/√2' },
      { name: '|-⟩', theta: Math.PI/2, phi: Math.PI, description: 'Hadamard basis |-⟩ = (|0⟩-|1⟩)/√2' },
      { name: '|+i⟩', theta: Math.PI/2, phi: Math.PI/2, description: 'Circular basis |+i⟩ = (|0⟩+i|1⟩)/√2' },
      { name: '|-i⟩', theta: Math.PI/2, phi: 3*Math.PI/2, description: 'Circular basis |-i⟩ = (|0⟩-i|1⟩)/√2' },
    ]
  };

  const setPresetState = (preset: any) => {
    setCustomTheta(preset.theta);
    setCustomPhi(preset.phi);
    setInteractiveMode(true);
    setIsPaused(true);
    announce(`Set state to ${preset.name}: ${preset.description}`);
    trackQuantumEvents.blochInteraction('reset');
  };

  return (
    <Card className="relative rounded-2xl border border-purple-400/30 bg-slate-900/80 shadow-xl hover:border-purple-400/50 transition-all duration-300">
      {/* Accent bar */}
      <div className="absolute left-0 top-0 bottom-0 w-1 bg-purple-400 rounded-r-full"></div>
      
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-purple-500/20 border border-purple-400/30 rounded-lg flex items-center justify-center">
              <Eye className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <div className="flex items-center gap-2 mb-1">
                <CardTitle className="text-lg font-semibold text-slate-200">Bloch Sphere</CardTitle>
                <Badge className={`text-xs ${
                  learningMode === 'tutorial' ? 'bg-green-500/20 text-green-300 border-green-400/30' :
                  learningMode === 'exploration' ? 'bg-yellow-500/20 text-yellow-300 border-yellow-400/30' :
                  'bg-red-500/20 text-red-300 border-red-400/30'
                }`}>
                  <GraduationCap className="w-3 h-3 mr-1" />
                  {learningMode.charAt(0).toUpperCase() + learningMode.slice(1)}
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground">Quantum state visualization with {quantumPresets[learningMode].length} presets</p>
            </div>
          </div>
          <div className="flex gap-2">
            <StatusChip 
              variant={isPaused ? "idle" : (reduceMotion ? "info" : "running")}
              icon={isPaused ? <Pause className="w-3 h-3" /> : <Orbit className="w-3 h-3" />}
            >
              {isPaused ? "Paused" : (reduceMotion ? "Static" : "Live")}
            </StatusChip>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Learning Mode Selector */}
        <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-600/30">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-medium text-slate-300 flex items-center gap-2">
              <Settings className="w-4 h-4" />
              Learning Mode
            </h4>
            <Button
              onClick={() => setShowExplanations(!showExplanations)}
              variant="ghost"
              size="sm"
              className="text-xs"
            >
              <BookOpen className="w-3 h-3 mr-1" />
              {showExplanations ? 'Hide' : 'Show'} Help
            </Button>
          </div>
          <div className="flex gap-2">
            {(['tutorial', 'exploration', 'advanced'] as LearningMode[]).map((mode) => (
              <Button
                key={mode}
                onClick={() => {
                  setLearningMode(mode);
                  handleReset();
                  announce(`Switched to ${mode} mode`);
                }}
                variant={learningMode === mode ? "primary" : "outline"}
                size="sm"
                className={`flex-1 text-xs ${
                  mode === 'tutorial' ? 'hover:bg-green-500/20 hover:border-green-400/40' :
                  mode === 'exploration' ? 'hover:bg-yellow-500/20 hover:border-yellow-400/40' :
                  'hover:bg-red-500/20 hover:border-red-400/40'
                }`}
              >
                {mode === 'tutorial' && '🧭'}
                {mode === 'exploration' && '🔍'}
                {mode === 'advanced' && '⚡'}
                <span className="ml-1 capitalize">{mode}</span>
              </Button>
            ))}
          </div>
          {showExplanations && (
            <div className="mt-3 p-3 rounded-lg bg-slate-900/50 border border-slate-700/30">
              <p className="text-xs text-muted-foreground">
                {learningMode === 'tutorial' && '🧭 Learn basic quantum states |0⟩, |1⟩, and superposition |+⟩ with detailed explanations'}
                {learningMode === 'exploration' && '🔍 Explore more quantum states including negative superposition and imaginary phases'}
                {learningMode === 'advanced' && '⚡ Master all Bloch sphere states with mathematical notation and basis definitions'}
              </p>
            </div>
          )}
        </div>

        {/* Quantum State Presets */}
        <div>
          <h4 className="text-sm font-medium text-slate-300 mb-3">Quantum State Presets</h4>
          <div className={`grid gap-2 ${
            learningMode === 'tutorial' ? 'grid-cols-1' :
            learningMode === 'exploration' ? 'grid-cols-2' :
            'grid-cols-3'
          }`}>
            {quantumPresets[learningMode].map((preset) => (
              <Button
                key={preset.name}
                onClick={() => setPresetState(preset)}
                variant="outline"
                size="sm"
                className={`h-auto p-3 flex-col gap-1 text-left ${
                  interactiveMode && 
                  Math.abs(customTheta - preset.theta) < 0.1 && 
                  Math.abs(customPhi - preset.phi) < 0.1
                    ? 'border-purple-400/50 bg-purple-500/10' 
                    : 'hover:border-purple-400/40'
                }`}
              >
                <div className="font-mono font-semibold text-sm">{preset.name}</div>
                {(learningMode === 'tutorial' || showExplanations) && (
                  <div className="text-xs text-muted-foreground leading-relaxed">
                    {preset.description}
                  </div>
                )}
              </Button>
            ))}
          </div>
        </div>

        {/* Controls */}
        <div className="space-y-3">
          <div className="flex gap-2">
            <Button
              onClick={handlePause}
              onKeyDown={handleKeyDown}
              aria-label={isPaused ? 'Resume Bloch sphere animation' : 'Pause Bloch sphere animation'}
              variant={isPaused ? "primary" : "outline"}
              size="sm"
              className="flex-1"
              disabled={interactiveMode}
            >
              {isPaused ? (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  Resume
                </>
              ) : (
                <>
                  <Pause className="w-4 h-4 mr-2" />
                  Pause
                </>
              )}
            </Button>
            
            <Button
              onClick={toggleInteractiveMode}
              variant={interactiveMode ? "primary" : "outline"}
              size="sm"
              className="flex-1"
            >
              <Activity className="w-4 h-4 mr-2" />
              {interactiveMode ? 'Auto' : 'Interactive'}
            </Button>
            
            <Button
              onClick={handleReset}
              onKeyDown={handleKeyDown}
              aria-label="Reset Bloch sphere to initial quantum state"
              variant="outline"
              size="sm"
            >
              <RotateCcw className="w-4 h-4 mr-2" />
              Reset
            </Button>
          </div>
          
          {interactiveMode && (
            <div className="text-xs text-muted-foreground text-center bg-purple-500/10 p-2 rounded-lg border border-purple-400/30">
              💡 {learningMode === 'tutorial' ? 'Use presets above or click the sphere to explore' : 'Click anywhere on the sphere to set a custom quantum state'}
            </div>
          )}
        </div>

        {/* Bloch Sphere Visualization */}
        <div className="bg-slate-800/30 rounded-lg p-6 border border-slate-600/30">
          {/* Screen reader description */}
          <div className="sr-only" id="bloch-description">
            Quantum state on Bloch sphere: |0⟩ probability {Math.round(prob0 * 100)}%, |1⟩ probability {Math.round(prob1 * 100)}%, phase {Math.round(phase * 180 / Math.PI)} degrees
          </div>
          <svg 
            viewBox="-60 -60 120 120" 
            className={`w-full h-48 drop-shadow-sm transition-all duration-200 ${
              interactiveMode ? 'cursor-pointer hover:drop-shadow-lg' : ''
            }`}
            onClick={interactiveMode ? handleSphereClick : undefined}
            role={interactiveMode ? 'button' : 'img'}
            tabIndex={interactiveMode ? 0 : -1}
            onKeyDown={interactiveMode ? (e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                // Center click for keyboard users
                handleSphereClick({
                  currentTarget: e.currentTarget,
                  clientX: e.currentTarget.getBoundingClientRect().left + e.currentTarget.getBoundingClientRect().width / 2,
                  clientY: e.currentTarget.getBoundingClientRect().top + e.currentTarget.getBoundingClientRect().height / 2
                } as any);
              }
            } : undefined}
            aria-describedby="bloch-description"
            aria-label={interactiveMode ? 'Interactive Bloch sphere - click to set quantum state' : 'Animated Bloch sphere visualization'}
          >
            {/* Sphere outline */}
            <circle 
              cx="0" 
              cy="0" 
              r="50" 
              fill="none" 
              stroke="hsl(var(--border))" 
              strokeWidth="1" 
              opacity="0.6"
            />
            
            {/* Equator */}
            <ellipse 
              cx="0" 
              cy="0" 
              rx="50" 
              ry="12" 
              fill="none" 
              stroke="hsl(var(--border))" 
              strokeWidth="0.5" 
              opacity="0.4"
            />
            
            {/* Meridian */}
            <ellipse 
              cx="0" 
              cy="0" 
              rx="12" 
              ry="50" 
              fill="none" 
              stroke="hsl(var(--border))" 
              strokeWidth="0.5" 
              opacity="0.4"
            />
            
            {/* Axes */}
            <line 
              x1="0" 
              y1="-55" 
              x2="0" 
              y2="55" 
              stroke="hsl(var(--muted))" 
              strokeWidth="0.5" 
              opacity="0.3"
            />
            <line 
              x1="-55" 
              y1="0" 
              x2="55" 
              y2="0" 
              stroke="hsl(var(--muted))" 
              strokeWidth="0.5" 
              opacity="0.3"
            />
            
            {/* State vector */}
            <line 
              x1="0" 
              y1="0" 
              x2={x2d} 
              y2={y2d} 
              stroke="hsl(var(--accent))" 
              strokeWidth="3" 
              className="drop-shadow-sm"
            />
            
            {/* State point */}
            <circle 
              cx={x2d} 
              cy={y2d} 
              r="4" 
              fill="hsl(var(--accent))" 
              className="drop-shadow-md"
            />
            
            {/* Glow effect */}
            <circle 
              cx={x2d} 
              cy={y2d} 
              r="8" 
              fill="hsl(var(--accent))" 
              opacity="0.3"
              className={!reduceMotion ? "animate-pulse" : ""}
            />
            
            {/* Projection traces */}
            <circle 
              cx={x2d} 
              cy="0" 
              r="1" 
              fill="hsl(var(--accent))" 
              opacity="0.5"
            />
            <circle 
              cx="0" 
              cy={y2d} 
              r="1" 
              fill="hsl(var(--accent))" 
              opacity="0.5"
            />
            
            {/* Axis labels */}
            <text 
              x="0" 
              y="-58" 
              textAnchor="middle" 
              className="text-xs fill-muted font-mono" 
              fontSize="8"
            >
              |0⟩
            </text>
            <text 
              x="0" 
              y="66" 
              textAnchor="middle" 
              className="text-xs fill-muted font-mono" 
              fontSize="8"
            >
              |1⟩
            </text>
            <text 
              x="58" 
              y="4" 
              textAnchor="middle" 
              className="text-xs fill-muted font-mono" 
              fontSize="8"
            >
              +
            </text>
            <text 
              x="-58" 
              y="4" 
              textAnchor="middle" 
              className="text-xs fill-muted font-mono" 
              fontSize="8"
            >
              -
            </text>
          </svg>
        </div>

        {/* Quantum State Presets */}
        {interactiveMode && (
          <div>
            <h4 className="text-small font-medium text-text mb-3">Common Quantum States</h4>
            <div className="grid grid-cols-2 gap-2">
              <Button
                variant="outline"
                size="sm"
                className="h-auto p-2 flex-col gap-1"
                onClick={() => {
                  setCustomTheta(0);
                  setCustomPhi(0);
                  announce('Set to |0⟩ state');
                }}
              >
                <span className="font-mono text-xs">|0⟩</span>
                <span className="text-xs text-muted">Ground</span>
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="h-auto p-2 flex-col gap-1"
                onClick={() => {
                  setCustomTheta(Math.PI);
                  setCustomPhi(0);
                  announce('Set to |1⟩ state');
                }}
              >
                <span className="font-mono text-xs">|1⟩</span>
                <span className="text-xs text-muted">Excited</span>
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="h-auto p-2 flex-col gap-1"
                onClick={() => {
                  setCustomTheta(Math.PI/2);
                  setCustomPhi(0);
                  announce('Set to |+⟩ superposition state');
                }}
              >
                <span className="font-mono text-xs">|+⟩</span>
                <span className="text-xs text-muted">Plus</span>
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="h-auto p-2 flex-col gap-1"
                onClick={() => {
                  setCustomTheta(Math.PI/2);
                  setCustomPhi(Math.PI);
                  announce('Set to |-⟩ superposition state');
                }}
              >
                <span className="font-mono text-xs">|-⟩</span>
                <span className="text-xs text-muted">Minus</span>
              </Button>
            </div>
          </div>
        )}

        {/* State Information */}
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-surface-2/30 rounded-lg p-3 border border-border">
              <div className="text-small text-muted mb-1">Probability |0⟩</div>
              <div className="text-h3 font-mono font-semibold text-text">
                {prob0.toFixed(3)}
              </div>
            </div>
            <div className="bg-surface-2/30 rounded-lg p-3 border border-border">
              <div className="text-small text-muted mb-1">Probability |1⟩</div>
              <div className="text-h3 font-mono font-semibold text-text">
                {prob1.toFixed(3)}
              </div>
            </div>
          </div>
          
          <div className="bg-surface-2/30 rounded-lg p-3 border border-border">
            <div className="text-small text-muted mb-1">Phase (radians)</div>
            <div className="text-h3 font-mono font-semibold text-text">
              {phase.toFixed(3)}
            </div>
          </div>
          
          <div className="bg-surface-2/30 rounded-lg p-3 border border-border text-center">
            <div className="text-small text-muted mb-1">State Vector</div>
            <div className="text-body font-mono text-text">
              ψ = α|0⟩ + β|1⟩
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};