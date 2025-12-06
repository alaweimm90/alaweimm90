import { memo, useEffect, useState } from 'react';
import { Cpu, Zap, Brain, Shield, Target, Activity } from 'lucide-react';

interface LoadingPhase {
  id: string;
  name: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  duration: number;
  color: string;
  progress: number;
}

interface PowerUpLoaderProps {
  show: boolean;
  onComplete?: () => void;
  customPhases?: LoadingPhase[];
}

const defaultPhases: LoadingPhase[] = [
  {
    id: 'initialization',
    name: 'System Initialization',
    description: 'Booting strategic command protocols...',
    icon: Cpu,
    duration: 1000,
    color: 'text-blue-400',
    progress: 0
  },
  {
    id: 'neural-network',
    name: 'Neural Network Activation',
    description: 'Connecting to AI evaluation matrix...',
    icon: Brain,
    duration: 1500,
    color: 'text-purple-400',
    progress: 0
  },
  {
    id: 'security-scan',
    name: 'Security Protocols',
    description: 'Running tactical security verification...',
    icon: Shield,
    duration: 800,
    color: 'text-green-400',
    progress: 0
  },
  {
    id: 'targeting',
    name: 'Target Acquisition',
    description: 'Locking onto evaluation parameters...',
    icon: Target,
    duration: 1200,
    color: 'text-orange-400',
    progress: 0
  },
  {
    id: 'power-up',
    name: 'Strategic Power-Up',
    description: 'All systems operational. Ready for combat!',
    icon: Zap,
    duration: 600,
    color: 'text-yellow-400',
    progress: 0
  }
];

const PowerUpLoaderComponent = ({ 
  show, 
  onComplete, 
  customPhases = defaultPhases 
}: PowerUpLoaderProps) => {
  const [currentPhaseIndex, setCurrentPhaseIndex] = useState(0);
  const [phases, setPhases] = useState(customPhases);
  const [globalProgress, setGlobalProgress] = useState(0);
  const [energyBars, setEnergyBars] = useState<number[]>([0, 0, 0, 0, 0, 0, 0, 0]);

  useEffect(() => {
    if (!show) {
      setCurrentPhaseIndex(0);
      setPhases(customPhases.map(phase => ({ ...phase, progress: 0 })));
      setGlobalProgress(0);
      setEnergyBars([0, 0, 0, 0, 0, 0, 0, 0]);
      return;
    }

    const phaseTimeouts: NodeJS.Timeout[] = [];
    const progressIntervals: NodeJS.Timeout[] = [];

    const executePhase = (phaseIndex: number) => {
      if (phaseIndex >= phases.length) {
        if (onComplete) onComplete();
        return;
      }

      setCurrentPhaseIndex(phaseIndex);
      const currentPhase = phases[phaseIndex];

      // Animate energy bars randomly during this phase
      const energyInterval = setInterval(() => {
        setEnergyBars(prev => prev.map(() => Math.random() * 100));
      }, 100);

      // Progress animation for current phase
      const progressInterval = setInterval(() => {
        setPhases(prevPhases => {
          const newPhases = [...prevPhases];
          if (newPhases[phaseIndex].progress < 100) {
            newPhases[phaseIndex].progress += 100 / (currentPhase.duration / 50);
          }
          return newPhases;
        });
      }, 50);

      // Update global progress
      const globalProgressInterval = setInterval(() => {
        const totalProgress = ((phaseIndex + phases[phaseIndex].progress / 100) / phases.length) * 100;
        setGlobalProgress(totalProgress);
      }, 50);

      progressIntervals.push(progressInterval, globalProgressInterval, energyInterval);

      // Move to next phase
      const phaseTimeout = setTimeout(() => {
        clearInterval(progressInterval);
        clearInterval(globalProgressInterval);
        clearInterval(energyInterval);
        
        setPhases(prevPhases => {
          const newPhases = [...prevPhases];
          newPhases[phaseIndex].progress = 100;
          return newPhases;
        });

        executePhase(phaseIndex + 1);
      }, currentPhase.duration);

      phaseTimeouts.push(phaseTimeout);
    };

    executePhase(0);

    return () => {
      phaseTimeouts.forEach(clearTimeout);
      progressIntervals.forEach(clearInterval);
    };
  }, [show, phases.length, onComplete]);

  if (!show) return null;

  const currentPhase = phases[currentPhaseIndex] || phases[phases.length - 1];
  const IconComponent = currentPhase.icon;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="glass-panel p-8 rounded-2xl border border-primary/30 max-w-md w-full mx-4">
        
        {/* Header */}
        <div className="text-center mb-8">
          <div className="glass-subtle p-4 rounded-xl mb-4 inline-block">
            <Activity className="h-8 w-8 text-primary animate-pulse mx-auto" />
          </div>
          <h2 className="heading-display text-2xl text-primary mb-2">Strategic Command Center</h2>
          <p className="text-sm text-muted-foreground">Initializing Combat Systems</p>
        </div>

        {/* Current Phase */}
        <div className="text-center mb-6">
          <div className={`glass-subtle p-6 rounded-xl border border-border/20 ${currentPhase.color.replace('text-', 'border-').replace('-400', '-400/30')}`}>
            <div className="relative">
              {/* Animated Icon */}
              <div className="absolute inset-0 animate-ping opacity-30">
                <IconComponent className={`h-12 w-12 mx-auto ${currentPhase.color}`} />
              </div>
              <IconComponent className={`h-12 w-12 mx-auto ${currentPhase.color} relative z-10`} />
            </div>
            
            <h3 className="heading-refined text-lg mt-4 mb-2">{currentPhase.name}</h3>
            <p className="text-sm text-muted-foreground mb-4">{currentPhase.description}</p>
            
            {/* Phase Progress */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">Phase Progress</span>
                <span className={currentPhase.color}>{Math.round(currentPhase.progress)}%</span>
              </div>
              <div className="w-full bg-muted/20 rounded-full h-2 overflow-hidden">
                <div 
                  className={`h-full rounded-full transition-all duration-300 ${currentPhase.color.replace('text-', 'bg-')}`}
                  style={{ width: `${currentPhase.progress}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Energy Bars */}
        <div className="mb-6">
          <div className="flex justify-between items-center mb-3">
            <span className="text-xs text-muted-foreground">System Energy Levels</span>
            <span className="text-xs text-primary">{Math.round(globalProgress)}% Complete</span>
          </div>
          <div className="flex gap-1 h-6">
            {energyBars.map((level, index) => (
              <div key={index} className="flex-1 bg-muted/20 rounded-sm overflow-hidden">
                <div 
                  className="w-full bg-gradient-to-t from-primary/60 to-primary transition-all duration-300"
                  style={{ height: `${level}%` }}
                />
              </div>
            ))}
          </div>
        </div>

        {/* Phase List */}
        <div className="space-y-2">
          {phases.map((phase, index) => {
            const PhaseIcon = phase.icon;
            const isCompleted = phase.progress === 100;
            const isCurrent = index === currentPhaseIndex;
            const isPending = index > currentPhaseIndex;

            return (
              <div 
                key={phase.id}
                className={`flex items-center gap-3 p-2 rounded-lg transition-all ${
                  isCurrent ? 'bg-primary/10 border border-primary/30' :
                  isCompleted ? 'bg-green-500/10 border border-green-500/30' :
                  'bg-muted/10 border border-muted/20'
                }`}
              >
                <PhaseIcon className={`h-4 w-4 ${
                  isCompleted ? 'text-green-400' :
                  isCurrent ? phase.color :
                  'text-muted-foreground'
                }`} />
                
                <span className={`text-sm flex-1 ${
                  isCompleted ? 'text-green-400' :
                  isCurrent ? 'text-primary' :
                  'text-muted-foreground'
                }`}>
                  {phase.name}
                </span>

                {isCompleted && (
                  <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                )}
                {isCurrent && (
                  <div className="w-2 h-2 rounded-full bg-primary animate-ping" />
                )}
                {isPending && (
                  <div className="w-2 h-2 rounded-full bg-muted-foreground/30" />
                )}
              </div>
            );
          })}
        </div>

        {/* Global Progress */}
        <div className="mt-6">
          <div className="w-full bg-muted/20 rounded-full h-1 overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-primary to-secondary transition-all duration-500"
              style={{ width: `${globalProgress}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export const PowerUpLoader = memo(PowerUpLoaderComponent);